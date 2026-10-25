"""FastAPI server: static front end, model/info endpoints and the /ws/transcribe WebSocket."""

import asyncio
import json
import queue
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .engine import SAMPLE_RATE, Engine

STATIC_DIR = Path(__file__).parent / "static"


class ModelChoice(BaseModel):
    id: str | None = None


def create_app(engine: Engine) -> FastAPI:
    app = FastAPI(title="Local Voice Transcriber")
    busy = {"on": False}  # one recording, or one model load, at a time

    @app.middleware("http")
    async def always_revalidate(request, call_next):
        # The page files are small and local; make the browser check them every time so that
        # after an update it never mixes a new page with an old script.
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/api/info")
    def info() -> dict:
        return engine.info()

    async def run_exclusive(func, *args):
        """Run a slow engine call in a thread while no recording or other load is running."""
        if busy["on"]:
            raise HTTPException(409, "Busy: stop the recording or wait for the model to finish loading.")
        busy["on"] = True
        try:
            await asyncio.to_thread(func, *args)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(500, f"{exc}")
        finally:
            busy["on"] = False

    @app.post("/api/model")
    async def switch_model(choice: ModelChoice) -> dict:
        known = {m["id"]: m for m in engine.available_models()}
        if choice.id not in known or not known[choice.id]["downloaded"]:
            raise HTTPException(404, "That model is not downloaded. See the README.")
        if not known[choice.id]["available"]:
            raise HTTPException(400, f"This model needs the Python package {known[choice.id]['needs']!r}. "
                                     "See requirements-multispeaker.txt.")
        await run_exclusive(engine.load, choice.id)
        return engine.info()

    @app.post("/api/diarizer")
    async def switch_diarizer(choice: ModelChoice) -> dict:
        if not choice.id:  # no diarization model
            await run_exclusive(engine.unload_diarizer)
            return engine.info()
        known = {d["id"]: d for d in engine.available_diarizers()}
        if choice.id not in known or not known[choice.id]["downloaded"]:
            raise HTTPException(404, "That diarization model is not downloaded. See the README.")
        if not known[choice.id]["available"]:
            raise HTTPException(400, f"This model needs the Python package {known[choice.id]['needs']!r}. "
                                     "See requirements-multispeaker.txt.")
        await run_exclusive(engine.load_diarizer, choice.id)
        return engine.info()

    @app.get("/")
    def index() -> HTMLResponse:
        # Add each file's modification time to its address, so a browser can never run an old script
        # (or style sheet) next to a new page.
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        for name in ("app.js", "style.css"):
            stamp = int((STATIC_DIR / name).stat().st_mtime)
            html = html.replace(f"/static/{name}", f"/static/{name}?v={stamp}")
        return HTMLResponse(html)

    @app.websocket("/ws/transcribe")
    async def transcribe(ws: WebSocket) -> None:
        await ws.accept()
        loop = asyncio.get_running_loop()
        pcm: "queue.Queue[bytes | None]" = queue.Queue()
        out: "asyncio.Queue[dict | None]" = asyncio.Queue()

        def push(item: dict | None) -> None:
            loop.call_soon_threadsafe(out.put_nowait, item)

        def work(language: str | None, mode: str) -> None:
            try:
                engine.transcribe(
                    pcm,
                    lambda text: push({"type": "text", "delta": text}),
                    language,
                    mode,
                    lambda turns: push({"type": "turns", "turns": turns}),
                )
                push({"type": "done"})
            except Exception as exc:  # report to the browser instead of dying silently
                push({"type": "error", "message": f"Engine error: {exc}"})
            finally:
                push(None)

        async def sender() -> None:
            while (item := await out.get()) is not None:
                await ws.send_text(json.dumps(item))

        async def send(obj: dict) -> None:
            await ws.send_text(json.dumps(obj))

        worker: threading.Thread | None = None
        sender_task = None
        owns_busy = False
        limit_bytes = None  # offline models stop buffering after a maximum length
        received = 0
        limit_sent = False
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    received += len(msg["bytes"])
                    if limit_bytes is not None and received > limit_bytes:
                        if not limit_sent:
                            limit_sent = True
                            await send({"type": "limit"})
                        continue
                    pcm.put(msg["bytes"])
                elif msg.get("text"):
                    data = json.loads(msg["text"])
                    kind = data.get("type")
                    if kind == "start" and worker is None:
                        if busy["on"]:
                            await send({"type": "error", "message": "Another recording is in progress (busy)."})
                            break
                        mode = data.get("mode") or "single"
                        try:
                            engine.check_mode(mode)
                            language = engine.resolve_language(data.get("language"))
                        except ValueError as exc:
                            await send({"type": "error", "message": str(exc)})
                            break
                        busy["on"] = owns_busy = True
                        max_s = engine.info()["max_seconds"]
                        if max_s:
                            limit_bytes = max_s * SAMPLE_RATE * 2
                        worker = threading.Thread(target=work, args=(language, mode), daemon=True)
                        worker.start()
                        sender_task = asyncio.create_task(sender())
                        await send({"type": "ready", "streaming": engine.kind == "streaming"})
                    elif kind == "stop":
                        pcm.put(None)
                        if sender_task is not None:
                            await sender_task  # flush the result, then finish
                        break
        except WebSocketDisconnect:
            pass
        finally:
            pcm.put(None)  # lets the worker finish if the browser vanished
            if owns_busy:
                if worker is not None and worker.is_alive():
                    await asyncio.to_thread(worker.join, 120)
                busy["on"] = False
            if sender_task is not None and not sender_task.done():
                sender_task.cancel()
            try:
                await ws.close()
            except RuntimeError:
                pass

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
