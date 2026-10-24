"""FastAPI server: static front end, model/info endpoints and the /ws/transcribe WebSocket."""

import asyncio
import json
import queue
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .engine import SAMPLE_RATE, Engine

STATIC_DIR = Path(__file__).parent / "static"


class ModelChoice(BaseModel):
    id: str


def create_app(engine: Engine) -> FastAPI:
    app = FastAPI(title="Local Voice Transcriber")
    busy = {"on": False}  # one recording, or one model load, at a time

    @app.get("/api/info")
    def info() -> dict:
        return engine.info()

    @app.post("/api/model")
    async def switch_model(choice: ModelChoice) -> dict:
        if busy["on"]:
            raise HTTPException(409, "Busy: stop the recording before changing the model.")
        known = {m["id"]: m for m in engine.available_models()}
        if choice.id not in known or not known[choice.id]["downloaded"]:
            raise HTTPException(404, "That model is not downloaded. See the README.")
        busy["on"] = True
        try:
            await asyncio.to_thread(engine.load, choice.id)
        except Exception as exc:
            raise HTTPException(500, f"Could not load the model: {exc}")
        finally:
            busy["on"] = False
        return engine.info()

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.websocket("/ws/transcribe")
    async def transcribe(ws: WebSocket) -> None:
        await ws.accept()
        loop = asyncio.get_running_loop()
        pcm: "queue.Queue[bytes | None]" = queue.Queue()
        out: "asyncio.Queue[dict | None]" = asyncio.Queue()

        def push(item: dict | None) -> None:
            loop.call_soon_threadsafe(out.put_nowait, item)

        def work(language: str | None) -> None:
            try:
                engine.transcribe(pcm, lambda text: push({"type": "text", "delta": text}), language)
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
                        try:
                            language = engine.resolve_language(data.get("language"))
                        except ValueError as exc:
                            await send({"type": "error", "message": str(exc)})
                            break
                        busy["on"] = owns_busy = True
                        max_s = engine.info()["max_seconds"]
                        if max_s:
                            limit_bytes = max_s * SAMPLE_RATE * 2
                        worker = threading.Thread(target=work, args=(language,), daemon=True)
                        worker.start()
                        sender_task = asyncio.create_task(sender())
                        await send({"type": "ready", "streaming": engine.kind == "streaming"})
                    elif kind == "stop":
                        pcm.put(None)
                        if sender_task is not None:
                            await sender_task  # flush the text, then finish
                        break
        except WebSocketDisconnect:
            pass
        finally:
            pcm.put(None)  # lets the worker finish if the browser vanished
            if owns_busy:
                if worker is not None and worker.is_alive():
                    await asyncio.to_thread(worker.join, 60)
                busy["on"] = False
            if sender_task is not None and not sender_task.done():
                sender_task.cancel()
            try:
                await ws.close()
            except RuntimeError:
                pass

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
