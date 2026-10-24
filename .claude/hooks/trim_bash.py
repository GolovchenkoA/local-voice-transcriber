"""PreToolUse hook for Bash: keeps noisy command output out of Claude's context.

- Blocks starting the server (run.py / run.bat) in the foreground, which never returns.
- Trims output of tests, the WAV engine test and pip to the last lines (exit code kept).
"""

import json
import re
import sys

RULES = [
    # (pattern, extra arguments, lines to keep)
    (re.compile(r"\bpytest\b"), " -q --tb=short --no-header", 60),
    (re.compile(r"transcribe_file\.py"), "", 30),
    (re.compile(r"\bpip\b.*\binstall\b"), " -q", 20),
    (re.compile(r"download_model\.py"), "", 15),
]
SERVER = re.compile(r"python(3|\.exe)?[\"']?\s+(\S*[/\\])?run\.py\b|(^|[\s/\\])run\.bat\b")


def main() -> None:
    data = json.load(sys.stdin)
    tool_input = data.get("tool_input") or {}
    cmd = tool_input.get("command") or ""

    if SERVER.search(cmd) and not tool_input.get("run_in_background"):
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": "The server never exits. Start run.py with run_in_background, "
                                        "check it with curl http://127.0.0.1:8000/api/info, then stop it.",
        }}))
        return

    if not re.search(r"[|>;&]", cmd):  # only simple, single commands
        for pattern, extra, keep in RULES:
            if pattern.search(cmd):
                new = f"set -o pipefail; {cmd}{extra} 2>&1 | tail -n {keep}"
                print(json.dumps({"hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "updatedInput": {**tool_input, "command": new},
                }}))
                return

    print("{}")


if __name__ == "__main__":
    main()
