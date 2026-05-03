"""Streamlit subprocess lifecycle manager for e2e tests."""

from __future__ import annotations

import contextlib
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Generator


STARTUP_TIMEOUT = 30.0
KILL_TIMEOUT = 5.0
STARTUP_OUTPUT_TAIL = 30


def _find_free_port() -> int:
    """Bind to port 0 and return the ephemeral port number."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _shutdown(proc: subprocess.Popen[str]) -> None:
    """Gracefully terminate *proc*, killing it if necessary."""
    proc.terminate()
    try:
        proc.wait(timeout=KILL_TIMEOUT)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _read_stdout(
    proc: subprocess.Popen[str],
    lines: list[str],
    stop_event: threading.Event,
) -> None:
    """Background thread: read stdout lines into *lines* until *stop_event*."""
    if proc.stdout is None:
        return
    for line in iter(proc.stdout.readline, ""):
        if stop_event.is_set():
            break
        if line:
            lines.append(line)


@contextlib.contextmanager
def streamlit_server(
    entrypoint: str | Path = Path(__file__).with_name("run_app.py"),
) -> Generator[str, None, None]:
    """Start a Streamlit app in a subprocess and yield its URL.

    Usage:
        with streamlit_server() as url:
            page.goto(url)

    The subprocess is terminated (and killed if necessary) on exit.
    """
    port = _find_free_port()
    url = f"http://localhost:{port}"

    cmd = [
        sys.executable,
        "-m", "streamlit", "run",
        str(entrypoint),
        "--server.port", str(port),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    lines: list[str] = []
    stop_event = threading.Event()
    reader = threading.Thread(
        target=_read_stdout, args=(proc, lines, stop_event), daemon=True
    )
    reader.start()

    ready = False
    start = time.time()
    while time.time() - start < STARTUP_TIMEOUT:
        if any("You can now view your Streamlit app" in ln for ln in lines):
            ready = True
            break
        if proc.poll() is not None:
            break
        time.sleep(0.5)

    if not ready:
        _shutdown(proc)
        stop_event.set()
        reader.join(timeout=2.0)
        tail = "".join(lines[-STARTUP_OUTPUT_TAIL:])
        raise RuntimeError(
            f"Streamlit did not start within {STARTUP_TIMEOUT}s. "
            f"Exit code: {proc.returncode}\n"
            f"--- Last {STARTUP_OUTPUT_TAIL} lines of output ---\n"
            f"{tail}"
        )

    try:
        yield url
    finally:
        _shutdown(proc)
        stop_event.set()
        reader.join(timeout=2.0)
