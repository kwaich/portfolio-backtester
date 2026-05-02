"""Streamlit subprocess lifecycle manager for e2e tests."""

from __future__ import annotations

import contextlib
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Generator


# Timeout constants
STARTUP_TIMEOUT = 30.0  # seconds to wait for "You can now view"
KILL_TIMEOUT = 5.0      # seconds to wait after terminate() before kill()
HEALTH_POLL_INTERVAL = 0.5  # seconds between stdout polls


def _find_free_port() -> int:
    """Bind to port 0 and return the ephemeral port number."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@contextlib.contextmanager
def streamlit_server(entrypoint: str | Path = "e2e/run_app.py") -> Generator[str, None, None]:
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

    # Poll stdout for the "ready" signal
    ready = False
    start = time.time()
    while time.time() - start < STARTUP_TIMEOUT:
        assert proc.stdout is not None
        line = proc.stdout.readline()
        if line and "You can now view your Streamlit app" in line:
            ready = True
            break
        if proc.poll() is not None:
            break
        time.sleep(HEALTH_POLL_INTERVAL)

    if not ready:
        proc.terminate()
        try:
            proc.wait(timeout=KILL_TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        raise RuntimeError(
            f"Streamlit did not start within {STARTUP_TIMEOUT}s. "
            f"Exit code: {proc.returncode}"
        )

    try:
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=KILL_TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
