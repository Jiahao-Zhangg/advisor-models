"""vLLM server helpers used by evaluation scripts.

This module intentionally stays lightweight and only depends on the standard
library so it can run in the same environment as the evaluation entrypoints.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Optional


def _is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) == 0


def _wait_for_openai_server(
    *,
    host: str,
    port: int,
    timeout_s: int,
    poll_interval_s: float,
    process: subprocess.Popen,
) -> None:
    """Wait until the vLLM OpenAI server responds to `/v1/models`."""
    url = f"http://{host}:{port}/v1/models"
    deadline = time.time() + timeout_s

    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"vLLM server process exited early with code {process.returncode}."
            )
        try:
            with urllib.request.urlopen(url, timeout=0.5) as resp:
                if resp.status != 200:
                    time.sleep(poll_interval_s)
                    continue
                body = resp.read().decode("utf-8", errors="replace")
                json.loads(body)
                return
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            time.sleep(poll_interval_s)

    raise TimeoutError(f"Timed out waiting for vLLM server to become ready at {url}.")


def start_vllm_server(
    *,
    model_to_serve_name: str,
    served_model_name: str = "advisor_model",
    host: str = "127.0.0.1",
    port: int = 8000,
    tensor_parallel_size: int = 1,
    max_model_len: int = 32768,
    gpu_memory_utilization: float = 0.90,
    trust_remote_code: bool = True,
    dtype: Optional[str] = None,
    timeout_s: int = 300,
) -> subprocess.Popen:
    """Start a local vLLM OpenAI-compatible server and wait until it is ready.

    Returns the `subprocess.Popen` handle so callers can terminate it.
    """
    if _is_port_open(host, port):
        raise RuntimeError(
            f"Port {port} on {host} is already in use; stop the existing server or pick a different port."
        )

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_to_serve_name,
        "--served-model-name",
        served_model_name,
        "--host",
        host,
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
    ]

    if trust_remote_code:
        cmd.append("--trust-remote-code")

    if dtype is not None:
        cmd.extend(["--dtype", dtype])

    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=None,
        stderr=None,
    )

    try:
        _wait_for_openai_server(
            host=host,
            port=port,
            timeout_s=timeout_s,
            poll_interval_s=0.5,
            process=process,
        )
        return process
    except Exception:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        raise
