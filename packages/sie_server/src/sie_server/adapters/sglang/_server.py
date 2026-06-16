"""Shared SGLang server subprocess plumbing.

Used by both the embedding adapter (``embedding.py``) and the generation
adapter (``generation.py``). The two adapters launch the same
``sglang.launch_server`` binary with different flags, but the
port-allocation, subprocess-supervision, health-polling, and termination
patterns are identical — hence this module.

This module deliberately contains no model-specific logic; it just owns the
lifecycle of a single SGLang HTTP server child process.
"""

from __future__ import annotations

import logging
import math
import os
import random
import signal
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# In-process record of ports already handed out by ``find_free_port`` but
# not yet bound by their SGLang child. ``find_free_port`` only confirms a
# port is bindable *now*; the caller then closes the probe socket and the
# child binds it moments later — a TOCTOU window. Two concurrent loads in
# the same worker process could otherwise probe-and-hand-out the same port
# (both probes succeed because neither child has bound yet). Recording the
# handed-out port and excluding it on subsequent calls closes the common
# in-process case. Guarded by ``_RESERVED_PORTS_LOCK`` because loads can
# run from different threads (registry load executor).
_RESERVED_PORTS: set[int] = set()
_RESERVED_PORTS_LOCK = threading.Lock()

# 8B+ models can take 5+ min just to download from HF on a fresh cache,
# plus SGLang itself then loads the model onto the GPU. Override via the
# SGLang-specific env var below, or one of the adapter/server aliases that
# operators commonly try when tuning cold-start budgets.
DEFAULT_STARTUP_TIMEOUT_S = 900.0
STARTUP_TIMEOUT_ENV_VARS = (
    "SIE_SGLANG_STARTUP_TIMEOUT_S",
    "SIE_MODEL_READY_TIMEOUT_S",
    "SIE_ADAPTER_STARTUP_TIMEOUT_S",
    "SIE_SERVER_STARTUP_TIMEOUT_S",
)
LIVENESS_BUDGET_ENV_VAR = "SIE_WORKER_LIVENESS_BUDGET_S"
STARTUP_TIMEOUT_S = DEFAULT_STARTUP_TIMEOUT_S
HEALTH_CHECK_INTERVAL_S = 2.0
BASE_PORT = 30000  # Starting port for SGLang servers

ERR_SERVER_STARTUP = "SGLang server failed to start within timeout"


def _resolve_liveness_budget() -> float | None:
    raw = os.environ.get(LIVENESS_BUDGET_ENV_VAR)
    if raw is None or raw.strip() == "":
        return None
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; expected seconds", LIVENESS_BUDGET_ENV_VAR, raw)
        return None
    if math.isfinite(value) and value > 0:
        return value
    logger.warning("Ignoring invalid %s=%r; expected finite seconds > 0", LIVENESS_BUDGET_ENV_VAR, raw)
    return None


def _validate_liveness_budget(timeout_s: float) -> float:
    budget_s = _resolve_liveness_budget()
    if budget_s is not None and timeout_s >= budget_s:
        msg = (
            f"SGLang startup timeout {timeout_s:g}s must be lower than "
            f"{LIVENESS_BUDGET_ENV_VAR}={budget_s:g}s so kubelet liveness does not kill the worker mid-load"
        )
        raise ValueError(msg)
    return timeout_s


def resolve_startup_timeout(timeout_s: float | None = None) -> float:
    """Resolve the SGLang startup-health timeout.

    Precedence:
    1. Explicit adapter/profile value (``adapter_options.loadtime.startup_timeout_s``).
    2. Environment variables in ``STARTUP_TIMEOUT_ENV_VARS`` order.
    3. ``DEFAULT_STARTUP_TIMEOUT_S``.
    """
    if timeout_s is not None:
        try:
            value = float(timeout_s)
        except (TypeError, ValueError):
            value = 0.0
        if math.isfinite(value) and value > 0:
            return _validate_liveness_budget(value)
        logger.warning("Ignoring invalid SGLang startup timeout override: %r (must be finite > 0)", timeout_s)

    for name in STARTUP_TIMEOUT_ENV_VARS:
        raw = os.environ.get(name)
        if raw is None or raw.strip() == "":
            continue
        try:
            value = float(raw)
        except ValueError:
            logger.warning("Ignoring invalid %s=%r; expected seconds", name, raw)
            continue
        if math.isfinite(value) and value > 0:
            return _validate_liveness_budget(value)
        logger.warning("Ignoring invalid %s=%r; expected finite seconds > 0", name, raw)

    return _validate_liveness_budget(DEFAULT_STARTUP_TIMEOUT_S)


def find_free_port(start_port: int = BASE_PORT) -> int:
    """Find a free port in ``[start_port, start_port + 100)``.

    Mitigates the TOCTOU race between probing a port here and the SGLang
    child binding it later: ports handed out by a previous (not-yet-bound)
    call are excluded via ``_RESERVED_PORTS`` so concurrent in-process
    loads can't both pick the same one. The scan start is also randomized
    within the range so two near-simultaneous calls are unlikely to probe
    the same port in the same order. The race against *external* processes
    (outside this interpreter) remains inherent — there is no way to
    atomically reserve a TCP port without holding it open — but the common
    in-process collision is closed. Reserved ports leak intentionally:
    once handed out a port stays excluded for the process lifetime (the
    range is 100 ports; a worker hosts a handful of SGLang servers).
    """
    span = 100
    offset = random.randrange(span)  # noqa: S311 — port selection, not crypto
    with _RESERVED_PORTS_LOCK:
        for i in range(span):
            port = start_port + ((offset + i) % span)
            if port in _RESERVED_PORTS:
                continue
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("localhost", port))
                except OSError:
                    continue
            _RESERVED_PORTS.add(port)
            return port
    msg = f"Could not find free port in range {start_port}-{start_port + span - 1}"
    raise RuntimeError(msg)


def parse_device_index(device: str) -> int:
    """Parse device index from device string (e.g. ``"cuda:0"`` → ``0``)."""
    if device in {"cuda", "cpu"}:
        return 0
    if device.startswith("cuda:"):
        return int(device.split(":")[1])
    return 0


def open_output_log(prefix: str = "sglang_") -> tempfile._TemporaryFileWrapper:
    """Open a named temp file for capturing subprocess stdout/stderr."""
    return tempfile.NamedTemporaryFile(
        mode="w",
        prefix=prefix,
        suffix=".log",
        delete=False,
    )


def launch_sglang_server(
    cmd: list[str],
    *,
    device_index: int,
    output_file: tempfile._TemporaryFileWrapper,
    extra_env: dict[str, str] | None = None,
) -> subprocess.Popen[bytes]:
    """Launch an SGLang HTTP server subprocess.

    Args:
        cmd: Full argv (must already include ``python -m sglang.launch_server``
            plus all flags).
        device_index: CUDA device index for ``CUDA_VISIBLE_DEVICES``.
        output_file: Temp file open for write — subprocess stdout/stderr is
            redirected here for debugging.
        extra_env: Additional environment variables to set on the subprocess.
            Used by callers that need to set sglang-specific env knobs (e.g.
            ``SGLANG_ENABLE_SPEC_V2=1`` for NEXTN-on-hybrid-architecture
            models like Qwen3.5-4B).

    Returns:
        The ``Popen`` handle. Subprocess is started in a new process group
        (``start_new_session=True``) so the entire group can be signalled on
        shutdown without affecting the parent.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device_index)
    if extra_env:
        env.update(extra_env)
    logger.info("SGLang subprocess output will be logged to: %s", output_file.name)
    return subprocess.Popen(  # noqa: S603 — intentional subprocess call
        cmd,
        env=env,
        stdout=output_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def wait_for_server(
    server_url: str,
    process: subprocess.Popen[bytes],
    *,
    output_file: tempfile._TemporaryFileWrapper | None = None,
    timeout_s: float | None = None,
) -> bool:
    """Poll the SGLang ``/health`` endpoint until the server is ready.

    Returns:
        True if the server reports healthy before the timeout; False if the
        timeout elapses or the subprocess dies. Subprocess output (when
        ``output_file`` is provided) is logged on failure for diagnostics.
    """
    timeout_s = resolve_startup_timeout(timeout_s)
    health_url = f"{server_url}/health"
    start_time = time.monotonic()

    while time.monotonic() - start_time < timeout_s:
        # Check if process died.
        if process.poll() is not None:
            exit_code = process.returncode
            logger.error("SGLang server exited prematurely with code %s", exit_code)
            _log_subprocess_output(output_file)
            return False

        try:
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                return True
        except requests.RequestException:
            # Health endpoint not up yet — keep polling until timeout.
            pass

        time.sleep(HEALTH_CHECK_INTERVAL_S)

    logger.error("SGLang server startup timeout after %ds", timeout_s)
    _log_subprocess_output(output_file)
    return False


def _log_subprocess_output(output_file: tempfile._TemporaryFileWrapper | None) -> None:
    if output_file is None:
        return
    try:
        output_file.flush()
    except Exception:  # noqa: BLE001
        return
    try:
        with Path(output_file.name).open() as f:
            output = f.read()
        logger.error("SGLang subprocess output from %s:\n%s", output_file.name, output[-5000:])
    except OSError as e:
        logger.error("Failed to read SGLang log: %s", e)


def terminate_process(process: subprocess.Popen[bytes] | None) -> None:
    """Terminate the subprocess group: SIGTERM, wait, SIGKILL fallback."""
    if process is None:
        return

    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            process.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass
