"""Process and media helpers for the ComfyUI-VidDub node."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
import uuid
import wave
from pathlib import Path
from typing import Callable, Mapping, Sequence


class VidDubNodeError(RuntimeError):
    """Raised when the external VidDub runtime cannot complete a job."""


def resolve_executable(value: str) -> Path:
    """Resolve an absolute executable path or a command available on PATH."""
    expanded = Path(value).expanduser()
    if expanded.is_file():
        # Do not resolve a virtualenv's python symlink: Python locates
        # pyvenv.cfg from the invoked path, so resolving it would silently run
        # the base interpreter without the environment's dependencies.
        return expanded.absolute()
    resolved = shutil.which(value)
    if resolved:
        return Path(resolved).absolute()
    raise VidDubNodeError(
        f"VidDub Python executable not found: {value!r}. "
        "Use the Python binary from the environment where VidDub is installed."
    )


def validate_runtime_paths(
    python_executable: str,
    viddub_root: str,
    weight_root: str,
) -> tuple[Path, Path, Path]:
    python_path = resolve_executable(python_executable)
    project = Path(viddub_root).expanduser().resolve()
    script = project / "scripts" / "infer.py"
    if not script.is_file():
        raise VidDubNodeError(
            f"VidDub inference script not found: {script}. "
            "Set viddub_root to a VidDub repository checkout."
        )
    weights = Path(weight_root).expanduser().resolve()
    if not weights.is_dir():
        raise VidDubNodeError(
            f"VidDub weight root not found: {weights}. "
            "Download the base assets described in the VidDub README."
        )
    return python_path, project, weights


def write_audio_wav(audio: Mapping[str, object], destination: Path) -> None:
    """Write one ComfyUI AUDIO value to a 16-bit PCM WAV file."""
    import numpy as np

    if "waveform" not in audio or "sample_rate" not in audio:
        raise VidDubNodeError("AUDIO input must contain waveform and sample_rate")
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    if sample_rate <= 0:
        raise VidDubNodeError(f"Invalid audio sample rate: {sample_rate}")

    # ComfyUI AUDIO is a torch tensor with shape [batch, channels, samples].
    try:
        values = waveform.detach().to(device="cpu").float().numpy()
    except AttributeError as exc:
        raise VidDubNodeError("AUDIO waveform must be a torch tensor") from exc
    if values.ndim != 3 or values.shape[0] != 1:
        raise VidDubNodeError(
            "VidDub accepts one audio item at a time; expected [1, channels, samples], "
            f"got {tuple(values.shape)}"
        )
    channels = int(values.shape[1])
    if channels not in (1, 2):
        raise VidDubNodeError(f"Expected mono or stereo audio, got {channels} channels")

    pcm = np.rint(np.clip(values[0], -1.0, 1.0) * 32767.0)
    pcm = np.asarray(pcm.T, dtype="<i2").tobytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(destination), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm)


def build_command(
    *,
    python_path: Path,
    project: Path,
    model: str,
    weight_root: Path,
    video: Path,
    audio: Path,
    output: Path,
    seed: int,
    mode: str,
    overlap: int,
    feather: int,
    harmonize: bool,
) -> list[str]:
    if mode not in {"auto", "clip", "windowed"}:
        raise VidDubNodeError(f"Unsupported inference mode: {mode}")
    if overlap < 0 or overlap >= 81:
        raise VidDubNodeError("overlap must be between 0 and 80 frames")
    if feather < 0:
        raise VidDubNodeError("feather must be non-negative")
    if not model.strip():
        raise VidDubNodeError("model must be a local directory or Hugging Face repo ID")

    command = [
        str(python_path),
        str(project / "scripts" / "infer.py"),
        "--model",
        model.strip(),
        "--weight-root",
        str(weight_root),
        "--video",
        str(video),
        "--audio",
        str(audio),
        "--output",
        str(output),
        "--seed",
        str(int(seed)),
        "--mode",
        mode,
        "--overlap",
        str(int(overlap)),
        "--feather",
        str(int(feather)),
    ]
    if not harmonize:
        command.append("--no-color-harmonization")
    return command


def allocate_output_path(output_directory: Path) -> Path:
    folder = output_directory / "viddub"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"VidDub_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.mp4"


def _stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is None:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                check=False,
            )
        else:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                process.kill()
            else:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
    if process.stdout is not None:
        process.stdout.close()


def run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout_seconds: int,
    check_interrupted: Callable[[], None] | None = None,
    environment: Mapping[str, str] | None = None,
) -> str:
    """Run VidDub with cancellation, timeout, and bounded error reporting."""
    if timeout_seconds <= 0:
        raise VidDubNodeError("timeout_seconds must be positive")
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    if environment:
        env.update(environment)
    creationflags = (
        subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    )
    process = subprocess.Popen(
        list(command),
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=os.name != "nt",
        creationflags=creationflags,
    )
    deadline = time.monotonic() + timeout_seconds
    output = ""
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise VidDubNodeError(
                    f"VidDub inference exceeded the {timeout_seconds}-second timeout"
                )
            try:
                output, _ = process.communicate(timeout=min(1.0, remaining))
                break
            except subprocess.TimeoutExpired:
                if check_interrupted is not None:
                    check_interrupted()
    except BaseException:
        _stop_process(process)
        raise

    if process.returncode != 0:
        excerpt = output[-8000:].strip()
        raise VidDubNodeError(
            f"VidDub inference failed with exit code {process.returncode}.\n{excerpt}"
        )
    return output
