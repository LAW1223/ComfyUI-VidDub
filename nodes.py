"""ComfyUI nodes for running VidDub in an isolated Python environment."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import comfy.model_management
import folder_paths
from comfy_api.latest import ComfyExtension, InputImpl, Types, io, ui
from typing_extensions import override

from .viddub_runtime import (
    VidDubNodeError,
    allocate_output_path,
    build_command,
    run_command,
    validate_runtime_paths,
    write_audio_wav,
)


class VidDubVideo(io.ComfyNode):
    """Dub one ComfyUI VIDEO with replacement AUDIO using the VidDub CLI."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VidDubVideo",
            display_name="VidDub Video",
            category="video/dubbing",
            description=(
                "Runs the official VidDub inference CLI in a separate Python "
                "environment. The source must contain one detectable, mostly "
                "front-facing face and at least 81 frames."
            ),
            inputs=[
                io.Video.Input("source_video", tooltip="Source performance and scene."),
                io.Audio.Input(
                    "replacement_audio",
                    tooltip="Replacement speech; VidDub resamples it to 16 kHz.",
                ),
                io.String.Input(
                    "viddub_python",
                    default=os.environ.get("VIDDUB_PYTHON", "python"),
                    tooltip=(
                        "Python executable from the environment where VidDub "
                        "and its requirements are installed."
                    ),
                ),
                io.String.Input(
                    "viddub_root",
                    default=os.environ.get("VIDDUB_ROOT", ""),
                    tooltip="Path to the official VidDub repository checkout.",
                ),
                io.String.Input(
                    "model",
                    default="AIPeanutman/VidDub",
                    tooltip="Local VidDub release directory or Hugging Face repo ID.",
                ),
                io.String.Input(
                    "weight_root",
                    default=os.environ.get("VIDDUB_WEIGHT_ROOT", ""),
                    tooltip=(
                        "Directory containing Wan2.1-VACE-1.3B and whisper-small."
                    ),
                ),
                io.Combo.Input(
                    "mode",
                    options=["auto", "clip", "windowed"],
                    default="auto",
                    tooltip=(
                        "Auto uses one call for exactly 81 frames and overlapping "
                        "windows for longer videos."
                    ),
                ),
                io.Int.Input("seed", default=0, min=0, max=0x7FFFFFFF),
                io.Int.Input(
                    "overlap",
                    default=17,
                    min=0,
                    max=80,
                    tooltip="Shared frames in windowed mode; the paper setting is 17.",
                ),
                io.Int.Input(
                    "feather",
                    default=15,
                    min=0,
                    max=128,
                    tooltip="Mouth-region paste-back feather radius.",
                ),
                io.Boolean.Input(
                    "color_harmonization",
                    default=True,
                    tooltip="Apply the released lower-face color correction.",
                ),
                io.String.Input(
                    "device",
                    default="cuda",
                    tooltip="Torch device passed to VidDub, for example cuda or cuda:0.",
                ),
                io.Int.Input(
                    "timeout_seconds",
                    default=7200,
                    min=60,
                    max=86400,
                    tooltip="Terminate external inference after this many seconds.",
                ),
            ],
            outputs=[
                io.Video.Output("dubbed_video"),
                io.String.Output("output_path"),
            ],
            is_output_node=True,
            has_intermediate_output=True,
        )

    @classmethod
    def execute(
        cls,
        source_video,
        replacement_audio,
        viddub_python,
        viddub_root,
        model,
        weight_root,
        mode,
        seed,
        overlap,
        feather,
        color_harmonization,
        device,
        timeout_seconds,
    ):
        if not str(viddub_root).strip():
            raise VidDubNodeError(
                "viddub_root is empty. Set it to the official VidDub checkout."
            )
        if not str(weight_root).strip():
            weight_root = str(Path(viddub_root).expanduser() / "weights")
        python_path, project, weights = validate_runtime_paths(
            str(viddub_python), str(viddub_root), str(weight_root)
        )

        output_path = allocate_output_path(
            Path(folder_paths.get_output_directory()).resolve()
        )
        with tempfile.TemporaryDirectory(prefix="comfyui-viddub-") as temporary:
            temporary_path = Path(temporary)
            source_path = temporary_path / "source.mp4"
            audio_path = temporary_path / "replacement.wav"
            source_video.save_to(
                str(source_path),
                format=Types.VideoContainer.MP4,
                codec=Types.VideoCodec.H264,
                preset="ultrafast",
            )
            write_audio_wav(replacement_audio, audio_path)
            command = build_command(
                python_path=python_path,
                project=project,
                model=str(model),
                weight_root=weights,
                video=source_path,
                audio=audio_path,
                output=output_path,
                seed=int(seed),
                mode=str(mode),
                overlap=int(overlap),
                feather=int(feather),
                harmonize=bool(color_harmonization),
            )
            command.extend(["--device", str(device)])

            # VidDub runs in another process. Release ComfyUI model allocations
            # first so both runtimes do not compete for the same VRAM.
            comfy.model_management.unload_all_models()
            comfy.model_management.soft_empty_cache()
            try:
                log = run_command(
                    command,
                    cwd=project,
                    timeout_seconds=int(timeout_seconds),
                    check_interrupted=(
                        comfy.model_management.throw_exception_if_processing_interrupted
                    ),
                )
            except BaseException:
                output_path.unlink(missing_ok=True)
                raise
            if log.strip():
                print(log.rstrip())

        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise VidDubNodeError(
                f"VidDub exited successfully but did not write a video: {output_path}"
            )
        result = InputImpl.VideoFromFile(str(output_path))
        preview = ui.PreviewVideo(
            [
                ui.SavedResult(
                    output_path.name,
                    output_path.parent.name,
                    io.FolderType.output,
                )
            ]
        )
        return io.NodeOutput(result, str(output_path), ui=preview)


class VidDubExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [VidDubVideo]


async def comfy_entrypoint() -> VidDubExtension:
    return VidDubExtension()
