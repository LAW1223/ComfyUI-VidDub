# ComfyUI-VidDub

ComfyUI custom node for the official
[VidDub](https://github.com/LAW1223/VidDub) full-context two-step video-dubbing
model.

The node accepts ComfyUI `VIDEO` and `AUDIO` values, runs the official VidDub
CLI in an isolated Python environment, and returns a dubbed `VIDEO`. Keeping
VidDub in a separate environment avoids replacing ComfyUI's PyTorch and Wan
dependencies.

> [!IMPORTANT]
> VidDub is currently released and tested for Linux with an NVIDIA CUDA GPU.
> For Windows, run both ComfyUI and VidDub inside WSL2/Ubuntu. Native Windows
> execution has not been validated.

## What the node does

```text
Load Video ───────────────┐
                          ├─> VidDub Video ─> Save Video
Load Audio (new speech) ──┘
```

`VidDub Video`:

- materializes the incoming ComfyUI video and audio into a temporary job;
- releases ComfyUI model allocations before external inference;
- calls `VidDub/scripts/infer.py` without a shell;
- supports `auto`, exact 81-frame `clip`, and long-form `windowed` modes;
- returns a native ComfyUI `VIDEO` and the saved output path;
- stops the external process on ComfyUI cancellation or timeout.

## Requirements

- A current ComfyUI release with built-in `VIDEO` support
- Linux, or WSL2 with ComfyUI running inside WSL2
- Python 3.10+
- FFmpeg/ffprobe
- NVIDIA CUDA GPU with enough memory for VidDub/Wan2.1-VACE
- A separate working [VidDub](https://github.com/LAW1223/VidDub) installation

## Installation

### 1. Prepare VidDub separately

Follow the official VidDub README. A typical layout is:

```bash
git clone https://github.com/LAW1223/VidDub.git ~/VidDub
python3 -m venv ~/viddub-venv
source ~/viddub-venv/bin/activate
pip install -r ~/VidDub/requirements.txt

mkdir -p ~/VidDub/weights
huggingface-cli download Wan-AI/Wan2.1-VACE-1.3B \
  --local-dir ~/VidDub/weights/Wan2.1-VACE-1.3B
huggingface-cli download openai/whisper-small \
  --local-dir ~/VidDub/weights/whisper-small
huggingface-cli download AIPeanutman/VidDub \
  --local-dir ~/VidDub/weights/VidDub
```

Verify the CLI once before using ComfyUI:

```bash
~/viddub-venv/bin/python ~/VidDub/scripts/infer.py \
  --model ~/VidDub/weights/VidDub \
  --weight-root ~/VidDub/weights \
  --video input.mp4 \
  --audio replacement.wav \
  --output output.mp4
```

### 2. Install the custom node

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/LAW1223/ComfyUI-VidDub.git
```

Restart ComfyUI. The node appears under **video → dubbing → VidDub Video**.
The custom node has no additional pip dependencies beyond current ComfyUI.

### 3. Configure the node

| Input | Example | Meaning |
|---|---|---|
| `viddub_python` | `/home/user/viddub-venv/bin/python` | Python executable with VidDub installed |
| `viddub_root` | `/home/user/VidDub` | Official VidDub repository checkout |
| `model` | `/home/user/VidDub/weights/VidDub` | Local model directory or `AIPeanutman/VidDub` |
| `weight_root` | `/home/user/VidDub/weights` | Contains Wan2.1-VACE and Whisper assets |
| `mode` | `auto` | One call for exactly 81 frames; windows otherwise |
| `seed` | `0` | Base sampling seed; window seeds advance by index |
| `overlap` | `17` | Shared frames in windowed mode |
| `feather` | `15` | Mouth-region paste-back feather radius |
| `color_harmonization` | `true` | Released lower-face color correction |
| `device` | `cuda` | Torch device used by the external VidDub process |
| `timeout_seconds` | `7200` | Maximum runtime for one node execution |

The following environment variables can provide UI defaults:

```bash
export VIDDUB_PYTHON=/home/user/viddub-venv/bin/python
export VIDDUB_ROOT=/home/user/VidDub
export VIDDUB_WEIGHT_ROOT=/home/user/VidDub/weights
```

For API use, copy `examples/api_workflow.json`, place `source.mp4` and
`replacement.wav` in ComfyUI's input directory, and replace the example paths.

## Windows

The supported route is:

1. enable WSL2 and install Ubuntu;
2. install the NVIDIA WSL driver;
3. install and run ComfyUI inside WSL2;
4. install VidDub in a separate WSL2 virtual environment;
5. open the ComfyUI web interface from the Windows browser.

A native Windows ComfyUI process cannot currently delegate to a WSL VidDub
environment through this node. Supporting that safely requires a dedicated
WSL path/argument bridge and remains future work.

## Scope and limitations

- The source must contain one detectable, mostly front-facing face and at
  least 81 frames.
- Extreme pose, occlusion, multiple faces, and severe source artifacts can
  fail.
- Long videos use offline overlapping-window inference; this is not a
  real-time or streaming node.
- VidDub is loaded in a new external process for each execution. This protects
  ComfyUI's environment but repeats model-loading cost.
- Input `VIDEO` values may be materialized as temporary H.264 MP4 files before
  inference.
- ComfyUI's interrupt button is checked once per second while VidDub runs.

## Development

Runtime helpers are intentionally independent from ComfyUI:

```bash
python -m unittest discover -s tests -v
python -m compileall -q .
```

The node targets ComfyUI's V3 custom-node API and first-class `VIDEO` type.

## Responsible use and licenses

Use VidDub only with media, identities, voices, and outputs for which you have
the required permissions. Do not use it for impersonation, fraud, harassment,
non-consensual content, or deceptive political or media manipulation.

This custom-node wrapper is available under the [Apache License 2.0](LICENSE).
VidDub code and model components remain governed by the
[VidDub Individual Commercial and Research License](https://github.com/LAW1223/VidDub/blob/main/LICENSE).
The wrapper license does not override VidDub or third-party model terms.
In particular, exposing VidDub inference to third parties through a hosted
service requires separate written permission under the VidDub license.
