import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from viddub_runtime import (
    VidDubNodeError,
    allocate_output_path,
    build_command,
    run_command,
    validate_runtime_paths,
    write_audio_wav,
)


class FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)

    def detach(self):
        return self

    def to(self, **_kwargs):
        return self

    def float(self):
        return self

    def numpy(self):
        return self.values


def make_runtime(root):
    project = root / "VidDub"
    script = project / "scripts" / "infer.py"
    script.parent.mkdir(parents=True)
    script.write_text("print('stub')\n")
    weights = project / "weights"
    weights.mkdir()
    return project, weights


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_validate_runtime_paths(self):
        project, weights = make_runtime(self.root)
        python, validated_project, validated_weights = validate_runtime_paths(
            sys.executable, str(project), str(weights)
        )
        self.assertEqual(python, Path(sys.executable).absolute())
        self.assertEqual(validated_project, project.resolve())
        self.assertEqual(validated_weights, weights.resolve())

    def test_validate_runtime_paths_rejects_wrong_checkout(self):
        weights = self.root / "weights"
        weights.mkdir()
        with self.assertRaisesRegex(VidDubNodeError, "inference script not found"):
            validate_runtime_paths(sys.executable, str(self.root), str(weights))

    def test_virtualenv_python_symlink_is_not_resolved(self):
        project, weights = make_runtime(self.root)
        fake_python = self.root / "venv" / "bin" / "python"
        fake_python.parent.mkdir(parents=True)
        fake_python.symlink_to(sys.executable)
        python, _, _ = validate_runtime_paths(
            str(fake_python), str(project), str(weights)
        )
        self.assertEqual(python, fake_python.absolute())

    def test_build_command_uses_argument_list(self):
        project, weights = make_runtime(self.root)
        command = build_command(
            python_path=Path(sys.executable),
            project=project,
            model="AIPeanutman/VidDub",
            weight_root=weights,
            video=self.root / "source.mp4",
            audio=self.root / "speech.wav",
            output=self.root / "result.mp4",
            seed=7,
            mode="windowed",
            overlap=17,
            feather=15,
            harmonize=False,
        )
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[1], str(project / "scripts" / "infer.py"))
        self.assertEqual(command[command.index("--seed") + 1], "7")
        self.assertEqual(command[command.index("--mode") + 1], "windowed")
        self.assertEqual(command[-1], "--no-color-harmonization")

    def test_build_command_validates_window_options(self):
        project, weights = make_runtime(self.root)
        with self.assertRaisesRegex(VidDubNodeError, "overlap"):
            build_command(
                python_path=Path(sys.executable),
                project=project,
                model="model",
                weight_root=weights,
                video=self.root / "source.mp4",
                audio=self.root / "speech.wav",
                output=self.root / "result.mp4",
                seed=0,
                mode="auto",
                overlap=81,
                feather=15,
                harmonize=True,
            )

    def test_write_audio_wav(self):
        destination = self.root / "audio.wav"
        write_audio_wav(
            {
                "waveform": FakeTensor([[[0.0, 0.5, -0.5, 1.0]]]),
                "sample_rate": 16000,
            },
            destination,
        )
        with wave.open(str(destination), "rb") as audio:
            self.assertEqual(audio.getnchannels(), 1)
            self.assertEqual(audio.getframerate(), 16000)
            self.assertEqual(audio.getnframes(), 4)

    def test_run_command_reports_success_and_failure(self):
        output = run_command(
            [sys.executable, "-c", "print('finished')"],
            cwd=self.root,
            timeout_seconds=10,
        )
        self.assertIn("finished", output)

        with self.assertRaisesRegex(VidDubNodeError, "exit code 3"):
            run_command(
                [sys.executable, "-c", "print('bad job'); raise SystemExit(3)"],
                cwd=self.root,
                timeout_seconds=10,
            )

    def test_run_command_timeout(self):
        with self.assertRaisesRegex(VidDubNodeError, "timeout"):
            run_command(
                [sys.executable, "-c", "import time; time.sleep(5)"],
                cwd=self.root,
                timeout_seconds=1,
            )

    def test_allocate_output_path(self):
        first = allocate_output_path(self.root)
        second = allocate_output_path(self.root)
        self.assertEqual(first.parent, self.root / "viddub")
        self.assertEqual(first.suffix, ".mp4")
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
