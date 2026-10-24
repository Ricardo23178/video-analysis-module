"""Portable offline contract check; requires complete FFmpeg on PATH."""

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "video_analysis.py"
if not MODULE.is_file():
    MODULE = ROOT / "plugins/video-parser/skills/video-parser/scripts/video_analysis.py"
spec = importlib.util.spec_from_file_location("video_analysis", MODULE)
video_analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(video_analysis)


class AnalysisContract(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "Requires ffmpeg and ffprobe on PATH")
    def test_local_video_produces_timed_evidence_without_changing_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                            "color=c=blue:size=160x90:rate=10:duration=2", "-f", "lavfi", "-i",
                            "sine=frequency=440:duration=2", "-c:v", "mpeg4", "-c:a", "aac",
                            "-shortest", str(source)], check=True, capture_output=True)
            source.with_suffix(".srt").write_text("1\n00:00:00,100 --> 00:00:01,000\nHello\n", encoding="utf-8")
            before = source.read_bytes()
            result = video_analysis.analyze_video(source, root / "analysis", max_frames=2)
            self.assertEqual(result["status"], "ready")
            self.assertEqual(len(result["frames"]), 2)
            segments = json.loads(Path(result["transcript"]["segments_path"]).read_text(encoding="utf-8"))
            self.assertEqual(segments, [{"start": 0.1, "end": 1.0, "text": "Hello"}])
            self.assertEqual(source.read_bytes(), before)
            with self.assertRaises(FileExistsError):
                video_analysis.analyze_video(source, root / "analysis")


if __name__ == "__main__":
    unittest.main()
