"""Analyze a local video without downloading or modifying its source.

Python API: analyze_video(video_path, output_dir, max_frames=12).
CLI: python video_analysis.py VIDEO --output ANALYSIS_DIR
"""

from __future__ import annotations

import argparse
from html import unescape
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

_WHISPER_DLL_HANDLES: list[object] = []


def whisper_cuda_library_dirs() -> list[Path]:
    """Find CUDA DLL directories installed by the NVIDIA Python wheels on Windows."""

    if sys.platform != "win32":
        return []
    site_packages = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    directories = [
        site_packages / "cublas" / "bin",
        site_packages / "cudnn" / "bin",
        site_packages / "cuda_nvrtc" / "bin",
    ]
    return [directory for directory in directories if directory.is_dir()]


def _configure_whisper_cuda_dll_search() -> None:
    """Make NVIDIA wheel DLLs discoverable before importing faster-whisper."""

    directories = whisper_cuda_library_dirs()
    if not directories or not hasattr(os, "add_dll_directory"):
        return
    for directory in directories:
        directory_text = str(directory)
        _WHISPER_DLL_HANDLES.append(os.add_dll_directory(directory_text))
        path_entries = os.environ.get("PATH", "").split(os.pathsep)
        if directory_text not in path_entries:
            os.environ["PATH"] = os.pathsep.join([directory_text, *path_entries])


def write_manifest(path: Path, payload: dict | list) -> None:
    """Write an analysis manifest as UTF-8 JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_command(args: list[str]) -> str:
    """Run a fixed external media command and return UTF-8 stdout."""

    resolved_args = list(args)
    if not shutil.which(args[0]):
        local_binary = local_media_binary(args[0])
        if local_binary is not None:
            resolved_args[0] = str(local_binary)
    try:
        completed = subprocess.run(
            resolved_args,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required media command is unavailable: {args[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{args[0]} timed out after 600 seconds") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip().splitlines()[-1:] or ["command failed"]
        raise RuntimeError(f"{args[0]} failed: {detail[0]}") from exc
    return completed.stdout


def local_media_binary(name: str) -> Path | None:
    """Find a user-local FFmpeg binary installed by the Skill bootstrap."""

    suffix = ".exe" if sys.platform == "win32" else ""
    candidate = Path.home() / ".codex" / "tools" / "ffmpeg" / f"{name}{suffix}"
    return candidate if candidate.is_file() else None


def _parse_ratio(value: object) -> float | None:
    try:
        numerator, denominator = str(value).split("/", 1)
        return round(float(numerator) / float(denominator), 3) if float(denominator) else None
    except (ValueError, TypeError, ZeroDivisionError):
        return None


def probe_media(video_path: Path) -> dict[str, object]:
    """Read stable media metadata through ffprobe."""

    raw = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration,start_time:stream=codec_type,codec_name,width,height,r_frame_rate,duration,start_time",
            "-of",
            "json",
            str(video_path),
        ]
    )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("ffprobe returned invalid JSON") from exc
    streams = payload.get("streams") or []
    video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), {})
    if not video_stream:
        raise ValueError("Input has no video stream")
    audio_stream = next((stream for stream in streams if stream.get("codec_type") == "audio"), {})
    duration_value = (payload.get("format") or {}).get("duration")
    try:
        duration = round(float(duration_value), 3) if duration_value is not None else 0.0
    except (TypeError, ValueError):
        duration = 0.0
    try:
        video_duration = float(video_stream.get("duration") or duration)
        video_start = max(0.0, float(video_stream.get("start_time") or 0.0)
                          - float((payload.get("format") or {}).get("start_time") or 0.0))
    except (TypeError, ValueError):
        video_duration, video_start = duration, 0.0
    result: dict[str, object] = {
        "duration": duration,
        "video_duration": video_duration,
        "video_start": video_start,
        "width": video_stream.get("width"),
        "height": video_stream.get("height"),
        "fps": _parse_ratio(video_stream.get("r_frame_rate")),
        "video_codec": video_stream.get("codec_name"),
        "audio_codec": audio_stream.get("codec_name"),
        "has_audio": bool(audio_stream),
    }
    return {key: value for key, value in result.items() if value is not None}


def extract_audio(video_path: Path, audio_path: Path) -> None:
    """Extract a model-friendly mono 16 kHz WAV file."""

    audio_path.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-af",
            "aresample=async=1:first_pts=0",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(audio_path),
        ]
    )


def sample_frame_times(duration: float, max_frames: int = 12) -> list[float]:
    """Sample the whole video in chronological order, avoiding the EOF boundary."""
    if max_frames < 1 or not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration and max_frames must be positive and finite")
    if max_frames == 1:
        return [0.0]
    end = max(0.0, duration - 0.1)
    return [round(index * end / (max_frames - 1), 6) for index in range(max_frames)]


def extract_frames(video_path: Path, frames_dir: Path, duration: float, max_frames: int, start: float = 0.0):
    frames_dir.mkdir()
    for index, timestamp in enumerate(sample_frame_times(duration, max_frames), 1):
        timestamp = round(start + timestamp, 6)
        path = frames_dir / f"frame-{index:03d}-{timestamp:.3f}s.jpg"
        run_command(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", str(timestamp),
                     "-i", str(video_path), "-frames:v", "1", "-q:v", "2", str(path)])
        if not path.is_file() or not path.stat().st_size:
            raise RuntimeError(f"No frame produced at {timestamp:.3f}s")
        yield {"path": str(path), "time_seconds": timestamp}


_CUE = re.compile(r"(\d+:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d+:\d{2}:\d{2}[,.]\d{3})[^\n]*\n(.*?)(?=\n\s*\n|\Z)", re.S)


def parse_subtitle_segments(srt: str) -> list[dict]:
    """Parse FFmpeg's normalized SRT, keeping cue timing and readable text."""
    def seconds(value):
        hours, minutes, secs = value.replace(",", ".").split(":")
        return round(int(hours) * 3600 + int(minutes) * 60 + float(secs), 3)

    segments = []
    for match in _CUE.finditer(srt.replace("\r\n", "\n")):
        text = unescape(re.sub(r"<[^>]*>|\{[^}]*\}", "", match[3])).strip()
        start, end = seconds(match[1]), seconds(match[2])
        if text and end >= start:
            segments.append({"start": start, "end": end, "text": text})
    return segments


def find_or_create_transcript(task_dir: Path, audio_path: Path, *, subtitle_path: Path | None = None,
                              whisper_model: str = "small", transcribe: bool = True) -> dict:
    """Prefer a supplied subtitle; otherwise use optional local faster-whisper."""
    text_path = task_dir / "transcript.txt"
    segments_path = task_dir / "transcript_segments.json"
    result = {"status": "unavailable", "path": str(text_path), "segments_path": str(segments_path)}
    segments = []
    if subtitle_path is not None:
        try:
            normalized = run_command(["ffmpeg", "-nostdin", "-v", "error", "-i",
                                      str(subtitle_path), "-f", "srt", "-"])
            segments = parse_subtitle_segments(normalized)
            if segments:
                result.update(status="subtitle", source=str(subtitle_path))
            else:
                result["subtitle_error"] = "Subtitle contains no timed text"
        except RuntimeError as exc:
            result["subtitle_error"] = str(exc)
    if not segments:
        if not transcribe:
            result["reason"] = "Local transcription was disabled"
        elif not audio_path.is_file():
            result["reason"] = "No usable audio was extracted"
        else:
            try:
                _configure_whisper_cuda_dll_search()
                from faster_whisper import WhisperModel
                model = WhisperModel(whisper_model, device="auto", compute_type="int8")
                generated, _ = model.transcribe(str(audio_path), vad_filter=True)
                segments = [{"start": round(float(part.start), 3), "end": round(float(part.end), 3),
                             "text": part.text.strip()} for part in generated if part.text.strip()]
                if segments:
                    result.update(status="whisper", model=whisper_model)
                else:
                    result["reason"] = "No speech was recognized; the transcript is empty"
            except Exception as exc:
                result["reason"] = f"Local transcription unavailable: {exc}"
    text_path.write_text("\n".join(part["text"] for part in segments), encoding="utf-8")
    write_manifest(segments_path, segments)
    return result


def analyze_video(video_path: str | Path, output_dir: str | Path, *, max_frames: int = 12,
                  subtitle_path: str | Path | None = None, whisper_model: str = "small",
                  transcribe: bool = True) -> dict:
    """Build a new analysis directory. Never overwrite or modify the input video."""
    video_path = Path(video_path).expanduser().resolve(strict=True)
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Analysis directory already exists: {output_dir}")
    if not video_path.is_file():
        raise ValueError("Input must be a video file")
    if max_frames < 1:
        raise ValueError("max_frames must be greater than zero")
    media = probe_media(video_path)
    sample_frame_times(float(media["video_duration"]), max_frames)
    if subtitle_path is not None:
        subtitle_path = Path(subtitle_path).expanduser().resolve(strict=True)
    else:
        subtitles = [p for p in video_path.parent.iterdir() if p.is_file()
                     and p.suffix.lower() in {".vtt", ".srt", ".ass"}
                     and (p.stem == video_path.stem or p.stem.startswith(video_path.stem + "."))]
        subtitles.sort(key=lambda p: (0 if any(s in p.name.lower() for s in [".zh", ".chi"]) else 1, p.name))
        subtitle_path = subtitles[0] if subtitles else None
    output_dir.mkdir(parents=True, exist_ok=False)
    errors = {}
    audio_path = output_dir / "audio.wav"
    audio = None
    if media.get("has_audio"):
        try:
            extract_audio(video_path, audio_path)
            if not audio_path.is_file() or audio_path.stat().st_size <= 44:
                raise RuntimeError("No usable audio was produced")
            audio = str(audio_path)
        except RuntimeError as exc:
            audio_path.unlink(missing_ok=True)
            errors["audio"] = str(exc)
    else:
        errors["audio"] = "Source video has no audio stream"
    frames = []
    try:
        frames.extend(extract_frames(video_path, output_dir / "frames", float(media["video_duration"]),
                                     max_frames, float(media["video_start"])))
    except RuntimeError as exc:
        errors["frames"] = str(exc)
    transcript = find_or_create_transcript(output_dir, audio_path, subtitle_path=subtitle_path,
                                           whisper_model=whisper_model, transcribe=transcribe)
    if transcript["status"] == "unavailable":
        errors["transcript"] = transcript["reason"]
    result = {
        "schema_version": 1,
        "status": "partial" if errors else "ready",
        "analysis_dir": str(output_dir), "analysis_json": str(output_dir / "analysis.json"),
        "media_path": str(video_path), "media": media, "audio": audio, "frames": frames,
        "transcript": transcript, "degraded_modalities": list(errors), "errors": errors,
    }
    if not frames and transcript["status"] == "unavailable":
        result["status"] = "unavailable"
    write_manifest(output_dir / "analysis.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze a local video; leave source files untouched")
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--subtitle", type=Path)
    parser.add_argument("--max-frames", type=int, default=12)
    parser.add_argument("--whisper-model", default="small")
    parser.add_argument("--skip-transcription", action="store_true")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    try:
        result = analyze_video(args.video, args.output, max_frames=args.max_frames,
                               subtitle_path=args.subtitle, whisper_model=args.whisper_model,
                               transcribe=not args.skip_transcription)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Video analysis failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result["status"] == "unavailable" else 0


if __name__ == "__main__":
    raise SystemExit(main())
