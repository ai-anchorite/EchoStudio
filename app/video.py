"""Video handling utilities for Echo TTS Studio.

Pure ffmpeg/ffprobe wrapper for extracting audio, video streams,
and muxing new audio back onto video. No ML dependencies.
"""

import json
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Tuple


def _find_ffmpeg() -> str:
    """Find ffmpeg binary. Checks PATH."""
    path = shutil.which("ffmpeg")
    if path:
        return path
    raise FileNotFoundError(
        "ffmpeg not found on PATH. Install ffmpeg or ensure it's accessible."
    )


def _find_ffprobe() -> str:
    """Find ffprobe binary."""
    path = shutil.which("ffprobe")
    if path:
        return path
    raise FileNotFoundError("ffprobe not found on PATH.")


def get_video_info(video_path: str) -> dict:
    """Get video metadata via ffprobe.

    Returns dict with keys: duration, width, height, fps, has_audio, audio_codec, video_codec.
    """
    ffprobe = _find_ffprobe()
    cmd = [
        ffprobe, "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    data = json.loads(result.stdout)
    info = {
        "duration": float(data.get("format", {}).get("duration", 0)),
        "has_audio": False,
        "width": 0,
        "height": 0,
        "fps": 0,
        "video_codec": "",
        "audio_codec": "",
    }

    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video" and not info["video_codec"]:
            info["video_codec"] = stream.get("codec_name", "")
            info["width"] = int(stream.get("width", 0))
            info["height"] = int(stream.get("height", 0))
            # Parse fps from r_frame_rate (e.g. "30/1")
            fps_str = stream.get("r_frame_rate", "0/1")
            try:
                num, den = fps_str.split("/")
                info["fps"] = round(int(num) / int(den), 2)
            except (ValueError, ZeroDivisionError):
                info["fps"] = 0
        elif stream.get("codec_type") == "audio":
            info["has_audio"] = True
            info["audio_codec"] = stream.get("codec_name", "")

    return info


def extract_audio(video_path: str, output_path: str, sample_rate: int = 44100) -> str:
    """Extract audio from video as WAV.

    Args:
        video_path: Input video file.
        output_path: Where to write the extracted audio.
        sample_rate: Output sample rate.

    Returns:
        Path to the extracted audio file.
    """
    ffmpeg = _find_ffmpeg()
    cmd = [
        ffmpeg, "-y",
        "-i", str(video_path),
        "-vn",  # no video
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",  # mono
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"Audio extraction failed: {result.stderr}")
    return str(output_path)


def mux_audio_to_video(
    video_path: str,
    audio_path: str,
    output_path: str,
    keep_original_audio: bool = False,
) -> str:
    """Replace (or mix) audio track on a video.

    Args:
        video_path: Original video file.
        audio_path: New audio file to use.
        output_path: Where to write the output video.
        keep_original_audio: If True, mix both tracks. If False, replace entirely.

    Returns:
        Path to the output video.
    """
    ffmpeg = _find_ffmpeg()

    if keep_original_audio:
        # Mix original + new audio
        cmd = [
            ffmpeg, "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=first[aout]",
            "-map", "0:v",
            "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            str(output_path),
        ]
    else:
        # Replace audio entirely
        cmd = [
            ffmpeg, "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-map", "0:v",
            "-map", "1:a",
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            str(output_path),
        ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"Muxing failed: {result.stderr}")
    return str(output_path)


def get_duration(file_path: str) -> float:
    """Get duration of any media file in seconds."""
    ffprobe = _find_ffprobe()
    cmd = [
        ffprobe, "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(file_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    return float(result.stdout.strip())


def time_stretch_audio(input_path: str, output_path: str, target_duration: float) -> str:
    """Time-stretch an audio file to match a target duration using ffmpeg.

    Uses the atempo filter which preserves pitch while changing speed.
    Handles extreme ratios by chaining multiple atempo filters (each limited to 0.5-2.0x).

    Args:
        input_path: Source audio file.
        output_path: Where to write the stretched audio.
        target_duration: Desired duration in seconds.

    Returns:
        Path to the output file.
    """
    source_duration = get_duration(input_path)
    if source_duration <= 0 or target_duration <= 0:
        raise ValueError(f"Invalid durations: source={source_duration}, target={target_duration}")

    ratio = source_duration / target_duration  # >1 = speed up, <1 = slow down

    # Clamp to reasonable bounds to avoid extreme distortion
    ratio = max(0.25, min(4.0, ratio))

    # Build atempo filter chain (each filter limited to 0.5-2.0 range)
    filters = []
    remaining = ratio
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    filters.append(f"atempo={remaining:.6f}")

    filter_str = ",".join(filters)

    ffmpeg = _find_ffmpeg()
    cmd = [
        ffmpeg, "-y",
        "-i", str(input_path),
        "-filter:a", filter_str,
        "-vn",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"Time stretch failed: {result.stderr}")
    return str(output_path)


def clip_audio(input_path: str, output_path: str, start: float, end: float, sample_rate: int = 44100) -> str:
    """Clip audio between start and end times (seconds) using ffmpeg.

    Returns path to the clipped audio file.
    """
    ffmpeg = _find_ffmpeg()
    duration = end - start
    if duration <= 0:
        raise ValueError(f"Invalid clip range: {start:.1f}s - {end:.1f}s")
    cmd = [
        ffmpeg, "-y",
        "-i", str(input_path),
        "-ss", f"{start:.3f}",
        "-t", f"{duration:.3f}",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"Clip failed: {result.stderr}")
    return str(output_path)


def trim_silence(input_path: str, output_path: str, threshold_db: int = -40) -> str:
    """Remove leading/trailing silence from audio using ffmpeg silenceremove filter.

    Returns path to the trimmed audio file.
    """
    ffmpeg = _find_ffmpeg()
    # Remove silence from start and end
    af = (
        f"silenceremove=start_periods=1:start_duration=0.05:start_threshold={threshold_db}dB"
        f",areverse"
        f",silenceremove=start_periods=1:start_duration=0.05:start_threshold={threshold_db}dB"
        f",areverse"
    )
    cmd = [
        ffmpeg, "-y",
        "-i", str(input_path),
        "-af", af,
        "-acodec", "pcm_s16le",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"Trim silence failed: {result.stderr}")
    return str(output_path)


def normalize_audio(input_path: str, output_path: str) -> str:
    """Normalize audio volume using ffmpeg loudnorm filter (EBU R128).

    Returns path to the normalized audio file.
    """
    ffmpeg = _find_ffmpeg()
    cmd = [
        ffmpeg, "-y",
        "-i", str(input_path),
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-acodec", "pcm_s16le",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"Normalize failed: {result.stderr}")
    return str(output_path)

