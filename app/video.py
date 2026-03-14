"""Video handling utilities for Echo TTS Studio.

Pure ffmpeg/ffprobe wrapper for extracting audio, video streams,
and muxing new audio back onto video. No ML dependencies.
"""
import os
import re
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


def trim_silence(input_path: str, output_path: str, threshold_db: int = -40, stop_duration: float = 0.5, remove_internal: bool = True) -> str:
    """Remove silence from audio using ffmpeg silenceremove filter.

    Args:
        input_path: Source audio file.
        output_path: Where to write the trimmed audio.
        threshold_db: Silence threshold in dB (default -40).
        stop_duration: Minimum silence duration to remove in seconds (default 0.5).
        remove_internal: If True, removes silence between sentences/segments. If False, only removes leading/trailing.

    Returns path to the trimmed audio file.
    """
    ffmpeg = _find_ffmpeg()
    
    if remove_internal:
        # Remove all silence periods (both start and stop)
        # stop_periods=-1 removes all silence periods between audio
        # start_periods=1 removes leading silence
        af = f"silenceremove=start_periods=1:start_duration=0.05:start_threshold={threshold_db}dB:stop_periods=-1:stop_duration={stop_duration}:stop_threshold={threshold_db}dB"
    else:
        # Remove only leading/trailing silence
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


# def normalize_audio(input_path: str, output_path: str) -> str:
    # """Normalize audio volume using a 2-pass ffmpeg loudnorm filter (EBU R128).

    # Returns path to the normalized audio file.
    # """
    # ffmpeg = _find_ffmpeg()
    # cmd = [
        # ffmpeg, "-y",
        # "-i", str(input_path),
        # "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        # "-acodec", "pcm_s16le",
        # str(output_path),
    # ]
    # result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    # if result.returncode != 0:
        # raise RuntimeError(f"Normalize failed: {result.stderr}")
    # return str(output_path)
    
def normalize_audio(input_path: str, output_path: str, target_i=-16.0, target_lra=11.0, target_tp=-1.5):
    ffmpeg = _find_ffmpeg()
    
    # --- PASS 1 ---
    pass1_cmd = [
        ffmpeg, "-hide_banner", "-nostats", "-y", 
        "-i", str(input_path),
        "-af", f"loudnorm=I={target_i}:LRA={target_lra}:TP={target_tp}:print_format=json",
        "-f", "null", "-"
    ]
    process1 = subprocess.run(pass1_cmd, capture_output=True, text=True, encoding='utf-8')
    
    start_index = process1.stderr.rfind('{')
    end_index = process1.stderr.rfind('}') + 1
    stats = json.loads(process1.stderr[start_index:end_index])

    # --- PASS 2 ---
    pass2_cmd = [
        ffmpeg, "-hide_banner", "-nostats", "-y",
        "-i", str(input_path),
        "-af", (
            f"loudnorm=linear=true:I={target_i}:LRA={target_lra}:TP={target_tp}:"
            f"measured_I={stats['input_i']}:"
            f"measured_LRA={stats['input_lra']}:"
            f"measured_TP={stats['input_tp']}:"
            f"measured_thresh={stats['input_thresh']}:"
            f"offset={stats['target_offset']}"
        ),
        "-acodec", "pcm_s16le", "-ar", "44100", str(output_path),
    ]
    subprocess.run(pass2_cmd, capture_output=True, check=True)

    # Calculate how much we changed the volume for the UI
    applied_gain = float(stats['target_offset'])
    return str(output_path), applied_gain



def separate_vocals(
    input_path: str,
    output_dir: str,
    model_filename: str = "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    segment_size: int = 256,
    overlap: int = 8,
    sample_rate: int = 44100,
) -> Tuple[str, str]:
    """Separate audio into vocals and instrumental/background stems.

    Uses audio-separator (UVR models) for high-quality source separation.

    Args:
        input_path: Source audio file.
        output_dir: Directory to write output stems.
        model_filename: Model to use. Default is BS-Roformer (best vocal SDR).
        segment_size: Processing segment size. Larger = more VRAM, potentially better quality.
        overlap: Overlap between segments. Higher = better quality, slower.
        sample_rate: Output sample rate.

    Returns:
        Tuple of (vocals_path, instrumental_path).
    """
    from audio_separator.separator import Separator

    separator = Separator(
        output_dir=output_dir,
        output_format="WAV",
        sample_rate=sample_rate,
        mdxc_params={"segment_size": segment_size, "overlap": overlap, "batch_size": 1, "pitch_shift": 0},
    )
    separator.load_model(model_filename=model_filename)
    output_files = separator.separate(input_path)

    # output_files may be just filenames — ensure they're full paths
    resolved = []
    for f in output_files:
        p = Path(f)
        if not p.is_absolute():
            p = Path(output_dir) / p
        resolved.append(str(p))

    # Identify which is which by filename convention
    vocals_path = None
    instrumental_path = None
    for f in resolved:
        fl = f.lower()
        if "vocal" in fl:
            vocals_path = f
        elif "instrument" in fl or "no_vocal" in fl or "other" in fl:
            instrumental_path = f

    # Fallback: if naming didn't match, assume first=vocals, second=instrumental
    if vocals_path is None and len(resolved) >= 1:
        vocals_path = resolved[0]
    if instrumental_path is None and len(resolved) >= 2:
        instrumental_path = resolved[1]

    return vocals_path, instrumental_path


def mix_audio(
    audio_path_1: str,
    audio_path_2: str,
    output_path: str,
    volume_1: float = 1.0,
    volume_2: float = 1.0,
) -> str:
    """Mix two audio files together with independent volume control.

    Args:
        audio_path_1: First audio file (e.g. TTS vocals).
        audio_path_2: Second audio file (e.g. background/ambience).
        output_path: Where to write the mixed audio.
        volume_1: Volume multiplier for first audio (1.0 = unchanged).
        volume_2: Volume multiplier for second audio (1.0 = unchanged).

    Returns:
        Path to the mixed audio file.
    """
    ffmpeg = _find_ffmpeg()
    # Use amix weights for relative balance and let normalize prevent clipping
    filter_complex = (
        f"[0:a][1:a]amix=inputs=2:duration=longest:weights={volume_1} {volume_2}:normalize=1[aout]"
    )
    cmd = [
        ffmpeg, "-y",
        "-i", str(audio_path_1),
        "-i", str(audio_path_2),
        "-filter_complex", filter_complex,
        "-map", "[aout]",
        "-acodec", "pcm_s16le",
        "-ar", "44100",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"Audio mixing failed: {result.stderr}")
    return str(output_path)
