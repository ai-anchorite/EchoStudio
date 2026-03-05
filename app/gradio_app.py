"""Echo TTS Studio — Gradio Application

A polished local TTS studio with voice cloning, character management,
and long-form generation via automatic text chunking.
"""

import os
import json
import re
import shutil
import subprocess
import time
import secrets
import logging
import tempfile
import gc
import random
from pathlib import Path
from typing import Tuple, Any, Optional

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import gradio as gr
import torch
import torchaudio

from inference import (
    load_audio,
    get_speaker_latent_and_mask,
    sample_pipeline,
    compile_model,
    compile_fish_ae,
    sample_euler_cfg_independent_guidances,
)
from model_manager import ModelManager, get_device
from generate import generate_long_form, generate_dubbed_audio, DubSegment, SAMPLE_RATE
from voices import (
    list_voices,
    get_voice_names,
    save_voice,
    load_voice_audio,
    load_voice_latents,
    delete_voice,
    get_voice_preview_path,
)
from utils import preprocess_text, chunk_text_by_time, format_chunks
from transcribe import transcribe as whisper_transcribe, WHISPER_MODELS, unload_model as unload_whisper
from video import (
    extract_audio as video_extract_audio, mux_audio_to_video, get_video_info,
    get_duration as get_media_duration, clip_audio, trim_silence, normalize_audio,
)

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
MODEL_DTYPE = torch.bfloat16

AUDIO_PROMPT_FOLDER = Path("./audio_prompts")
SAMPLER_PRESETS_PATH = Path("./sampler_presets.json")
UI_SETTINGS_PATH = Path("./ui_settings.json")
TEMP_AUDIO_DIR = Path("./temp_gradio_audio")
TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DUB_DIR = Path("./temp_dub")
TEMP_DUB_DIR.mkdir(parents=True, exist_ok=True)

# GRADIO_TEMP_DIR from environment (Pinokio sets this)
GRADIO_TEMP_DIR = Path(os.environ.get("GRADIO_TEMP_DIR", tempfile.gettempdir()))

# OUTPUT_DIR will be loaded from settings after load_ui_settings() is defined
OUTPUT_DIR = None

NO_VOICE_LABEL = "Random voice (no reference)"

# -------------------------------------------------------------------
# Model loading (initialized after settings are loaded)
# -------------------------------------------------------------------
# ModelManager will be initialized after load_ui_settings() is defined

model_compiled = None
fish_ae_compiled = None

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def make_stem(prefix: str, user_id: str | None = None) -> str:
    ts = int(time.time() * 1000)
    rand = secrets.token_hex(4)
    if user_id:
        return f"{prefix}__{user_id}__{ts}_{rand}"
    return f"{prefix}__{ts}_{rand}"


def get_random_seed(seed_value: int | None) -> int:
    """
    Generate a random seed for inference.
    
    Args:
        seed_value: User-provided seed value. If 0 or None, generates a random seed.
    
    Returns:
        A valid seed value for PyTorch (0 to 2^32-1)
    """
    if seed_value is None or seed_value == 0:
        # Use secrets module for cryptographically strong randomness
        return secrets.randbelow(2**32)
    return int(seed_value)


def cleanup_temp_audio(dir_: Path, user_id: str | None, max_age_sec: int = 300):
    now = time.time()
    for p in dir_.glob("*"):
        try:
            if p.is_file() and (now - p.stat().st_mtime) > max_age_sec:
                p.unlink(missing_ok=True)
        except Exception:
            pass
    if user_id:
        for p in dir_.glob(f"*__{user_id}__*"):
            try:
                if p.is_file():
                    p.unlink(missing_ok=True)
            except Exception:
                pass


def save_audio_with_format(audio_tensor: torch.Tensor, base_path: Path, filename: str, sample_rate: int, audio_format: str) -> Path:
    if audio_format == "mp3":
        try:
            output_path = base_path / f"{filename}.mp3"
            torchaudio.save(str(output_path), audio_tensor, sample_rate, format="mp3")
            return output_path
        except Exception:
            pass
    output_path = base_path / f"{filename}.wav"
    torchaudio.save(str(output_path), audio_tensor, sample_rate)
    return output_path


def load_sampler_presets():
    if SAMPLER_PRESETS_PATH.exists():
        with open(SAMPLER_PRESETS_PATH, "r") as f:
            return json.load(f)
    return {}


def load_ui_settings():
    """Load UI settings from JSON file."""
    if UI_SETTINGS_PATH.exists():
        try:
            with open(UI_SETTINGS_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_ui_settings(settings: dict):
    """Save UI settings to JSON file."""
    try:
        with open(UI_SETTINGS_PATH, "w") as f:
            json.dump(settings, f, indent=2)
        return True
    except Exception:
        return False


def get_theme_from_settings():
    """Get the theme from saved settings, or return default."""
    settings = load_ui_settings()
    theme_name = settings.get("theme", "Citrus")
    
    # Built-in themes
    theme_map = {
        "Default": gr.themes.Default(),
        "Soft": gr.themes.Soft(),
        "Monochrome": gr.themes.Monochrome(),
        "Glass": gr.themes.Glass(),
        "Base": gr.themes.Base(),
        "Ocean": gr.themes.Ocean(),
        "Origin": gr.themes.Origin(),
        "Citrus": gr.themes.Citrus(),
    }
    
    # Community themes (string references)
    community_themes = {
        "Miku": "NoCrypt/miku",
        "Interstellar": "Nymbo/Interstellar",
        "xkcd": "gstaff/xkcd",
        "kotaemon": "lone17/kotaemon",
    }
    
    if theme_name in theme_map:
        return theme_map[theme_name]
    elif theme_name in community_themes:
        return community_themes[theme_name]
    else:
        return gr.themes.Soft()  # Default fallback


def get_memory_settings():
    """Get memory settings from ui_settings.json with defaults."""
    settings = load_ui_settings()
    
    # FISH_AE_DTYPE: "float32" (default) or "bfloat16" (for 8GB GPUs)
    fish_ae_dtype_str = settings.get("fish_ae_dtype", "float32")
    fish_ae_dtype = torch.bfloat16 if fish_ae_dtype_str == "bfloat16" else torch.float32
    
    return fish_ae_dtype


# Load memory settings
FISH_AE_DTYPE = get_memory_settings()
DEFAULT_SAMPLE_LATENT_LENGTH = 640

# Load auto-unload setting
auto_unload_models = load_ui_settings().get("auto_unload_models", False)

# Load OUTPUT_DIR from settings
settings = load_ui_settings()
OUTPUT_DIR = Path(settings.get("output_dir", "./outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Clear temp files on startup if enabled
if settings.get("clear_temp_on_start", False):
    for temp_dir in [TEMP_AUDIO_DIR, TEMP_DUB_DIR, GRADIO_TEMP_DIR]:
        if temp_dir.exists():
            for item in temp_dir.iterdir():
                try:
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                except Exception:
                    pass  # Ignore errors during cleanup

# Initialize ModelManager with memory settings
model_manager = ModelManager(
    model_dtype=MODEL_DTYPE,
    fish_ae_dtype=FISH_AE_DTYPE,
    device=get_device(),
)
model_manager.pre_download_models()


def init_session():
    return secrets.token_hex(8)


# Track the last generated file path for save functionality
_last_generated_path: dict = {"path": None, "format": "wav"}
_last_dubbed_video_path: dict = {"path": None}


# -------------------------------------------------------------------
# Voice management callbacks
# -------------------------------------------------------------------



def on_save_voice(name: str, audio_path: str):
    if not name or not name.strip():
        return gr.update(), gr.update(value="Enter a name for the voice.")
    if not audio_path:
        return gr.update(), gr.update(value="Upload or record audio first.")
    name = name.strip()
    try:
        audio = load_audio(audio_path).to(model_manager.device)
        speaker_latent, speaker_mask = get_speaker_latent_and_mask(
            model_manager.fish_ae, model_manager.pca_state, audio.to(model_manager.fish_ae.dtype).to(model_manager.device),
        )
        save_voice(name, audio_path, speaker_latent, speaker_mask)
    except Exception as e:
        return gr.update(), gr.update(value=f"Error: {e}")
    names = get_voice_names()
    return (
        gr.update(choices=[NO_VOICE_LABEL] + names, value=name),
        gr.update(value=f"Voice '{name}' saved with cached latents."),
    )


def on_delete_voice(voice_name: str):
    if not voice_name or voice_name == "None":
        return gr.update(), gr.update(value="No voice selected.")
    delete_voice(voice_name)
    names = get_voice_names()
    return (
        gr.update(choices=["None"] + names, value="None"),
        gr.update(value=f"Voice '{voice_name}' deleted."),
    )


# -------------------------------------------------------------------
# Output file management
# -------------------------------------------------------------------

def save_to_outputs(audio_format: str):
    """Copy the last generated audio to the outputs folder."""
    src = _last_generated_path.get("path")
    if not src or not Path(src).exists():
        return gr.update(value="No audio to save. Generate something first.")
    src_path = Path(src)
    ts = time.strftime("%Y%m%d_%H%M%S")
    dst = OUTPUT_DIR / f"echo_tts_{ts}{src_path.suffix}"
    shutil.copy2(str(src_path), str(dst))
    return gr.update(value=f"Saved to {dst.name}")


def auto_save_if_enabled(auto_save: bool, audio_format: str):
    """Auto-save after generation if checkbox is enabled."""
    if not auto_save:
        return gr.update(value="")
    return save_to_outputs(audio_format)


def open_outputs_folder():
    """Open the outputs folder in the system file explorer."""
    folder = str(OUTPUT_DIR.resolve())
    try:
        if os.name == "nt":
            os.startfile(folder)
        elif os.name == "posix":
            subprocess.Popen(["xdg-open", folder])
        else:
            subprocess.Popen(["open", folder])
    except Exception:
        pass
    return gr.update(value=f"Opened: {folder}")


def save_dub_to_outputs(audio_format: str):
    """Copy the last dubbed video to the outputs folder."""
    src = _last_dubbed_video_path.get("path")
    if not src or not Path(src).exists():
        return gr.update(value="No video to save. Generate a dub first.")
    src_path = Path(src)
    ts = time.strftime("%Y%m%d_%H%M%S")
    dst = OUTPUT_DIR / f"dubbed_{ts}.mp4"
    shutil.copy2(str(src_path), str(dst))
    return gr.update(value=f"Saved to {dst.name}")


# -------------------------------------------------------------------
# Dub workflow callbacks
# -------------------------------------------------------------------

def dub_extract_and_transcribe(
    video_path: str,
    whisper_model: str,
    task: str,
    progress=gr.Progress(track_tqdm=False),
):
    """Extract audio from video, transcribe, and return editable text."""
    if not video_path:
        raise gr.Error("Please upload a video file first.")

    progress(0.1, desc="Extracting audio from video...")

    # Get video info
    try:
        info = get_video_info(video_path)
    except Exception as e:
        raise gr.Error(f"Could not read video: {e}")

    # Extract audio
    extracted_audio_path = str(TEMP_DUB_DIR / "extracted_audio.wav")
    try:
        video_extract_audio(video_path, extracted_audio_path, sample_rate=44100)
    except Exception as e:
        raise gr.Error(f"Audio extraction failed: {e}")

    progress(0.3, desc=f"Transcribing with Whisper ({whisper_model})...")

    # Transcribe
    whisper_task = "translate" if task == "Translate to English" else "transcribe"
    try:
        result = whisper_transcribe(
            audio_path=extracted_audio_path,
            model_name=whisper_model,
            task=whisper_task,
        )
    except Exception as e:
        raise gr.Error(f"Transcription failed: {e}")

    # Free Whisper VRAM before TTS generation
    unload_whisper()

    progress(1.0, desc="Done")

    duration_str = f"{info['duration']:.1f}s" if info.get("duration") else "unknown"
    resolution = f"{info.get('width', '?')}x{info.get('height', '?')}"
    status = (
        f"Extracted & transcribed: {duration_str}, {resolution}, "
        f"language: {result.language}, {len(result.segments)} segments"
    )

    # Format transcript as numbered segments for editing
    # Each line is prefixed with segment index so we can map edits back
    segment_lines = []
    for i, s in enumerate(result.segments):
        if s.text.strip():
            segment_lines.append(f"[{i}] {s.text.strip()}")
    formatted_transcript = "\n".join(segment_lines)

    return (
        gr.update(value=formatted_transcript),                 # transcript textbox
        gr.update(value=result.language, visible=True),        # detected language
        gr.update(value=extracted_audio_path, visible=True),   # extracted audio player
        gr.update(value=status),                               # dub status
        extracted_audio_path,                                   # state: raw path string
        # Store segments as serializable dicts for the dub step
        [{"text": s.text, "start": s.start, "end": s.end} for s in result.segments],
        info.get("duration", 0.0),                             # video duration
    )


def dub_generate_and_mux(
    video_path: str,
    transcript: str,
    # Speaker 1
    dub_spk1_source: str,
    dub_spk1_saved: str,
    dub_spk1_upload: str,
    # Speaker 2
    dub_spk2_source: str,
    dub_spk2_saved: str,
    dub_spk2_upload: str,
    # Generation parameters
    num_steps: int,
    rng_seed: int,
    cfg_scale_text: float,
    cfg_scale_speaker: float,
    cfg_min_t: float,
    cfg_max_t: float,
    truncation_factor: float,
    rescale_k: float,
    rescale_sigma: float,
    force_speaker: bool,
    speaker_kv_scale: float,
    speaker_kv_min_t: float,
    speaker_kv_max_layers: int,
    audio_format: str,
    session_id: str,
    segments_data,
    video_duration: float,
    extracted_audio_path,
    progress=gr.Progress(track_tqdm=False),
):
    """Generate time-aligned TTS from segments and mux onto the original video."""
    if not video_path:
        raise gr.Error("No video loaded.")
    if not transcript or not transcript.strip():
        raise gr.Error("Transcript is empty. Nothing to generate.")

    start_time = time.time()

    # Resolve speakers — concatenate audio for multi-speaker (same logic as TTS tab)
    dub_spk2_active = dub_spk2_saved != "Inactive" or dub_spk2_source == "upload"
    spk_slots = [(dub_spk1_source, dub_spk1_saved, dub_spk1_upload)]
    if dub_spk2_active:
        spk_slots.append((dub_spk2_source, dub_spk2_saved, dub_spk2_upload))

    audio_parts = []
    for src, saved, upload in spk_slots:
        # Handle "Clone from video" as a special case
        if src == "Clone from video":
            if extracted_audio_path and Path(str(extracted_audio_path)).exists():
                part = load_audio(str(extracted_audio_path)).to(model_manager.device)
            else:
                raise gr.Error("No extracted audio available. Run Extract & Transcribe first.")
        else:
            part = _resolve_speaker_audio(src, saved, upload)
        
        if part is not None:
            if part.ndim == 1:
                part = part.unsqueeze(0)
            audio_parts.append(part)

    speaker_audio = None
    speaker_latent = None
    speaker_mask = None

    if audio_parts:
        # Concatenate speaker audio in reverse order (S1 goes last for causal encoder)
        audio_parts.reverse()
        speaker_audio = torch.cat(audio_parts, dim=-1)

    # Prepare KV scaling parameters
    spk_kv_scale = float(speaker_kv_scale) if force_speaker else None
    spk_kv_min_t = float(speaker_kv_min_t) if force_speaker else None
    spk_kv_max_layers = int(speaker_kv_max_layers) if force_speaker else None

    # Build DubSegments from stored transcription data
    # Strategy: use original segment timing, merge short consecutive segments
    # into chunks of at least MIN_SEGMENT_DURATION seconds so TTS has enough
    # context and we avoid extreme time-stretch ratios.
    MIN_SEGMENT_DURATION = 4.0  # seconds — merge segments shorter than this

    dub_segments = []
    if segments_data and len(segments_data) > 0:
        # Parse edited transcript — extract text per segment index if tagged,
        # otherwise fall back to original segment text
        edited_texts = {}
        for line in transcript.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # Try to parse "[index] text" format
            m = re.match(r'^\[(\d+)\]\s*(.*)', line)
            if m:
                edited_texts[int(m.group(1))] = m.group(2).strip()

        # Build raw segments with edited text where available
        raw_segments = []
        for i, seg_dict in enumerate(segments_data):
            text = edited_texts.get(i, seg_dict["text"]).strip()
            if not text:
                continue
            raw_segments.append(DubSegment(
                text=text,
                start=seg_dict["start"],
                end=seg_dict["end"],
            ))

        # Merge short consecutive segments into longer chunks
        if raw_segments:
            merged = [raw_segments[0]]
            for seg in raw_segments[1:]:
                prev = merged[-1]
                prev_duration = prev.end - prev.start
                # Merge if previous segment is too short
                if prev_duration < MIN_SEGMENT_DURATION:
                    merged[-1] = DubSegment(
                        text=prev.text + " " + seg.text,
                        start=prev.start,
                        end=seg.end,
                    )
                else:
                    merged.append(seg)
            # Final pass: merge the last segment if it's too short
            if len(merged) > 1 and (merged[-1].end - merged[-1].start) < MIN_SEGMENT_DURATION:
                last = merged.pop()
                merged[-1] = DubSegment(
                    text=merged[-1].text + " " + last.text,
                    start=merged[-1].start,
                    end=last.end,
                )
            dub_segments = merged
    else:
        raise gr.Error("No segment timing data. Please re-run Extract & Transcribe first.")

    # Get actual video duration if not stored
    if not video_duration or video_duration <= 0:
        try:
            video_duration = get_media_duration(video_path)
        except Exception:
            video_duration = dub_segments[-1].end + 1.0 if dub_segments else 60.0

    total_segs = len(dub_segments)

    def progress_cb(seg_idx, total, seg_text):
        progress(0.05 + 0.75 * (seg_idx / total), desc=f"Generating segment {seg_idx + 1}/{total}...")

    progress(0.05, desc=f"Generating {total_segs} segment(s) with time alignment...")

    audio_out, seg_info = generate_dubbed_audio(
        model=model_manager.model, fish_ae=model_manager.fish_ae, pca_state=model_manager.pca_state,
        segments=dub_segments,
        total_duration=video_duration,
        speaker_audio=speaker_audio,
        speaker_latent=speaker_latent,
        speaker_mask=speaker_mask,
        rng_seed=get_random_seed(rng_seed),
        num_steps=min(max(int(num_steps), 1), 80),
        cfg_scale_text=float(cfg_scale_text),
        cfg_scale_speaker=float(cfg_scale_speaker),
        cfg_min_t=float(cfg_min_t),
        cfg_max_t=float(cfg_max_t),
        truncation_factor=float(truncation_factor),
        rescale_k=float(rescale_k),
        rescale_sigma=float(rescale_sigma),
        speaker_kv_scale=spk_kv_scale,
        speaker_kv_min_t=spk_kv_min_t,
        speaker_kv_max_layers=spk_kv_max_layers,
        sample_latent_length=DEFAULT_SAMPLE_LATENT_LENGTH,
        progress_callback=progress_cb,
    )

    # Save generated audio to temp
    audio_to_save = audio_out.cpu()
    if audio_to_save.ndim == 1:
        audio_to_save = audio_to_save.unsqueeze(0)
    elif audio_to_save.ndim == 3:
        audio_to_save = audio_to_save[0]

    tts_audio_path = str(TEMP_DUB_DIR / "dub_tts.wav")
    torchaudio.save(tts_audio_path, audio_to_save, SAMPLE_RATE)

    progress(0.85, desc="Muxing audio onto video...")

    # Mux onto video - save to temp first
    stem = make_stem("dubbed", session_id)
    output_video_path = str(TEMP_DUB_DIR / f"{stem}.mp4")
    try:
        mux_audio_to_video(video_path, tts_audio_path, output_video_path, keep_original_audio=False)
    except Exception as e:
        raise gr.Error(f"Muxing failed: {e}")

    # Track for save button
    _last_dubbed_video_path["path"] = output_video_path

    progress(1.0, desc="Done")

    gen_time = time.time() - start_time
    audio_duration = audio_to_save.shape[-1] / SAMPLE_RATE
    stretched_count = sum(1 for s in seg_info if not s.get("skipped") and abs(s.get("speed_ratio", 1.0) - 1.0) > 0.05)
    status = (
        f"Dubbed in {gen_time:.1f}s — {audio_duration:.1f}s audio, "
        f"{total_segs} segments ({stretched_count} time-stretched)."
    )

    # Auto-unload models if enabled
    if auto_unload_models:
        model_manager.unload_all()
        unload_whisper()
        gc.collect()
        print("[dub] Models unloaded after inference")

    return (
        gr.update(value=output_video_path, visible=True),  # output video
        gr.update(value=status),                             # dub status
        #gr.update(value=""),                                 # clear save status
    )


# -------------------------------------------------------------------
# Generation callback
# -------------------------------------------------------------------

def _resolve_speaker_audio(source: str, saved_name: str, upload_path: str):
    """Resolve a single speaker slot to an audio tensor or None."""
    if source == "saved" and saved_name and saved_name != NO_VOICE_LABEL:
        # Try cached latents first — but for multi-speaker we need raw audio
        # to concatenate, so load the audio file directly
        audio_path = load_voice_audio(saved_name)
        if audio_path:
            return load_audio(audio_path).to(model_manager.device)
    elif source == "upload" and upload_path:
        return load_audio(upload_path).to(model_manager.device)
    return None


def generate_audio(
    text_prompt: str,
    # Speaker 1
    spk1_source: str, spk1_saved: str, spk1_upload: str,
    # Speaker 2
    spk2_source: str, spk2_saved: str, spk2_upload: str,
    num_steps: int,
    rng_seed: int,
    cfg_scale_text: float,
    cfg_scale_speaker: float,
    cfg_min_t: float,
    cfg_max_t: float,
    truncation_factor: float,
    rescale_k: float,
    rescale_sigma: float,
    force_speaker: bool,
    speaker_kv_scale: float,
    speaker_kv_min_t: float,
    speaker_kv_max_layers: int,
    audio_format: str,
    session_id: str,
    inter_chunk_silence: int = 350,
    crossfade_ms_val: int = 80,
    progress=gr.Progress(track_tqdm=False),
):
    """Main generation function — handles both short and long-form text."""
    if not text_prompt or not text_prompt.strip():
        raise gr.Error("Please enter some text to generate.")

    cleanup_temp_audio(TEMP_AUDIO_DIR, session_id)
    start_time = time.time()

    # Resolve speakers — concatenate audio for multi-speaker
    spk2_active = spk2_saved != "Inactive" or spk2_source == "upload"
    spk_slots = [(spk1_source, spk1_saved, spk1_upload)]
    if spk2_active:
        spk_slots.append((spk2_source, spk2_saved, spk2_upload))

    audio_parts = []
    for src, saved, upload in spk_slots:
        part = _resolve_speaker_audio(src, saved, upload)
        if part is not None:
            if part.ndim == 1:
                part = part.unsqueeze(0)
            audio_parts.append(part)

    speaker_audio = None
    speaker_latent = None
    speaker_mask = None

    if audio_parts:
        # Concatenate speaker audio in reverse order — the causal speaker encoder
        # gives the strongest representation to the last audio in the sequence,
        # which the model maps to [S1]. So S1 audio goes last.
        audio_parts.reverse()
        speaker_audio = torch.cat(audio_parts, dim=-1)

    spk_kv_scale = float(speaker_kv_scale) if force_speaker else None
    spk_kv_min_t = float(speaker_kv_min_t) if force_speaker else None
    spk_kv_max_layers = int(speaker_kv_max_layers) if force_speaker else None

    chunks = chunk_text_by_time(text_prompt)
    is_long_form = len(chunks) > 1

    def progress_cb(chunk_idx, total, chunk_text):
        progress((chunk_idx / total), desc=f"Generating chunk {chunk_idx + 1}/{total}...")

    if is_long_form:
        progress(0, desc=f"Generating {len(chunks)} chunks...")

    audio_out, normalized_text, chunk_list = generate_long_form(
        model=model_manager.model, fish_ae=model_manager.fish_ae, pca_state=model_manager.pca_state,
        text=text_prompt,
        speaker_audio=speaker_audio,
        speaker_latent=speaker_latent,
        speaker_mask=speaker_mask,
        rng_seed=get_random_seed(rng_seed),
        num_steps=min(max(int(num_steps), 1), 80),
        cfg_scale_text=float(cfg_scale_text),
        cfg_scale_speaker=float(cfg_scale_speaker),
        cfg_min_t=float(cfg_min_t),
        cfg_max_t=float(cfg_max_t),
        truncation_factor=float(truncation_factor),
        rescale_k=float(rescale_k),
        rescale_sigma=float(rescale_sigma),
        speaker_kv_scale=spk_kv_scale,
        speaker_kv_min_t=spk_kv_min_t,
        speaker_kv_max_layers=spk_kv_max_layers,
        sample_latent_length=DEFAULT_SAMPLE_LATENT_LENGTH,
        progress_callback=progress_cb if is_long_form else None,
        crossfade_ms=int(crossfade_ms_val),
        inter_chunk_silence_ms=int(inter_chunk_silence),
    )

    # Ensure 2D for saving
    audio_to_save = audio_out.cpu()
    if audio_to_save.ndim == 1:
        audio_to_save = audio_to_save.unsqueeze(0)
    elif audio_to_save.ndim == 3:
        audio_to_save = audio_to_save[0]

    stem = make_stem("generated", session_id)
    output_path = save_audio_with_format(audio_to_save, TEMP_AUDIO_DIR, stem, SAMPLE_RATE, audio_format)

    # Track for save button
    _last_generated_path["path"] = str(output_path)
    _last_generated_path["format"] = audio_format

    gen_time = time.time() - start_time
    audio_duration = audio_to_save.shape[-1] / SAMPLE_RATE
    chunk_info = f" ({len(chunk_list)} chunks)" if len(chunk_list) > 1 else ""
    time_str = f"Generated {audio_duration:.1f}s of audio in {gen_time:.1f}s{chunk_info}"

    # Auto-unload models if enabled
    if auto_unload_models:
        model_manager.unload_all()
        unload_whisper()
        gc.collect()
        print("[generate] Models unloaded after inference")

    return (
        gr.update(value=str(output_path), visible=True),
        gr.update(value=time_str, visible=True),
        gr.update(value=normalized_text, visible=True),
        gr.update(value=""),  # clear save status
    )


# -------------------------------------------------------------------
# Preset helpers
# -------------------------------------------------------------------

def apply_sampler_preset(preset_name):
    presets = load_sampler_presets()
    if preset_name == "Custom" or preset_name not in presets:
        return [gr.update()] * 8
    p = presets[preset_name]
    def n(val, default):
        try:
            return float(val)
        except (ValueError, TypeError):
            return default
    return [
        gr.update(value=int(n(p.get("num_steps", "40"), 40))),
        gr.update(value=n(p.get("cfg_scale_text", "3.0"), 3.0)),
        gr.update(value=n(p.get("cfg_scale_speaker", "5.0"), 5.0)),
        gr.update(value=n(p.get("cfg_min_t", "0.5"), 0.5)),
        gr.update(value=n(p.get("cfg_max_t", "1.0"), 1.0)),
        gr.update(value=n(p.get("truncation_factor", "0.8"), 0.8)),
        gr.update(value=n(p.get("rescale_k", "1.2"), 1.2)),
        gr.update(value=n(p.get("rescale_sigma", "3.0"), 3.0)),
    ]


AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".aac", ".opus"}


def get_audio_prompt_files(search_query: str = ""):
    if AUDIO_PROMPT_FOLDER is None or not AUDIO_PROMPT_FOLDER.exists():
        return []
    files = sorted(
        [f.name for f in AUDIO_PROMPT_FOLDER.iterdir() if f.is_file() and f.suffix.lower() in AUDIO_EXTS],
        key=str.lower,
    )
    if search_query.strip():
        q = search_query.lower()
        files = [f for f in files if q in f.lower()]
    return [[f] for f in files]


def select_audio_prompt_file(evt: gr.SelectData):
    if evt.value and AUDIO_PROMPT_FOLDER is not None:
        file_path = AUDIO_PROMPT_FOLDER / evt.value
        if file_path.exists():
            return gr.update(value=str(file_path))
    return gr.update()


# -------------------------------------------------------------------
# Emotion/Tone Tags - Claude leave this for now, we'll circle back later
# -------------------------------------------------------------------

EMOTION_TAGS = [
    # 12 popular emotional markers
    "(angry)", "(sad)", "(delighted)", "(nervous)", "(excited)", "(surprised)",
    "(scared)", "(frustrated)", "(sarcastic)", "(hysterical)", "(sobbing)", "(laughing)",
    # Tone markers
    "(shouting)", "(screaming)", "(whispering)", "(soft tone)",
    # Special markers
    "(chuckling)", "(crying loudly)", "(sighing)", "(panting)", "(groaning)",
    "(crowd laughing)", "(background laughter)", "(audience laughing)",
]

# -------------------------------------------------------------------
# CSS
# -------------------------------------------------------------------

APP_CSS = """
.generated-audio-player {
    border: 2px solid #667eea; border-radius: 12px; padding: 16px;
    background: linear-gradient(135deg, rgba(102,126,234,0.06) 0%, rgba(118,75,162,0.03) 100%);
    box-shadow: 0 2px 8px rgba(102,126,234,0.15); margin: 0.5rem 0;
}
.section-sep {
    height: 2px; border: none; margin: 1.2rem 0;
    background: linear-gradient(90deg, transparent 0%, #667eea 30%, #764ba2 70%, transparent 100%);
}
.save-row { display: flex; align-items: center; gap: 8px; }
.gr-group { border: 1px solid #d1d5db; background: #f9fafb; border-radius: 8px; }
.dark .gr-group { border: 1px solid #4b5563; background: #1f2937; }
.tab-nav button { font-weight: 600; }
.media-window video {
    max-height: 60vh !important;
    object-fit: contain;
    width: 100%;
}
"""


# -------------------------------------------------------------------
# UI Layout
# -------------------------------------------------------------------

with gr.Blocks(title="Echo TTS Studio", css=APP_CSS, theme=get_theme_from_settings()) as demo:

    session_id_state = gr.State(None)

    # gr.Markdown("# Echo TTS Studio")
    # gr.Markdown("Local text-to-speech with voice cloning and long-form generation.")

    with gr.Tabs() as main_tabs:

        # ==============================================================
        # TAB 1: TTS
        # ==============================================================
        with gr.Tab("TTS", id="tab_generate"):

            # --- Voice Section ---
            with gr.Accordion("Voice", open=True):
                gr.Markdown(
                    "Select a voice for`[S1]` `[S2]` is optional"
                )

                with gr.Tabs() as speaker_tabs:
                    # --- Speaker 1 (always visible) ---
                    with gr.Tab("Speaker 1", id="spk_tab_1"):
                        spk1_source = gr.Radio(
                            choices=["saved", "upload"], value="saved",
                            label="Source", show_label=False,
                        )
                        with gr.Group(visible=True) as spk1_saved_group:
                            voice_names = get_voice_names()
                            spk1_dropdown = gr.Dropdown(
                                choices=[NO_VOICE_LABEL] + voice_names,
                                value=NO_VOICE_LABEL, label="Select Voice", interactive=True,
                            )
                            spk1_preview = gr.Audio(label="Preview", interactive=False, visible=False)
                        with gr.Group(visible=False) as spk1_upload_group:
                            spk1_audio = gr.Audio(
                                sources=["upload", "microphone"], type="filepath",
                                label="Upload or Record",
                                max_length=600,
                            )
                            with gr.Accordion("Audio Library", open=False):
                                audio_prompt_search = gr.Textbox(
                                    placeholder="Search audio prompts...", label="", lines=1, max_lines=1,
                                )
                                audio_prompt_table = gr.Dataframe(
                                    value=get_audio_prompt_files(),
                                    headers=["Filename"], datatype=["str"],
                                    row_count=(6, "dynamic"), col_count=(1, "fixed"),
                                    interactive=False, label="",
                                )

                    # --- Speaker 2 (always visible, inactive by default) ---
                    with gr.Tab("Speaker 2", id="spk_tab_2"):
                        spk2_source = gr.Radio(
                            choices=["saved", "upload"], value="saved",
                            label="Source", show_label=False,
                        )
                        with gr.Group(visible=True) as spk2_saved_group:
                            spk2_dropdown = gr.Dropdown(
                                choices=["Inactive", NO_VOICE_LABEL] + voice_names,
                                value="Inactive", label="Select Voice", interactive=True,
                                info="Set to 'Inactive' to disable Speaker 2.",
                            )
                            spk2_preview = gr.Audio(label="Preview", interactive=False, visible=False)
                        with gr.Group(visible=False) as spk2_upload_group:
                            spk2_audio = gr.Audio(
                                sources=["upload", "microphone"], type="filepath",
                                label="Upload or Record",
                                max_length=600,
                            )

                # Alias for audio library event wiring
                uploaded_audio = spk1_audio

            # --- Text Section ---
            with gr.Accordion("Text", open=True):
                text_prompt = gr.Textbox(
                    label="",
                    show_label=False,
                    container=False,
                    placeholder="Enter your text here:\n\n The EchoTTS model is trained for 30s of audio output, so longer text will automatically be split into chunks to accomodate this.\n\n "
                    "Click `Pre-process Text` to preview (and edit) the text before generation. This will chunk the text and normalize punctuation. This is optional, "
                    "the app will automatically do this on generation start, but will respect your edits if you've pre-prosessed. \n "
                    "Longer chunks of text can also work if manually edited. This will cause the model to speak more quickly to try and fit into the 30s window.\n\n "
                    "Use [S1], [S2] for speaker tags. Use (laughs), (groans), (singing), (angry), (frantic) etc. for expression.\n Leave untagged for automatic [S1] tagging.",
                    lines=9, max_lines=30,
                    elem_id="text_prompt_input",
                )

                with gr.Row():
                    # # Emotion/Tone tag dropdown. Claude leave this for now, we'll circle back later
                    # emotion_dropdown = gr.Dropdown(
                        # choices=EMOTION_TAGS,
                        # label="",
                        # show_label=False,
                        # container=False,
                        # info="Select a tag to insert at cursor position",
                        # scale=2,
                        # elem_id="emotion_dropdown",
                    # )
                    char_count = gr.Textbox(
                        value="0 chars | ~0s estimated", label="", show_label=False, container=False, interactive=False, max_lines=1, scale=3,
                    ) 
                    preprocess_btn = gr.Button("Pre-process Text", size="sm", scale=1)
                    clear_text_btn = gr.Button("Clear", size="sm", variant="stop", scale=1)
              

            # --- Settings Accordion ---
            with gr.Accordion("Settings", open=False):
                presets = load_sampler_presets()
                preset_keys = list(presets.keys())
                first_preset = preset_keys[0] if preset_keys else "Custom"

                with gr.Row():
                    preset_dropdown = gr.Dropdown(
                        choices=["Custom"] + preset_keys, value=first_preset,
                        label="Preset",
                        info="Quick configurations. 'Balanced' is a good starting point.",
                        scale=2,
                    )
                    num_steps = gr.Slider(
                        label="Steps", value=40, precision=0, minimum=5, maximum=80, step=5, scale=1,
                        info="Diffusion steps. 20-40 is fast, 60 for quality.",
                    )
                    rng_seed = gr.Number(
                        label="Seed", value=0, precision=0, scale=1,
                        info="0 = random. Same seed + settings = same output.",
                    )
                    audio_format = gr.Radio(choices=["wav", "mp3"], value="wav", label="Format", scale=1)

                with gr.Accordion("Guidance (CFG)", open=False):
                    with gr.Row():
                        cfg_scale_text = gr.Slider(
                            label="Text CFG", value=5.0, minimum=1, maximum=20, step=0.5,
                            info="How closely to follow the text. Higher = more literal, may reduce naturalness.",
                        )
                        cfg_scale_speaker = gr.Slider(
                            label="Speaker CFG", value=5.0, minimum=1, maximum=20, step=0.5,
                            info="How closely to match the reference voice. Higher = stronger cloning.",
                        )
                    with gr.Row():
                        cfg_min_t = gr.Number(
                            label="CFG Min t", value=0.5, minimum=0, maximum=1, step=0.05,
                            info="Timestep below which guidance is disabled. 0.5 works well.",
                        )
                        cfg_max_t = gr.Number(
                            label="CFG Max t", value=1.0, minimum=0, maximum=1, step=0.05,
                            info="Timestep above which guidance is disabled. Usually 1.0.",
                        )

                with gr.Accordion("Sampling Style", open=False):
                    with gr.Row():
                        truncation_factor = gr.Number(
                            label="Initial Noise Scale", value=0.8, minimum=0, step=0.05,
                            info="Scales starting noise. 0.8 reduces artifacts, 1.0 for full range.",
                        )
                        rescale_k = gr.Number(
                            label="Rescale k", value=1.2, minimum=0, step=0.05,
                            info="Temporal score rescaling. ~1.2 = smoother/flatter, ~0.96 = sharper/vivid.",
                        )
                        rescale_sigma = gr.Number(
                            label="Rescale σ", value=3.0, minimum=0, step=0.1,
                            info="Rescaling bandwidth. 3.0 is the default.",
                        )

                with gr.Accordion("Long-form Stitching", open=False):
                    with gr.Row():
                        inter_chunk_silence = gr.Number(
                            label="Chunk Gap (ms)", value=350, minimum=0, maximum=1000, step=50, precision=0,
                            info="Silence inserted between chunks. 300-500ms feels like a natural breath pause.",
                        )
                        crossfade_ms = gr.Number(
                            label="Crossfade (ms)", value=80, minimum=0, maximum=500, step=10, precision=0,
                            info="Smooth blend between chunks. Keeps transitions click-free.",
                        )

                force_speaker = gr.Checkbox(
                    label="Enable Force Speaker (KV Scaling)",
                    value=False,
                    info="Amplifies speaker attention in the model. Can improve cloning but may reduce quality at high values.",
                )
                with gr.Group(visible=False) as kv_scaling_group:
                    with gr.Row():
                        speaker_kv_scale = gr.Number(
                            label="KV Scale", value=1.5, minimum=0, step=0.1,
                            info="Multiplier for speaker key/value attention. 1.0 = normal, higher = stronger.",
                        )
                        speaker_kv_min_t = gr.Number(
                            label="KV Min t", value=0.9, minimum=0, maximum=1, step=0.05,
                            info="Only apply KV scaling above this timestep.",
                        )
                        speaker_kv_max_layers = gr.Number(
                            label="Max Layers", value=24, precision=0, minimum=0, maximum=24,
                            info="Number of decoder layers to apply scaling to (24 = all).",
                        )

            # --- Generate Button ---
            generate_btn = gr.Button("Generate Audio", variant="primary", size="lg")

            gr.HTML('<hr class="section-sep">')

            # --- Output Section ---
            gr.Markdown("### Output")
            generation_time_display = gr.Markdown("", visible=False)

            with gr.Group(elem_classes=["generated-audio-player"]):
                generated_audio = gr.Audio(label="Generated Audio", visible=False)

            # --- Save Controls ---
            with gr.Row():
                with gr.Group():
                    save_audio_btn = gr.Button("💾 Save to Outputs", size="sm")
                    open_folder_btn = gr.Button("📂 Open Outputs Folder", size="sm")
                    auto_save_checkbox = gr.Checkbox(label="Auto-save", value=False)
                save_status = gr.Textbox(value="", label="", show_label=False, interactive=False, max_lines=1, scale=2)

            text_display = gr.Markdown("", visible=False)


        # ==============================================================
        # TAB 3: Dub
        # ==============================================================
        with gr.Tab("Dub", id="tab_dub"):

            # --- Step 1: Upload & Transcribe ---
            dub_video_input = gr.Video(label="Upload Video", sources=["upload"], elem_classes=["media-window"])

            with gr.Row():
                dub_task = gr.Radio(
                    choices=["Transcribe", "Translate to English"],
                    value="Transcribe",
                    label="Task",
                    show_label=False,
                    container=False,
                    scale=1,
                )
                dub_extract_btn = gr.Button("Extract & Transcribe", variant="primary", size="sm", scale=3)

            with gr.Accordion("Transcription Settings", open=False):
                dub_whisper_model = gr.Dropdown(
                    choices=WHISPER_MODELS, value="large",
                    label="Whisper Model",
                    info="Larger = more accurate, slower. Turbo is fast but can't translate.",
                )
                dub_detected_lang = gr.Textbox(label="Detected Language", interactive=False, visible=False, max_lines=1)
                dub_extracted_audio = gr.Audio(label="Extracted Audio", interactive=False, visible=False)

            dub_extracted_audio_path = gr.State(None)
            dub_segments_state = gr.State(None)
            dub_video_duration_state = gr.State(0.0)

            # --- Step 2: Edit Transcript ---
            with gr.Accordion("Edit Transcript", open=True):
                dub_transcript = gr.Textbox(
                    label="Transcript",
                    show_label=False,
                    container=False,
                    placeholder="Transcription will appear here after extraction. Each line is tagged [n] with its segment index. Edit the text freely — timing is preserved from the original segments.",
                    lines=12, max_lines=12, interactive=True,
                    autoscroll=False,
                )

            # --- Step 3: Voice & Generate ---
            with gr.Accordion("Voice & Settings", open=True):
                
                # --- Speaker 1 (always visible) ---
                with gr.Tab("Speaker 1", id="dub_spk_tab_1"):
                    dub_spk1_source = gr.Radio(
                        choices=["Clone from video", "saved", "upload"],
                        value="Clone from video",
                        label="Source",
                        show_label=False,
                    )
                    with gr.Group(visible=False) as dub_spk1_saved_group:
                        dub_spk1_dropdown = gr.Dropdown(
                            choices=[NO_VOICE_LABEL] + voice_names,
                            value=NO_VOICE_LABEL, label="Select Voice", interactive=True,
                        )
                        dub_spk1_preview = gr.Audio(label="Preview", interactive=False, visible=False)
                    with gr.Group(visible=False) as dub_spk1_upload_group:
                        dub_spk1_audio = gr.Audio(
                            sources=["upload", "microphone"], type="filepath",
                            label="Upload or Record",
                            max_length=600,
                        )

                # --- Speaker 2 (inactive by default) ---
                with gr.Tab("Speaker 2", id="dub_spk_tab_2"):
                    dub_spk2_source = gr.Radio(
                        choices=["Clone from video", "saved", "upload"],
                        value="saved",
                        label="Source",
                        show_label=False,
                    )
                    with gr.Group(visible=True) as dub_spk2_saved_group:
                        dub_spk2_dropdown = gr.Dropdown(
                            choices=["Inactive", NO_VOICE_LABEL] + voice_names,
                            value="Inactive", label="Select Voice", interactive=True,
                            info="Set to 'Inactive' to disable Speaker 2. Use [S1]/[S2] tags in transcript for multi-speaker.",
                        )
                        dub_spk2_preview = gr.Audio(label="Preview", interactive=False, visible=False)
                    with gr.Group(visible=False) as dub_spk2_upload_group:
                        dub_spk2_audio = gr.Audio(
                            sources=["upload", "microphone"], type="filepath",
                            label="Upload or Record",
                            max_length=600,
                        )

                with gr.Accordion("Generation Settings", open=False):
                    with gr.Row():
                        dub_num_steps = gr.Slider(
                            label="Steps", value=40, precision=0, minimum=5, maximum=80, step=5,
                            info="Diffusion steps. 20-40 is fast, 60 for quality.",
                        )
                        dub_rng_seed = gr.Number(
                            label="Seed", value=0, precision=0,
                            info="0 = random. Same seed + settings = same output.",
                        )
                        dub_audio_format = gr.Radio(choices=["wav", "mp3"], value="wav", label="Format")

                    with gr.Row():
                        dub_cfg_text = gr.Slider(
                            label="Text CFG", value=9.0, minimum=1, maximum=20, step=0.5,
                            info="Improve speaker's attention to the text. Raise if speaker is lapsing back into their native language.",
                        )
                        dub_cfg_speaker = gr.Slider(
                            label="Speaker CFG", value=3.0, minimum=1, maximum=10, step=0.5,
                            info="Decrease to 1 or 2 if speaker's accent is affecting translation. Higher = stronger cloning.",
                        )

                    # Advanced settings (consolidated, hidden by default)
                    with gr.Accordion("Advanced Settings", open=False):
                        gr.Markdown("**Guidance Timing**")
                        with gr.Row():
                            dub_cfg_min_t = gr.Number(
                                label="CFG Min t", value=0.5, minimum=0, maximum=1, step=0.05,
                                info="Timestep below which guidance is disabled. 0.5 works well.",
                            )
                            dub_cfg_max_t = gr.Number(
                                label="CFG Max t", value=1.0, minimum=0, maximum=1, step=0.05,
                                info="Timestep above which guidance is disabled. Usually 1.0.",
                            )
                        
                        gr.Markdown("**Sampling Style**")
                        with gr.Row():
                            dub_truncation_factor = gr.Number(
                                label="Initial Noise Scale", value=0.8, minimum=0, step=0.05,
                                info="Scales starting noise. 0.8 reduces artifacts, 1.0 for full range.",
                            )
                            dub_rescale_k = gr.Number(
                                label="Rescale k", value=1.2, minimum=0, step=0.05,
                                info="Temporal score rescaling. ~1.2 = smoother/flatter, ~0.96 = sharper/vivid.",
                            )
                            dub_rescale_sigma = gr.Number(
                                label="Rescale σ", value=3.0, minimum=0, step=0.1,
                                info="Rescaling bandwidth. 3.0 is the default.",
                            )

                # Hidden Force Speaker controls (not working well, kept for backend compatibility)
                dub_force_speaker = gr.Checkbox(value=False, visible=False)
                dub_speaker_kv_scale = gr.Number(value=1.5, visible=False)
                dub_speaker_kv_min_t = gr.Number(value=0.9, visible=False)
                dub_speaker_kv_max_layers = gr.Number(value=24, visible=False)

            dub_generate_btn = gr.Button("Generate & Dub", variant="primary", size="lg")

            gr.HTML('<hr class="section-sep">')

            # --- Output ---
            gr.Markdown("#### Output")
            dub_output_video = gr.Video(label="Dubbed Video", visible=False, elem_classes=["media-window"])
            
            # --- Save Controls ---
            with gr.Row():
                with gr.Group():
                    dub_save_video_btn = gr.Button("💾 Save to Outputs", size="sm")
                    dub_open_folder_btn = gr.Button("📂 Open Outputs Folder", size="sm")
                    dub_auto_save_checkbox = gr.Checkbox(label="Auto-save", value=False)
                dub_status = gr.Textbox(label="Status", show_label=False, interactive=False, lines=1.5, max_lines=3, scale=3)

            gr.HTML('<hr class="section-sep">')

        # ==============================================================
        # TAB 2: Voices
        # ==============================================================
        with gr.Tab("Voices", id="tab_voices"):

            # --- State for working audio ---
            vc_source_path = gr.State(None)    # original uploaded file path
            vc_working_path = gr.State(None)   # current working audio (after clip/trim/normalize)

            # --- Step 1: Source ---
            with gr.Accordion("1. Source Audio", open=True):
                gr.Markdown("Upload an audio or video file. Video audio is extracted automatically.")
                vc_upload = gr.File(
                    label="Upload Audio or Video",
                    file_types=["audio", "video"],
                    
                )
                with gr.Row():
                    vc_duration_info = gr.Textbox(
                        value="", label="Duration", interactive=False, max_lines=1, scale=2,
                    )
                    vc_extract_status = gr.Textbox(
                        value="", label="", interactive=False, max_lines=1, scale=3,
                    )

            # --- Step 2: Clip & Edit ---
            with gr.Accordion("2. Clip & Edit", open=True):
                with gr.Row():
                    vc_preview = gr.Audio(label="Preview", interactive=False, max_length=600)
                with gr.Row():
                    vc_start = gr.Number(label="Start (s)", container=False, value=0, minimum=0, step=0.5, precision=1, scale=1)
                    vc_end = gr.Number(label="End (s)", container=False, value=30, minimum=0, step=0.5, precision=1, scale=1)
                    vc_clip_btn = gr.Button("✂ Clip", size="sm", variant="primary", scale=1)
                with gr.Row():
                    vc_trim_btn = gr.Button("Remove Silence", size="sm", scale=1)
                    vc_normalize_btn = gr.Button("Normalize Volume", size="sm", scale=1)
                vc_edit_status = gr.Textbox(value="", label="", show_label=False, container=False, interactive=False, max_lines=1)

            # --- Step 3: Save ---
            with gr.Accordion("3. Save Character", open=True):
                with gr.Row():
                    new_voice_name = gr.Textbox(label="Character Name", placeholder="e.g. 'Morgan' or 'Narrator'", scale=2)
                    save_voice_btn = gr.Button("💾 Save", variant="primary", size="sm", scale=1)
                save_voice_status = gr.Textbox(label="", interactive=False, max_lines=2)

            gr.HTML('<hr class="section-sep">')

            # --- Manage Saved Voices ---
            with gr.Accordion("Saved Voices", open=False):
                with gr.Row():
                    manage_voice_dropdown = gr.Dropdown(
                        choices=["None"] + get_voice_names(), value="None",
                        label="Select Voice", interactive=True, scale=2,
                    )
                    delete_voice_btn = gr.Button("🗑 Delete", variant="stop", size="sm", scale=1)
                manage_voice_preview = gr.Audio(label="Preview", interactive=False)
                with gr.Row():
                    manage_voice_info = gr.Textbox(label="Info", interactive=False, lines=3, max_lines=4, scale=2)
                    delete_voice_status = gr.Textbox(label="", interactive=False, max_lines=2, scale=1)


        # ==============================================================
        # TAB 4: Settings
        # ==============================================================
        with gr.Tab("Settings", id="tab_settings"):
            gr.Markdown("### Settings")
            
            # Theme selector
            with gr.Group():
                gr.Markdown("**Appearance**")
                
                current_theme = load_ui_settings().get("theme", "Citrus")
                theme_choices = [
                    "Default", "Soft", "Monochrome", "Glass", "Base", "Ocean", "Origin", "Citrus",
                    "Miku", "Interstellar", "xkcd", "kotaemon"
                ]
                
                theme_dropdown = gr.Dropdown(
                    choices=theme_choices,
                    value=current_theme,
                    label="Theme",
                    info="Select a theme for the interface. Changes apply on next app startup.",
                )
                
                gr.Markdown("*Theme will apply on next app startup.*")
            
            gr.HTML('<hr class="section-sep">')
            
            # Model management
            with gr.Group():
                gr.Markdown("**Memory Management**")
                
                # Load current settings
                current_settings = load_ui_settings()
                current_fish_ae_dtype = current_settings.get("fish_ae_dtype", "float32")
                current_auto_unload = current_settings.get("auto_unload_models", False)
                
                fish_ae_dtype_dropdown = gr.Dropdown(
                    choices=["float32", "bfloat16"],
                    value=current_fish_ae_dtype,
                    label="Fish Autoencoder Dtype",
                    info="Use 'bfloat16' for 8GB GPUs to reduce memory usage. Requires app restart.",
                )
                
                gr.Markdown("*Dtype setting will apply on next app startup.*")
                
                gr.Markdown("---")
                
                auto_unload_checkbox = gr.Checkbox(
                    label="Auto-unload models after inference",
                    value=current_auto_unload,
                    info="Frees GPU/CPU memory after generation completes. Models will reload on next generation.",
                )
                
                unload_now_btn = gr.Button("Unload Models Now", variant="secondary")

            gr.HTML('<hr class="section-sep">')
            
            # File Management
            with gr.Group():
                gr.Markdown("**File Management**")
                
                # Load current settings
                current_output_dir = current_settings.get("output_dir", "./outputs")
                current_clear_temp = current_settings.get("clear_temp_on_start", False)
                
                output_dir_textbox = gr.Textbox(
                    label="Output Directory",
                    value=current_output_dir,
                    info="Custom location for saved audio files. Requires app restart.",
                    placeholder="./outputs"
                )
                
                gr.Markdown("*Output directory will apply on next app startup.*")
                
                gr.Markdown("---")
                
                clear_temp_on_start_checkbox = gr.Checkbox(
                    label="Clear temp files on app start",
                    value=current_clear_temp,
                    info="Automatically clears temporary audio and video files when the app starts.",
                )
                
                with gr.Row():
                    clear_temp_now_btn = gr.Button("Clear Temp Files Now", variant="secondary")
                    clear_temp_status = gr.Textbox(label="", show_label=False, interactive=False, scale=2)

            gr.HTML('<hr class="section-sep">')

    gr.Markdown(
        "Echo-TTS Model by [Jordan Darefsky](https://jordandarefsky.com/blog/2025/echo/) "
        "-- Outputs are subject to non-commercial use [CC-BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) due to the dependency on the [Fish Speech S1-DAC autoencoder](https://huggingface.co/fishaudio/s1-mini)"
    )
    # ==================================================================
    # Settings helpers
    # ==================================================================
    
    def on_auto_unload_toggle(value):
        """Handle auto-unload checkbox change."""
        global auto_unload_models
        auto_unload_models = value
        # Save to settings for persistence
        settings = load_ui_settings()
        settings["auto_unload_models"] = value
        save_ui_settings(settings)
    
    def on_unload_now():
        """Handle manual unload button."""
        model_manager.unload_all()
        unload_whisper()
        # return gr.update(value="Models unloaded")
    
    def on_theme_change(theme_name):
        """Handle theme dropdown change."""
        settings = load_ui_settings()
        settings["theme"] = theme_name
        save_ui_settings(settings)
        # if save_ui_settings(settings):
            # return gr.update(value=f"Theme '{theme_name}' saved. Restart the app to apply.")
        # else:
            # return gr.update(value="Failed to save theme setting.")
    
    def on_fish_ae_dtype_change(dtype_value):
        """Handle Fish AE dtype dropdown change."""
        settings = load_ui_settings()
        settings["fish_ae_dtype"] = dtype_value
        save_ui_settings(settings)
    
    def on_output_dir_change(dir_value):
        """Handle output directory textbox change."""
        settings = load_ui_settings()
        settings["output_dir"] = dir_value
        save_ui_settings(settings)
    
    def on_clear_temp_on_start_toggle(value):
        """Handle clear temp on start checkbox change."""
        settings = load_ui_settings()
        settings["clear_temp_on_start"] = value
        save_ui_settings(settings)
    
    def on_clear_temp_now():
        """Handle clear temp files now button."""
        cleared_count = 0
        for temp_dir in [TEMP_AUDIO_DIR, TEMP_DUB_DIR, GRADIO_TEMP_DIR]:
            if temp_dir.exists():
                for item in temp_dir.iterdir():
                    try:
                        if item.is_file():
                            item.unlink()
                            cleared_count += 1
                        elif item.is_dir():
                            shutil.rmtree(item)
                            cleared_count += 1
                    except Exception:
                        pass  # Ignore errors during cleanup
        return f"Cleared {cleared_count} items from temp directories"
    
    # ==================================================================
    # Event wiring
    # ==================================================================

    # ------------------------------------------------------------------
    # Settings tab events
    # ------------------------------------------------------------------

    # --- Per-speaker source toggles ---
    def _toggle_spk_source(source):
        return gr.update(visible=(source == "saved")), gr.update(visible=(source == "upload"))

    spk1_source.change(_toggle_spk_source, inputs=[spk1_source], outputs=[spk1_saved_group, spk1_upload_group])
    spk2_source.change(_toggle_spk_source, inputs=[spk2_source], outputs=[spk2_saved_group, spk2_upload_group])

    # Speaker dropdown previews
    def _on_spk_dropdown(voice_name):
        if not voice_name or voice_name in (NO_VOICE_LABEL, "Inactive"):
            return gr.update(value=None, visible=False)
        path = get_voice_preview_path(voice_name)
        return gr.update(value=path, visible=bool(path))

    spk1_dropdown.change(_on_spk_dropdown, inputs=[spk1_dropdown], outputs=[spk1_preview])
    spk2_dropdown.change(_on_spk_dropdown, inputs=[spk2_dropdown], outputs=[spk2_preview])

    all_spk_dropdowns = [spk1_dropdown, spk2_dropdown, dub_spk1_dropdown, dub_spk2_dropdown]

    def _refresh_all_spk_dropdowns():
        names = get_voice_names()
        spk1_update = gr.update(choices=[NO_VOICE_LABEL] + names)
        spk2_update = gr.update(choices=["Inactive", NO_VOICE_LABEL] + names)
        dub_spk1_update = gr.update(choices=[NO_VOICE_LABEL] + names)
        dub_spk2_update = gr.update(choices=["Inactive", NO_VOICE_LABEL] + names)
        return [spk1_update, spk2_update, dub_spk1_update, dub_spk2_update]

    # Preprocess / Clear text buttons
    def on_preprocess_text(text):
        if not text or not text.strip():
            return gr.update()
        chunks = chunk_text_by_time(text)
        if not chunks:
            return gr.update(value=preprocess_text(text))
        return gr.update(value=format_chunks(chunks))

    preprocess_btn.click(on_preprocess_text, inputs=[text_prompt], outputs=[text_prompt])
    clear_text_btn.click(lambda: gr.update(value=""), outputs=[text_prompt])

    # Force speaker toggle — show/hide KV scaling controls
    force_speaker.change(
        lambda enabled: gr.update(visible=enabled),
        inputs=[force_speaker], outputs=[kv_scaling_group],
    )

    # Manage voice preview
    def on_manage_voice_selected(name):
        if not name or name == "None":
            return gr.update(value=None), gr.update(value="")
        path = get_voice_preview_path(name)
        info_text = ""
        for v in list_voices():
            if v["name"] == name:
                info_text = f"Created: {v['created']}\nLatents cached: {v['has_latents']}\nFiles: {', '.join(v['audio_files'])}"
                break
        return gr.update(value=path), gr.update(value=info_text)

    manage_voice_dropdown.change(on_manage_voice_selected, inputs=[manage_voice_dropdown], outputs=[manage_voice_preview, manage_voice_info])

    # Audio library
    if AUDIO_PROMPT_FOLDER.exists():
        audio_prompt_table.select(select_audio_prompt_file, outputs=[uploaded_audio])
        audio_prompt_search.change(
            lambda q: gr.update(value=get_audio_prompt_files(q)),
            inputs=[audio_prompt_search], outputs=[audio_prompt_table],
        )

    # Text character count
    def update_char_count(text):
        if not text:
            return "0 chars | ~0s estimated"
        chars = len(text)
        est_seconds = max(chars / 14.0, len(text.split()) / 2.7)
        chunks = chunk_text_by_time(text)
        chunk_info = f" | {len(chunks)} chunk(s)" if len(chunks) > 1 else ""
        return f"{chars} chars | ~{est_seconds:.0f}s estimated{chunk_info}"

    text_prompt.change(update_char_count, inputs=[text_prompt], outputs=[char_count])

    # Sampler presets
    preset_dropdown.change(
        apply_sampler_preset, inputs=[preset_dropdown],
        outputs=[num_steps, cfg_scale_text, cfg_scale_speaker, cfg_min_t, cfg_max_t, truncation_factor, rescale_k, rescale_sigma],
    )

    # Generate — now outputs 4 values (added save_status)
    generate_btn.click(
        generate_audio,
        inputs=[
            text_prompt,
            spk1_source, spk1_dropdown, spk1_audio,
            spk2_source, spk2_dropdown, spk2_audio,
            num_steps, rng_seed,
            cfg_scale_text, cfg_scale_speaker, cfg_min_t, cfg_max_t,
            truncation_factor, rescale_k, rescale_sigma,
            force_speaker, speaker_kv_scale, speaker_kv_min_t, speaker_kv_max_layers,
            audio_format, session_id_state,
            inter_chunk_silence, crossfade_ms,
        ],
        outputs=[generated_audio, generation_time_display, text_display, save_status],
    ).then(
        auto_save_if_enabled,
        inputs=[auto_save_checkbox, audio_format],
        outputs=[save_status],
    )

    # Save / Open folder buttons
    save_audio_btn.click(save_to_outputs, inputs=[audio_format], outputs=[save_status])
    open_folder_btn.click(open_outputs_folder, outputs=[save_status])

    # ------------------------------------------------------------------
    # Dub tab events
    # ------------------------------------------------------------------

    # Speaker source toggles for Dub tab
    def toggle_dub_spk_source(source):
        is_saved = (source == "saved")
        is_upload = (source == "upload")
        return gr.update(visible=is_saved), gr.update(visible=is_upload)

    dub_spk1_source.change(
        toggle_dub_spk_source, 
        inputs=[dub_spk1_source], 
        outputs=[dub_spk1_saved_group, dub_spk1_upload_group]
    )
    dub_spk2_source.change(
        toggle_dub_spk_source, 
        inputs=[dub_spk2_source], 
        outputs=[dub_spk2_saved_group, dub_spk2_upload_group]
    )

    # Speaker dropdown previews for Dub tab
    def _on_dub_spk_dropdown(voice_name):
        if not voice_name or voice_name in (NO_VOICE_LABEL, "Inactive"):
            return gr.update(value=None, visible=False)
        path = get_voice_preview_path(voice_name)
        return gr.update(value=path, visible=bool(path))

    dub_spk1_dropdown.change(_on_dub_spk_dropdown, inputs=[dub_spk1_dropdown], outputs=[dub_spk1_preview])
    dub_spk2_dropdown.change(_on_dub_spk_dropdown, inputs=[dub_spk2_dropdown], outputs=[dub_spk2_preview])

    # Extract & Transcribe
    dub_extract_btn.click(
        dub_extract_and_transcribe,
        inputs=[dub_video_input, dub_whisper_model, dub_task],
        outputs=[dub_transcript, dub_detected_lang, dub_extracted_audio, dub_status, dub_extracted_audio_path, dub_segments_state, dub_video_duration_state],
    )

    # Generate & Dub
    dub_generate_btn.click(
        dub_generate_and_mux,
        inputs=[
            dub_video_input, dub_transcript,
            # Speaker 1
            dub_spk1_source, dub_spk1_dropdown, dub_spk1_audio,
            # Speaker 2
            dub_spk2_source, dub_spk2_dropdown, dub_spk2_audio,
            # Generation parameters
            dub_num_steps, dub_rng_seed,
            dub_cfg_text, dub_cfg_speaker, dub_cfg_min_t, dub_cfg_max_t,
            dub_truncation_factor, dub_rescale_k, dub_rescale_sigma,
            dub_force_speaker, dub_speaker_kv_scale, dub_speaker_kv_min_t, dub_speaker_kv_max_layers,
            dub_audio_format, session_id_state,
            dub_segments_state, dub_video_duration_state,
            dub_extracted_audio_path,
        ],
        outputs=[dub_output_video, dub_status],
    ).then(
        lambda auto_save, fmt: save_dub_to_outputs(fmt) if auto_save else gr.update(),
        inputs=[dub_auto_save_checkbox, dub_audio_format],
        outputs=[dub_status],
    )

    # Save / Open folder buttons for Dub tab
    dub_save_video_btn.click(save_dub_to_outputs, inputs=[dub_audio_format], outputs=[dub_status])
    dub_open_folder_btn.click(open_outputs_folder, outputs=[dub_status])

    # Session init + load first preset
    demo.load(init_session, outputs=[session_id_state]).then(
        lambda: apply_sampler_preset(list(load_sampler_presets().keys())[0]) if load_sampler_presets() else [gr.update()] * 8,
        outputs=[num_steps, cfg_scale_text, cfg_scale_speaker, cfg_min_t, cfg_max_t, truncation_factor, rescale_k, rescale_sigma],
    )

    # ------------------------------------------------------------------
    # Voices tab events
    # ------------------------------------------------------------------

    # Upload source — detect video vs audio, extract if needed, set duration
    def vc_on_upload(file_path):
        if not file_path:
            return None, None, gr.update(value=""), gr.update(value=""), gr.update(value=None), gr.update(value=0), gr.update(value=30)
        file_path = str(file_path)
        fp = Path(file_path)
        video_exts = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}
        try:
            if fp.suffix.lower() in video_exts:
                # Extract audio from video
                out_path = str(TEMP_DUB_DIR / "vc_extracted.wav")
                video_extract_audio(file_path, out_path)
                dur = get_media_duration(out_path)
                return (
                    out_path, out_path,
                    gr.update(value=f"{dur:.1f}s"),
                    gr.update(value="Audio extracted from video."),
                    gr.update(value=out_path),
                    gr.update(value=0),
                    gr.update(value=min(30, dur)),
                )
            else:
                dur = get_media_duration(file_path)
                return (
                    file_path, file_path,
                    gr.update(value=f"{dur:.1f}s"),
                    gr.update(value=""),
                    gr.update(value=file_path),
                    gr.update(value=0),
                    gr.update(value=min(30, dur)),
                )
        except Exception as e:
            return None, None, gr.update(value=""), gr.update(value=f"Error: {e}"), gr.update(value=None), gr.update(value=0), gr.update(value=30)

    vc_upload.change(
        vc_on_upload, inputs=[vc_upload],
        outputs=[vc_source_path, vc_working_path, vc_duration_info, vc_extract_status, vc_preview, vc_start, vc_end],
    )

    # Clip
    def vc_on_clip(source_path, start, end):
        if not source_path:
            return None, gr.update(value=None), gr.update(value="Upload audio first.")
        try:
            out = str(TEMP_DUB_DIR / "vc_clipped.wav")
            clip_audio(source_path, out, float(start), float(end))
            dur = get_media_duration(out)
            return out, gr.update(value=out), gr.update(value=f"Clipped to {dur:.1f}s")
        except Exception as e:
            return None, gr.update(), gr.update(value=f"Clip error: {e}")

    vc_clip_btn.click(
        vc_on_clip, inputs=[vc_source_path, vc_start, vc_end],
        outputs=[vc_working_path, vc_preview, vc_edit_status],
    )

    # Trim silence
    # Lets consider ffmpeg's silenceremove here for removing long pauses in the reference audio. with exposed useful parameters
    def vc_on_trim(working_path):
        if not working_path:
            return None, gr.update(), gr.update(value="No audio to trim.")
        try:
            out = str(TEMP_DUB_DIR / "vc_trimmed.wav")
            trim_silence(working_path, out)
            dur = get_media_duration(out)
            return out, gr.update(value=out), gr.update(value=f"Silence removed — {dur:.1f}s")
        except Exception as e:
            return None, gr.update(), gr.update(value=f"Trim error: {e}")

    vc_trim_btn.click(
        vc_on_trim, inputs=[vc_working_path],
        outputs=[vc_working_path, vc_preview, vc_edit_status],
    )

    # Normalize volume
    # lets consider ffmpeg-normalize instead here
    
    def vc_on_normalize(working_path):
        if not working_path:
            return None, gr.update(), gr.update(value="No audio to normalize.")
        try:
            out = str(TEMP_DUB_DIR / "vc_normalized.wav")
            normalize_audio(working_path, out)
            dur = get_media_duration(out)
            return out, gr.update(value=out), gr.update(value=f"Volume normalized — {dur:.1f}s")
        except Exception as e:
            return None, gr.update(), gr.update(value=f"Normalize error: {e}")

    vc_normalize_btn.click(
        vc_on_normalize, inputs=[vc_working_path],
        outputs=[vc_working_path, vc_preview, vc_edit_status],
    )

    # Save voice — uses working audio (clipped/edited)
    # We need to ensure that the save audio isn't too long.  in situation where a user has inputted a lengthy source via gr.File (no length parameters)
    # and tries to save a 30 minute voice reference file! we'll (likely) need to query `get_media_duration()` and block and inform to clip to 5 minutes max. 
    save_voice_btn.click(
        on_save_voice, inputs=[new_voice_name, vc_working_path],
        outputs=[spk1_dropdown, save_voice_status],
    ).then(
        _refresh_all_spk_dropdowns,
        outputs=all_spk_dropdowns,
    ).then(
        lambda: gr.update(choices=["None"] + get_voice_names()),
        outputs=[manage_voice_dropdown],
    )

    # Delete voice
    delete_voice_btn.click(
        on_delete_voice, inputs=[manage_voice_dropdown],
        outputs=[manage_voice_dropdown, delete_voice_status],
    ).then(
        _refresh_all_spk_dropdowns,
        outputs=all_spk_dropdowns,
    )

    # ------------------------------------------------------------------
    # Settings tab events
    # ------------------------------------------------------------------
    
    theme_dropdown.change(on_theme_change, inputs=[theme_dropdown], outputs=[])
    fish_ae_dtype_dropdown.change(on_fish_ae_dtype_change, inputs=[fish_ae_dtype_dropdown], outputs=[])
    auto_unload_checkbox.change(on_auto_unload_toggle, inputs=[auto_unload_checkbox], outputs=[])
    unload_now_btn.click(on_unload_now, outputs=[])
    output_dir_textbox.change(on_output_dir_change, inputs=[output_dir_textbox], outputs=[])
    clear_temp_on_start_checkbox.change(on_clear_temp_on_start_toggle, inputs=[clear_temp_on_start_checkbox], outputs=[])
    clear_temp_now_btn.click(on_clear_temp_now, outputs=[clear_temp_status])
    


if __name__ == "__main__":
    demo.launch(
        allowed_paths=[str(AUDIO_PROMPT_FOLDER), str(TEMP_AUDIO_DIR), str(OUTPUT_DIR), str(TEMP_DUB_DIR), str(GRADIO_TEMP_DIR)],
    )
