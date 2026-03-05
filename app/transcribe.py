"""Transcription module for Echo TTS Studio.

Provides a clean abstraction over speech-to-text backends.
Currently uses OpenAI Whisper. Designed to be swappable — all consumers
use TranscriptionResult and the transcribe() function, never Whisper directly.

To swap backends later, only this file needs to change.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch

# Lazy-loaded to avoid import cost at startup
_whisper_model = None
_whisper_model_name = None


@dataclass
class WordSegment:
    """A single word with timing information."""
    word: str
    start: float  # seconds
    end: float    # seconds


@dataclass
class Segment:
    """A transcribed segment (typically a sentence or phrase)."""
    text: str
    start: float
    end: float
    words: List[WordSegment] = field(default_factory=list)


@dataclass
class TranscriptionResult:
    """Standard transcription output format.

    All backends should produce this structure.
    """
    text: str                          # Full transcribed text
    language: str                      # Detected language code (e.g. 'en', 'es')
    segments: List[Segment]            # Time-aligned segments
    duration: float                    # Total audio duration in seconds
    backend: str = "whisper"           # Which backend produced this


def _load_whisper(model_name: str = "turbo"):
    """Lazy-load whisper model. Cached after first call."""
    global _whisper_model, _whisper_model_name
    if _whisper_model is not None and _whisper_model_name == model_name:
        return _whisper_model

    import whisper
    print(f"[transcribe] Loading Whisper model '{model_name}'...")
    t0 = time.time()
    _whisper_model = whisper.load_model(model_name)
    _whisper_model_name = model_name
    print(f"[transcribe] Whisper loaded in {time.time() - t0:.1f}s")
    return _whisper_model


def transcribe(
    audio_path: str,
    model_name: str = "turbo",
    language: Optional[str] = None,
    task: str = "transcribe",
) -> TranscriptionResult:
    """Transcribe an audio file.

    Args:
        audio_path: Path to audio file (any format ffmpeg supports).
        model_name: Whisper model size ('tiny', 'base', 'small', 'medium', 'large', 'turbo').
        language: Force language code, or None for auto-detect.
        task: 'transcribe' for same-language, 'translate' for translate-to-English.

    Returns:
        TranscriptionResult with text, language, and timed segments.
    """
    model = _load_whisper(model_name)

    print(f"[transcribe] Processing: {audio_path} (task={task}, lang={language or 'auto'})")
    t0 = time.time()

    opts = {
        "task": task,
        "word_timestamps": True,
        "verbose": False,
    }
    if language:
        opts["language"] = language

    result = model.transcribe(str(audio_path), **opts)

    # Parse into our standard format
    segments = []
    for seg in result.get("segments", []):
        words = []
        for w in seg.get("words", []):
            words.append(WordSegment(
                word=w.get("word", "").strip(),
                start=w.get("start", 0.0),
                end=w.get("end", 0.0),
            ))
        segments.append(Segment(
            text=seg.get("text", "").strip(),
            start=seg.get("start", 0.0),
            end=seg.get("end", 0.0),
            words=words,
        ))

    # Compute total duration from last segment end
    duration = segments[-1].end if segments else 0.0

    detected_lang = result.get("language", "en")
    full_text = result.get("text", "").strip()

    elapsed = time.time() - t0
    print(f"[transcribe] Done in {elapsed:.1f}s — language: {detected_lang}, segments: {len(segments)}")

    return TranscriptionResult(
        text=full_text,
        language=detected_lang,
        segments=segments,
        duration=duration,
        backend="whisper",
    )


def translate_to_english(
    audio_path: str,
    model_name: str = "turbo",
) -> TranscriptionResult:
    """Convenience: transcribe + translate to English in one pass."""
    return transcribe(audio_path, model_name=model_name, task="translate")


def unload_model():
    """Unload the Whisper model from GPU/RAM to free VRAM."""
    global _whisper_model, _whisper_model_name
    if _whisper_model is not None:
        del _whisper_model
        _whisper_model = None
        _whisper_model_name = None
        torch.cuda.empty_cache()
        print("[transcribe] Whisper model unloaded, VRAM freed")


# Available model sizes for UI dropdown
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large", "turbo"]
