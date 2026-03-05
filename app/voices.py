"""Voice/character management for Echo TTS.

Handles saving, loading, listing, and deleting named voice profiles.
Each voice is stored as a directory containing the reference audio file(s)
and precomputed speaker latents for fast reuse.
"""

import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio

VOICES_DIR = Path(__file__).resolve().parent / "voices"
VOICES_DIR.mkdir(parents=True, exist_ok=True)

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus", ".webm", ".aac"}
SAMPLE_RATE = 44_100


def _voice_dir(name: str) -> Path:
    """Return the directory for a named voice."""
    safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in name).strip()
    return VOICES_DIR / safe


def list_voices() -> List[Dict]:
    """Return list of saved voices with metadata."""
    voices = []
    if not VOICES_DIR.exists():
        return voices
    for entry in sorted(VOICES_DIR.iterdir()):
        if not entry.is_dir():
            continue
        meta_path = entry / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            # Check if latents are cached
            has_latents = (entry / "speaker_latent.pt").exists()
            voices.append({
                "name": meta.get("name", entry.name),
                "created": meta.get("created", ""),
                "audio_files": meta.get("audio_files", []),
                "has_latents": has_latents,
                "dir": str(entry),
            })
        except (json.JSONDecodeError, OSError):
            continue
    return voices


def get_voice_names() -> List[str]:
    """Return sorted list of voice names for dropdown."""
    return [v["name"] for v in list_voices()]


def save_voice(
    name: str,
    audio_path: str,
    speaker_latent: Optional[torch.Tensor] = None,
    speaker_mask: Optional[torch.Tensor] = None,
) -> Path:
    """Save a voice profile from an audio file.

    Copies the audio into the voice directory and optionally caches
    precomputed speaker latents for instant loading.
    """
    vdir = _voice_dir(name)
    vdir.mkdir(parents=True, exist_ok=True)

    src = Path(audio_path)
    if not src.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Copy audio file
    dst = vdir / f"reference{src.suffix}"
    shutil.copy2(str(src), str(dst))

    # Save latents if provided
    if speaker_latent is not None and speaker_mask is not None:
        torch.save({
            "speaker_latent": speaker_latent.cpu(),
            "speaker_mask": speaker_mask.cpu(),
        }, str(vdir / "speaker_latent.pt"))

    # Write metadata
    meta = {
        "name": name,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "audio_files": [dst.name],
    }
    (vdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return vdir


def load_voice_audio(name: str) -> Optional[str]:
    """Return path to the voice's reference audio file, or None."""
    vdir = _voice_dir(name)
    if not vdir.exists():
        return None
    for f in vdir.iterdir():
        if f.is_file() and f.suffix.lower() in AUDIO_EXTS:
            return str(f)
    return None


def load_voice_latents(name: str, device: str = "cuda") -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Load cached speaker latents if available."""
    vdir = _voice_dir(name)
    latent_path = vdir / "speaker_latent.pt"
    if not latent_path.exists():
        return None
    data = torch.load(str(latent_path), map_location=device, weights_only=True)
    return data["speaker_latent"], data["speaker_mask"]


def delete_voice(name: str) -> bool:
    """Delete a voice profile entirely."""
    vdir = _voice_dir(name)
    if vdir.exists() and vdir.is_dir():
        shutil.rmtree(str(vdir))
        return True
    return False


def get_voice_preview_path(name: str) -> Optional[str]:
    """Return path to voice audio for preview playback."""
    return load_voice_audio(name)
