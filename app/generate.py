"""Long-form generation engine for Echo TTS.

Handles chunked text-to-speech generation with crossfade stitching,
supporting both single-shot (short text) and multi-chunk (long text) modes.
"""

import time
from dataclasses import dataclass
from functools import partial
from typing import Callable, List, Optional, Tuple

import torch
import torchaudio

from inference import (
    ae_decode,
    ae_encode,
    crop_audio_to_flattening_point,
    find_flattening_point,
    get_speaker_latent_and_mask,
    get_text_input_ids_and_mask,
    load_audio,
    sample_euler_cfg_independent_guidances,
    sample_pipeline,
    PCAState,
)
from autoencoder import DAC
from model import EchoDiT
from utils import chunk_text_by_time, preprocess_text, format_chunks

SAMPLE_RATE = 44_100
CROSSFADE_MS = 80  # milliseconds of crossfade between chunks
CROSSFADE_SAMPLES = int(SAMPLE_RATE * CROSSFADE_MS / 1000)
INTER_CHUNK_SILENCE_MS = 350  # silence gap between chunks for natural pacing
INTER_CHUNK_SILENCE_SAMPLES = int(SAMPLE_RATE * INTER_CHUNK_SILENCE_MS / 1000)


def _crossfade_chunks(
    chunks: List[torch.Tensor],
    crossfade_samples: int = CROSSFADE_SAMPLES,
    silence_samples: int = INTER_CHUNK_SILENCE_SAMPLES,
) -> torch.Tensor:
    """Concatenate audio chunks with a silence gap and smooth crossfade."""
    if not chunks:
        raise ValueError("No audio chunks to concatenate")
    if len(chunks) == 1:
        return chunks[0]

    result = chunks[0]
    for i in range(1, len(chunks)):
        next_chunk = chunks[i]
        if next_chunk.numel() == 0:
            continue

        # Insert silence gap between chunks for natural pacing
        if silence_samples > 0:
            silence = torch.zeros(*result.shape[:-1], silence_samples, device=result.device)
            result = torch.cat([result, silence], dim=-1)

        fade_len = min(crossfade_samples, result.shape[-1], next_chunk.shape[-1])
        if fade_len <= 0:
            result = torch.cat([result, next_chunk], dim=-1)
            continue

        # Equal-power crossfade
        t = torch.linspace(0.0, 1.0, fade_len, device=result.device)
        fade_out = torch.cos(t * 3.14159 / 2)
        fade_in = torch.sin(t * 3.14159 / 2)

        # Blend the overlap region
        overlap = result[..., -fade_len:] * fade_out + next_chunk[..., :fade_len] * fade_in
        result = torch.cat([result[..., :-fade_len], overlap, next_chunk[..., fade_len:]], dim=-1)

    return result


def generate_single_chunk(
    model: EchoDiT,
    fish_ae: DAC,
    pca_state: PCAState,
    text: str,
    speaker_latent: torch.Tensor,
    speaker_mask: torch.Tensor,
    rng_seed: int,
    num_steps: int = 40,
    cfg_scale_text: float = 3.0,
    cfg_scale_speaker: float = 5.0,
    cfg_min_t: float = 0.5,
    cfg_max_t: float = 1.0,
    truncation_factor: float = 0.8,
    rescale_k: Optional[float] = 1.2,
    rescale_sigma: float = 3.0,
    speaker_kv_scale: Optional[float] = None,
    speaker_kv_min_t: Optional[float] = None,
    speaker_kv_max_layers: Optional[int] = None,
    sample_latent_length: int = 640,
) -> Tuple[torch.Tensor, str]:
    """Generate audio for a single text chunk using the standard pipeline."""
    sample_fn = partial(
        sample_euler_cfg_independent_guidances,
        num_steps=num_steps,
        cfg_scale_text=cfg_scale_text,
        cfg_scale_speaker=cfg_scale_speaker,
        cfg_min_t=cfg_min_t,
        cfg_max_t=cfg_max_t,
        truncation_factor=truncation_factor,
        rescale_k=rescale_k if rescale_k != 1.0 else None,
        rescale_sigma=rescale_sigma,
        speaker_kv_scale=speaker_kv_scale,
        speaker_kv_min_t=speaker_kv_min_t,
        speaker_kv_max_layers=speaker_kv_max_layers,
        sequence_length=sample_latent_length,
    )

    audio_out, normalized_text = sample_pipeline(
        model=model,
        fish_ae=fish_ae,
        pca_state=pca_state,
        sample_fn=sample_fn,
        text_prompt=text,
        speaker_audio=None,  # We pass latents directly below
        rng_seed=rng_seed,
        normalize_text=False,  # Already preprocessed by chunker
    )

    return audio_out, normalized_text


def generate_single_chunk_with_latents(
    model: EchoDiT,
    fish_ae: DAC,
    pca_state: PCAState,
    text: str,
    speaker_latent: torch.Tensor,
    speaker_mask: torch.Tensor,
    rng_seed: int,
    num_steps: int = 40,
    cfg_scale_text: float = 3.0,
    cfg_scale_speaker: float = 5.0,
    cfg_min_t: float = 0.5,
    cfg_max_t: float = 1.0,
    truncation_factor: float = 0.8,
    rescale_k: Optional[float] = 1.2,
    rescale_sigma: float = 3.0,
    speaker_kv_scale: Optional[float] = None,
    speaker_kv_min_t: Optional[float] = None,
    speaker_kv_max_layers: Optional[int] = None,
    sample_latent_length: int = 640,
) -> torch.Tensor:
    """Generate audio for a single chunk using precomputed speaker latents directly."""
    device, dtype = model.device, model.dtype

    text_input_ids, text_mask, normalized_text = get_text_input_ids_and_mask(
        [text], max_length=768, device=device,
        normalize=False,  # Already preprocessed
        return_normalized_text=True,
        pad_to_max=False,
    )

    latent_out = sample_euler_cfg_independent_guidances(
        model=model,
        speaker_latent=speaker_latent,
        speaker_mask=speaker_mask,
        text_input_ids=text_input_ids,
        text_mask=text_mask,
        rng_seed=rng_seed,
        num_steps=num_steps,
        cfg_scale_text=cfg_scale_text,
        cfg_scale_speaker=cfg_scale_speaker,
        cfg_min_t=cfg_min_t,
        cfg_max_t=cfg_max_t,
        truncation_factor=truncation_factor,
        rescale_k=rescale_k if rescale_k != 1.0 else None,
        rescale_sigma=rescale_sigma,
        speaker_kv_scale=speaker_kv_scale,
        speaker_kv_min_t=speaker_kv_min_t,
        speaker_kv_max_layers=speaker_kv_max_layers,
        sequence_length=sample_latent_length,
    )

    audio_out = ae_decode(fish_ae, pca_state, latent_out)
    audio_out = crop_audio_to_flattening_point(audio_out, latent_out[0])

    return audio_out


def generate_long_form(
    model: EchoDiT,
    fish_ae: DAC,
    pca_state: PCAState,
    text: str,
    speaker_audio: Optional[torch.Tensor],
    speaker_latent: Optional[torch.Tensor],
    speaker_mask: Optional[torch.Tensor],
    rng_seed: int,
    num_steps: int = 40,
    cfg_scale_text: float = 3.0,
    cfg_scale_speaker: float = 5.0,
    cfg_min_t: float = 0.5,
    cfg_max_t: float = 1.0,
    truncation_factor: float = 0.8,
    rescale_k: Optional[float] = 1.2,
    rescale_sigma: float = 3.0,
    speaker_kv_scale: Optional[float] = None,
    speaker_kv_min_t: Optional[float] = None,
    speaker_kv_max_layers: Optional[int] = None,
    sample_latent_length: int = 640,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    crossfade_ms: int = CROSSFADE_MS,
    inter_chunk_silence_ms: int = INTER_CHUNK_SILENCE_MS,
) -> Tuple[torch.Tensor, str, List[str]]:
    """Generate long-form audio by chunking text and stitching with crossfade.

    Args:
        text: Full text to generate (can be arbitrarily long).
        speaker_audio: Raw speaker audio tensor, or None.
        speaker_latent: Precomputed speaker latent, or None.
        speaker_mask: Precomputed speaker mask, or None.
        progress_callback: Optional fn(chunk_idx, total_chunks, chunk_text) for UI updates.

    Returns:
        (audio_tensor, normalized_full_text, chunk_texts)
    """
    device, dtype = model.device, model.dtype

    # Compute speaker latents if not provided
    no_speaker = speaker_latent is None and speaker_mask is None and speaker_audio is None
    if speaker_latent is None or speaker_mask is None:
        if speaker_audio is not None:
            speaker_latent, speaker_mask = get_speaker_latent_and_mask(
                fish_ae, pca_state,
                speaker_audio.to(fish_ae.dtype).to(device),
            )
        else:
            speaker_latent = torch.zeros((1, 4, 80), device=device, dtype=dtype)
            speaker_mask = torch.zeros((1, 4), device=device, dtype=torch.bool)

    # Ensure latent dimensions are divisible by patch size
    speaker_latent = speaker_latent[:, :speaker_latent.shape[1] // 4 * 4]
    speaker_mask = speaker_mask[:, :speaker_mask.shape[1] // 4 * 4]

    # Chunk the text
    chunks = chunk_text_by_time(text)
    if not chunks:
        chunks = [preprocess_text(text)]

    total = len(chunks)
    audio_chunks: List[torch.Tensor] = []

    print(f"[generate] Long-form: {total} chunk(s) to generate")

    for i, chunk_text in enumerate(chunks):
        # Use same seed for all chunks when no speaker reference (voice consistency)
        # Vary seed per chunk when speaker is provided (natural variation, voice is anchored)
        chunk_seed = rng_seed if no_speaker else rng_seed + i

        if progress_callback:
            progress_callback(i, total, chunk_text)

        print(f"[generate] Chunk {i+1}/{total}: {chunk_text[:80]}...")
        t0 = time.time()

        chunk_audio = generate_single_chunk_with_latents(
            model=model,
            fish_ae=fish_ae,
            pca_state=pca_state,
            text=chunk_text,
            speaker_latent=speaker_latent,
            speaker_mask=speaker_mask,
            rng_seed=chunk_seed,
            num_steps=num_steps,
            cfg_scale_text=cfg_scale_text,
            cfg_scale_speaker=cfg_scale_speaker,
            cfg_min_t=cfg_min_t,
            cfg_max_t=cfg_max_t,
            truncation_factor=truncation_factor,
            rescale_k=rescale_k,
            rescale_sigma=rescale_sigma,
            speaker_kv_scale=speaker_kv_scale,
            speaker_kv_min_t=speaker_kv_min_t,
            speaker_kv_max_layers=speaker_kv_max_layers,
            sample_latent_length=sample_latent_length,
        )

        elapsed = time.time() - t0
        audio_dur = chunk_audio.shape[-1] / SAMPLE_RATE
        print(f"[generate] Chunk {i+1}/{total}: {elapsed:.1f}s gen -> {audio_dur:.1f}s audio")

        flat_audio = chunk_audio[0] if chunk_audio.ndim == 3 else chunk_audio
        audio_chunks.append(flat_audio)

        # Self-clone: after first chunk with no speaker, encode its audio as the
        # speaker reference for all subsequent chunks. This locks the voice identity.
        if no_speaker and i == 0 and total > 1:
            print("[generate] Self-cloning: encoding chunk 1 audio as speaker reference for remaining chunks")
            with torch.inference_mode():
                clone_audio = flat_audio.unsqueeze(0) if flat_audio.ndim == 1 else flat_audio
                if clone_audio.ndim == 2:
                    clone_audio = clone_audio.unsqueeze(0)  # (1, 1, samples) -> need (1, samples)
                    clone_audio = clone_audio.squeeze(0)     # back to (1, samples)
                speaker_latent, speaker_mask = get_speaker_latent_and_mask(
                    fish_ae, pca_state,
                    clone_audio.to(fish_ae.dtype).to(device),
                )
                speaker_latent = speaker_latent[:, :speaker_latent.shape[1] // 4 * 4]
                speaker_mask = speaker_mask[:, :speaker_mask.shape[1] // 4 * 4]
            no_speaker = False  # Now we have a speaker, vary seeds for natural variation

    # Stitch chunks with crossfade
    if len(audio_chunks) > 1:
        cf_samples = int(SAMPLE_RATE * crossfade_ms / 1000)
        sil_samples = int(SAMPLE_RATE * inter_chunk_silence_ms / 1000)
        final_audio = _crossfade_chunks(audio_chunks, crossfade_samples=cf_samples, silence_samples=sil_samples)
    else:
        final_audio = audio_chunks[0]

    # Clone to escape inference mode before in-place ops
    final_audio = final_audio.clone()

    # Apply gentle fadeout at the very end to avoid clicks
    fadeout_samples = min(int(0.05 * SAMPLE_RATE), final_audio.shape[-1])
    if fadeout_samples > 0:
        t = torch.linspace(0.0, 1.0, fadeout_samples, device=final_audio.device)
        fade = (1.0 - t) ** 3
        final_audio[..., -fadeout_samples:] = final_audio[..., -fadeout_samples:] * fade

    normalized_text = format_chunks(chunks)

    return final_audio, normalized_text, chunks


# -------------------------------------------------------------------
# Segment-aware dubbed audio generation
# -------------------------------------------------------------------

@dataclass
class DubSegment:
    """A segment of text with timing from the source transcription."""
    text: str
    start: float  # seconds
    end: float    # seconds

    @property
    def duration(self) -> float:
        return self.end - self.start


def generate_dubbed_audio(
    model: EchoDiT,
    fish_ae: DAC,
    pca_state: PCAState,
    segments: List[DubSegment],
    total_duration: float,
    speaker_audio: Optional[torch.Tensor],
    speaker_latent: Optional[torch.Tensor],
    speaker_mask: Optional[torch.Tensor],
    rng_seed: int,
    num_steps: int = 40,
    cfg_scale_text: float = 3.0,
    cfg_scale_speaker: float = 5.0,
    cfg_min_t: float = 0.5,
    cfg_max_t: float = 1.0,
    truncation_factor: float = 0.8,
    rescale_k: Optional[float] = 1.2,
    rescale_sigma: float = 3.0,
    speaker_kv_scale: Optional[float] = None,
    speaker_kv_min_t: Optional[float] = None,
    speaker_kv_max_layers: Optional[int] = None,
    sample_latent_length: int = 640,
    stretch_range: Tuple[float, float] = (0.7, 1.5),
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Tuple[torch.Tensor, List[dict]]:
    """Generate time-aligned dubbed audio from transcription segments.

    For each segment:
      1. Generate TTS audio for the segment text
      2. Time-stretch to fit the original segment duration (within bounds)
      3. Place at the correct position in the timeline

    Gaps between segments are filled with silence.

    Args:
        segments: List of DubSegment with text and timing.
        total_duration: Total duration of the source video in seconds.
        stretch_range: (min_ratio, max_ratio) for time stretch. Outside this
                       range, audio is clamped rather than distorted.
        progress_callback: Optional fn(segment_idx, total, segment_text).

    Returns:
        (final_audio_tensor, segment_info_list)
        segment_info_list contains per-segment metadata for debugging/UI.
    """
    from video import time_stretch_audio
    import tempfile

    device, dtype = model.device, model.dtype

    # Resolve speaker latents
    no_speaker = speaker_latent is None and speaker_mask is None and speaker_audio is None
    if speaker_latent is None or speaker_mask is None:
        if speaker_audio is not None:
            speaker_latent, speaker_mask = get_speaker_latent_and_mask(
                fish_ae, pca_state,
                speaker_audio.to(fish_ae.dtype).to(device),
            )
        else:
            speaker_latent = torch.zeros((1, 4, 80), device=device, dtype=dtype)
            speaker_mask = torch.zeros((1, 4), device=device, dtype=torch.bool)

    speaker_latent = speaker_latent[:, :speaker_latent.shape[1] // 4 * 4]
    speaker_mask = speaker_mask[:, :speaker_mask.shape[1] // 4 * 4]

    # Allocate the full-length output buffer (silence)
    total_samples = int(total_duration * SAMPLE_RATE)
    final_audio = torch.zeros(1, total_samples, device="cpu")

    total = len(segments)
    segment_info = []

    print(f"[dub] Generating {total} segment(s) for {total_duration:.1f}s video")

    for i, seg in enumerate(segments):
        if not seg.text.strip():
            segment_info.append({"idx": i, "text": "", "skipped": True})
            continue

        chunk_seed = rng_seed + i if not no_speaker else rng_seed

        if progress_callback:
            progress_callback(i, total, seg.text)

        print(f"[dub] Segment {i+1}/{total} [{seg.start:.1f}s-{seg.end:.1f}s]: {seg.text[:60]}...")
        t0 = time.time()

        # Generate TTS for this segment
        chunk_audio = generate_single_chunk_with_latents(
            model=model, fish_ae=fish_ae, pca_state=pca_state,
            text=preprocess_text(seg.text),
            speaker_latent=speaker_latent, speaker_mask=speaker_mask,
            rng_seed=chunk_seed,
            num_steps=num_steps,
            cfg_scale_text=cfg_scale_text,
            cfg_scale_speaker=cfg_scale_speaker,
            cfg_min_t=cfg_min_t,
            cfg_max_t=cfg_max_t,
            truncation_factor=truncation_factor,
            rescale_k=rescale_k,
            rescale_sigma=rescale_sigma,
            speaker_kv_scale=speaker_kv_scale,
            speaker_kv_min_t=speaker_kv_min_t,
            speaker_kv_max_layers=speaker_kv_max_layers,
            sample_latent_length=sample_latent_length,
        )

        flat_audio = chunk_audio[0] if chunk_audio.ndim == 3 else chunk_audio
        if flat_audio.ndim == 1:
            flat_audio = flat_audio.unsqueeze(0)

        gen_duration = flat_audio.shape[-1] / SAMPLE_RATE
        target_duration = seg.duration
        speed_ratio = gen_duration / target_duration if target_duration > 0 else 1.0

        elapsed = time.time() - t0
        print(f"[dub] Segment {i+1}/{total}: {elapsed:.1f}s gen -> {gen_duration:.1f}s audio (target: {target_duration:.1f}s, ratio: {speed_ratio:.2f}x)")

        # Self-clone after first segment if no speaker provided
        if no_speaker and i == 0 and total > 1:
            print("[dub] Self-cloning: encoding segment 1 audio as speaker reference")
            with torch.inference_mode():
                clone_audio = flat_audio.to(fish_ae.dtype).to(device)
                speaker_latent, speaker_mask = get_speaker_latent_and_mask(
                    fish_ae, pca_state, clone_audio,
                )
                speaker_latent = speaker_latent[:, :speaker_latent.shape[1] // 4 * 4]
                speaker_mask = speaker_mask[:, :speaker_mask.shape[1] // 4 * 4]
            no_speaker = False

        # Time-stretch if needed (and within reasonable bounds)
        min_ratio, max_ratio = stretch_range
        stretched_audio = flat_audio.cpu()

        if abs(speed_ratio - 1.0) > 0.05 and target_duration > 0.5:
            # Only stretch if the ratio is within our quality bounds
            clamped_ratio = max(min_ratio, min(max_ratio, speed_ratio))

            if abs(clamped_ratio - 1.0) > 0.05:
                try:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as src_f:
                        src_path = src_f.name
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as dst_f:
                        dst_path = dst_f.name

                    torchaudio.save(src_path, stretched_audio, SAMPLE_RATE)
                    effective_target = gen_duration / clamped_ratio
                    time_stretch_audio(src_path, dst_path, effective_target)
                    stretched_audio, sr = torchaudio.load(dst_path)

                    # Clean up temp files
                    import os
                    os.unlink(src_path)
                    os.unlink(dst_path)

                    print(f"[dub] Segment {i+1}: stretched {gen_duration:.1f}s -> {stretched_audio.shape[-1]/SAMPLE_RATE:.1f}s")
                except Exception as e:
                    print(f"[dub] Segment {i+1}: stretch failed ({e}), using raw audio")

        # Place audio at the correct position in the timeline
        start_sample = int(seg.start * SAMPLE_RATE)
        available_samples = total_samples - start_sample
        seg_samples = min(stretched_audio.shape[-1], available_samples, int(target_duration * SAMPLE_RATE))

        if seg_samples > 0 and start_sample < total_samples:
            # Ensure mono/matching channels
            if stretched_audio.shape[0] > 1:
                stretched_audio = stretched_audio[:1]
            final_audio[:, start_sample:start_sample + seg_samples] = stretched_audio[:, :seg_samples]

        segment_info.append({
            "idx": i,
            "text": seg.text,
            "start": seg.start,
            "end": seg.end,
            "gen_duration": gen_duration,
            "placed_duration": seg_samples / SAMPLE_RATE if seg_samples > 0 else 0,
            "speed_ratio": speed_ratio,
            "skipped": False,
        })

    # Gentle fadeout at the end
    final_audio = final_audio.clone()
    fadeout_samples = min(int(0.05 * SAMPLE_RATE), final_audio.shape[-1])
    if fadeout_samples > 0:
        t = torch.linspace(0.0, 1.0, fadeout_samples)
        fade = (1.0 - t) ** 3
        final_audio[..., -fadeout_samples:] = final_audio[..., -fadeout_samples:] * fade

    return final_audio, segment_info
