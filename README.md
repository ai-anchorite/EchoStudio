<div align="center">

## Echo TTS Studio

**Local text-to-speech with voice cloning, video dubbing, and voice editing.**


[![Install with Pinokio](https://img.shields.io/badge/Install%20with-Pinokio-blue?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJ3aGl0ZSI+PHBhdGggZD0iTTEyIDJMMiAyMmgyMEwxMiAyeiIvPjwvc3ZnPg==)](https://beta.pinokio.co/apps/github-com-ai-anchorite-echostudio)

</div>

<p align="center">
  <img src="icon.png" alt="EchoStudio" width="240">
</p>

<div align="left">

## What is this?

An enhanced, one-click installable studio built on the [Echo-TTS](https://github.com/jordandare/echo-tts.git) project by Jordan Darefsky — enhanced for creators who want voice cloning, video dubbing, and voice editing without the setup headaches.

**Model:** [jordand/echo-tts-base](https://huggingface.co/jordand/echo-tts-base) | **Blog:** [echo-tts blog post](https://jordandarefsky.com/blog/2025/echo/)

## 🖥️ Requirements

- **VRAM:** 12 GB minimum (NVIDIA GPU recommended)
- **Platform:** Windows · Linux · macOS
- **Install:** One click via [Pinokio](https://beta.pinokio.co/apps/github-com-ai-anchorite-echostudio) — handles Python, dependencies, and model downloads automatically

## Getting Started

1. Install [Pinokio](https://pinokio.co/download.html) if you haven't already
2. Click the install badge above, or search "EchoStudio" in the Pinokio app
3. Hit Install → Start → done

## Features

### TTS
- Voice cloning from reference audio
- Multi-speaker support (S1/S2 tagging)
- Long-form generation with automatic text chunking and crossfade stitching
- Sampler presets and full control over CFG guidance, sampling style, and KV scaling

### Dub
- Upload video, extract audio, and transcribe/translate with Whisper
- Editable transcript with segment timing
- Re-voice with TTS using cloned or saved voices
- Preserve background audio — AI source separation mixes ambient/background with the new TTS voice
- Multi-speaker dubbing with S1/S2 tags

### Voices
- Upload audio or video files as voice sources
- Edit saved voices directly via "Send to Edit"
- Clip, trim silence, adjust speed, and normalize volume
- Vocal isolation — separate clean vocals from noisy recordings (BS-Roformer, MDX-Net via audio-separator)
- Background isolation for extracting ambience/music
- Save edited voices as named profiles with cached speaker latents

### Settings
- Theme selection, memory management, custom output directory, temp file cleanup


## Tips

### Generation Length

Echo generates up to 30 seconds of audio per chunk. Longer text is automatically split and stitched with configurable silence gaps and crossfade. Shorter text produces shorter outputs naturally.

### Reference Audio

Up to 5 minutes of reference audio is supported, but shorter clips (10 seconds or less) work well too. Use the Voices tab to clip, clean, and isolate vocals from noisy recordings.

### Force Speaker (KV Scaling)

If the model generates a different speaker than expected, enable "Force Speaker" (default scale 1.5). Aim for the lowest scale that produces the correct speaker.

### Text Prompt Format

Use `[S1]` and `[S2]` for speaker tags. Expression markers like `(laughs)`, `(angry)`, `(whispering)` control tone. Commas function as pauses.

## Responsible Use

Don't use this model to impersonate real people without consent or generate deceptive audio. You are responsible for complying with local laws regarding biometric data and voice cloning.

## License

Code in this repo is MIT-licensed except where file headers specify otherwise (e.g., autoencoder.py is Apache-2.0).

Audio outputs are CC-BY-NC-SA-4.0 due to the dependency on the [Fish Speech S1-DAC autoencoder](https://huggingface.co/fishaudio/s1-mini). Echo-TTS weights are released under CC-BY-NC-SA-4.0.

## Citation

```bibtex
@misc{darefsky2025echo,
    author = {Darefsky, Jordan},
    title = {Echo-TTS},
    year = {2025},
    url = {https://jordandarefsky.com/blog/2025/echo/}
}
```

---
