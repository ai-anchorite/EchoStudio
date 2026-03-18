"""Model management with lazy loading and device detection."""

import gc
import torch
from contextlib import contextmanager
from typing import Optional
from inference import (
    load_model_from_hf,
    load_fish_ae_from_hf,
    load_pca_state_from_hf,
)
from model import EchoDiT
from autoencoder import DAC
from inference import PCAState


def get_device() -> str:
    """Detect available device: cuda, mps (Apple Silicon), or cpu."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


class ModelManager:
    """Manages lazy loading and unloading of TTS models."""

    def __init__(
        self,
        model_dtype: torch.dtype = torch.bfloat16,
        fish_ae_dtype: torch.dtype = torch.float32,
        device: Optional[str] = None,
        token: Optional[str] = None,
        offload_fish_ae: bool = False,
    ):
        self.model_dtype = model_dtype
        self.fish_ae_dtype = fish_ae_dtype
        self.device = device or get_device()
        self.token = token
        self.offload_fish_ae = offload_fish_ae

        self._model: Optional[EchoDiT] = None
        self._fish_ae: Optional[DAC] = None
        self._pca_state: Optional[PCAState] = None

    def pre_download_models(self) -> None:
        """Pre-download models without loading them into memory."""
        print("[init] Pre-downloading models...")
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            "jordand/echo-tts-base",
            "pytorch_model.safetensors",
            token=self.token
        )
        hf_hub_download(
            "jordand/fish-s1-dac-min",
            "pytorch_model.safetensors",
            token=self.token
        )
        hf_hub_download(
            "jordand/echo-tts-base",
            "pca_state.safetensors",
            token=self.token
        )
        print("[init] Models pre-downloaded.")

    @property
    def model(self) -> EchoDiT:
        """Lazy-load main model."""
        if self._model is None:
            print(f"[model] Loading main model on device: {self.device}")
            self._model = load_model_from_hf(
                dtype=self.model_dtype,
                device=self.device,
                token=self.token,
                delete_blockwise_modules=True,
            )
            print("[model] Main model loaded.")
        return self._model

    @property
    def fish_ae(self) -> DAC:
        """Lazy-load fish-ae model."""
        if self._fish_ae is None:
            load_device = "cpu" if self.offload_fish_ae else self.device
            print(f"[model] Loading fish-ae on device: {load_device}" +
                  (" (offload mode)" if self.offload_fish_ae else ""))
            self._fish_ae = load_fish_ae_from_hf(
                dtype=self.fish_ae_dtype,
                device=load_device,
                token=self.token,
            )
            print("[model] Fish-ae model loaded.")
        return self._fish_ae

    def fish_ae_on_device(self):
        """Context manager: swaps main model off GPU, brings fish_ae on.

        When offload_fish_ae is enabled, the main EchoDiT model's real
        (non-meta) parameters are temporarily moved to CPU to free VRAM,
        then fish_ae is moved to GPU for encode/decode. Afterward, fish_ae
        goes back to CPU and the main model returns to GPU.

        Usage:
            with model_manager.fish_ae_on_device() as ae:
                latent = ae_encode(ae, pca_state, audio)
        """
        @contextmanager
        def _ctx():
            ae = self.fish_ae
            if not self.offload_fish_ae or str(next(ae.parameters()).device) == self.device:
                yield ae
            else:
                # Move main model's real parameters off GPU to make room
                # (some params may be meta tensors from deleted blockwise modules)
                moved_params = []
                if self._model is not None:
                    print("[model] Temporarily offloading main model to CPU")
                    for param in self._model.parameters():
                        if not param.is_meta and param.device.type != "cpu":
                            moved_params.append((param, param.data.device))
                            param.data = param.data.to("cpu")
                    for buf in self._model.buffers():
                        if not buf.is_meta and buf.device.type != "cpu":
                            moved_params.append((buf, buf.device))
                            buf.data = buf.data.to("cpu")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                print(f"[model] Moving fish-ae to {self.device}")
                ae.to(self.device)
                try:
                    yield ae
                finally:
                    ae.to("cpu")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print("[model] Fish-ae moved back to CPU")

                    if moved_params:
                        print(f"[model] Restoring main model to {self.device}")
                        for tensor, orig_device in moved_params:
                            tensor.data = tensor.data.to(orig_device)

        return _ctx()

    @property
    def pca_state(self) -> PCAState:
        """Lazy-load PCA state."""
        if self._pca_state is None:
            print(f"[model] Loading PCA state on device: {self.device}")
            self._pca_state = load_pca_state_from_hf(
                device=self.device,
                token=self.token,
            )
            print("[model] PCA state loaded.")
        return self._pca_state

    def _unload_codec(self) -> None:
        """Unload the codec (fish_ae) from memory."""
        if self._fish_ae is not None:
            del self._fish_ae
            self._fish_ae = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def unload_all(self) -> None:
        """Unload all models from memory."""
        if self._model is not None:
            del self._model
            self._model = None
        self._unload_codec()
        if self._pca_state is not None:
            del self._pca_state
            self._pca_state = None

        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        print("[model] All models unloaded, memory freed")

    def are_models_loaded(self) -> bool:
        """Check if any models are currently loaded."""
        return self._model is not None or self._fish_ae is not None or self._pca_state is not None
