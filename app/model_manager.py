"""Model management with lazy loading and device detection."""

import gc
import torch
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
    ):
        self.model_dtype = model_dtype
        self.fish_ae_dtype = fish_ae_dtype
        self.device = device or get_device()
        self.token = token
        
        self._model: Optional[EchoDiT] = None
        self._fish_ae: Optional[DAC] = None
        self._pca_state: Optional[PCAState] = None
    
    def pre_download_models(self) -> None:
        """Pre-download models without loading them into memory.
        
        This ensures models are cached locally before inference starts,
        so they load quickly on-demand.
        """
        print("[init] Pre-downloading models...")
        from huggingface_hub import hf_hub_download
        
        # Download main model
        hf_hub_download(
            "jordand/echo-tts-base",
            "pytorch_model.safetensors",
            token=self.token
        )
        
        # Download fish-ae model
        hf_hub_download(
            "jordand/fish-s1-dac-min",
            "pytorch_model.safetensors",
            token=self.token
        )
        
        # Download PCA state
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
            print(f"[model] Loading fish-ae model on device: {self.device}")
            self._fish_ae = load_fish_ae_from_hf(
                dtype=self.fish_ae_dtype,
                device=self.device,
                token=self.token,
            )
            print("[model] Fish-ae model loaded.")
        return self._fish_ae
    
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
    
    def unload_all(self) -> None:
        """Unload all models from memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._fish_ae is not None:
            del self._fish_ae
            self._fish_ae = None
        if self._pca_state is not None:
            del self._pca_state
            self._pca_state = None
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        print("[model] All models unloaded, memory freed")
    
    def are_models_loaded(self) -> bool:
        """Check if any models are currently loaded."""
        return self._model is not None or self._fish_ae is not None or self._pca_state is not None
