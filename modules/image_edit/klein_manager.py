from __future__ import annotations

import gc
import hashlib
import time
from dataclasses import dataclass

import torch
from diffusers import Flux2KleinPipeline
from PIL import Image

from config import Settings
from logger_config import logger
from hf_revisions import get_revision


@dataclass(slots=True)
class KleinResult:
    image: Image.Image
    generation_time: float
    seed: int


class KleinManager:
    """Handles FLUX.2-klein pipeline responsible for generating/editing 2D images."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.pipe = None
        self.device = f"cuda:{settings.klein_gpu}" if torch.cuda.is_available() else "cpu"
        self.dtype = self._resolve_dtype(settings.dtype)
        self.gpu_index = settings.klein_gpu

    async def startup(self) -> None:
        """Initialize the FLUX.2-klein pipeline."""
        logger.info("Initializing KleinManager...")
        self._load_pipeline()
        logger.success("KleinManager ready.")

    async def shutdown(self) -> None:
        """Shutdown the pipeline and free resources."""
        if self.pipe:
            try:
                self.pipe.to("cpu")
            except Exception:
                pass
        self.pipe = None
        self._clean_gpu_memory()
        logger.info("KleinManager closed.")

    def is_ready(self) -> bool:
        """Check if pipeline is loaded and ready."""
        return self.pipe is not None

    def _load_pipeline(self) -> None:
        """Load the FLUX.2-klein pipeline."""
        if torch.cuda.is_available():
            try:
                torch.cuda.set_device(self.gpu_index)
            except Exception as err:
                logger.warning(f"Failed to set CUDA device ({self.gpu_index}): {err}")

        t1 = time.time()

        self.pipe = Flux2KleinPipeline.from_pretrained(
            self.settings.klein_model_id,
            torch_dtype=self.dtype,
            revision=get_revision(self.settings.klein_model_id),
        )
        self.pipe.to(self.device)

        load_time = time.time() - t1
        logger.success(
            f"FLUX.2-klein pipeline ready (loading: {load_time:.2f}s). "
            f"gpu_id={self.gpu_index}, dtype={self.dtype}."
        )

    def _resolve_dtype(self, dtype: str) -> torch.dtype:
        mapping = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        resolved = mapping.get(dtype.lower(), torch.bfloat16)
        if not torch.cuda.is_available() and resolved in {torch.float16, torch.bfloat16}:
            return torch.float32
        return resolved

    def _derive_seed(self, prompt: str) -> int:
        hash_object = hashlib.md5(prompt.encode("utf-8"))
        return int(hash_object.hexdigest()[:8], 16) % (2**32)

    def _clean_gpu_memory(self) -> None:
        """Clean GPU memory."""
        gc.collect()
        torch.cuda.empty_cache()
