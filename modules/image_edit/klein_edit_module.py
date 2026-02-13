import math
import json
import time
from os import PathLike
from pathlib import Path
from typing import Optional

import torch
from pydantic import BaseModel, Field
from PIL import Image

from config import Settings
from logger_config import logger
from .klein_manager import KleinManager


class TextPrompting(BaseModel):
    prompt: str = Field(alias="positive")


class KleinEditModule(KleinManager):
    """FLUX.2-klein module for image editing operations."""

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self._empty_image = Image.new('RGB', (1024, 1024))

        self.prompt_path = settings.klein_edit_prompt_path
        self.prompting = self._load_prompting()

        self.pipe_config = {
            "num_inference_steps": settings.num_inference_steps,
            "guidance_scale": settings.guidance_scale,
            "height": settings.klein_edit_height,
            "width": settings.klein_edit_width,
        }

    def _load_prompting(self, path: Optional[PathLike] = None) -> TextPrompting:
        path = path or self.prompt_path
        with open(path, "r") as f:
            return TextPrompting.model_validate_json(json.dumps(json.load(f)))

    def _prepare_input_image(self, image: Image.Image, megapixels: float = 1.0) -> Image.Image:
        """Resize image to target megapixels while maintaining aspect ratio."""
        total = int(megapixels * 1024 * 1024)

        scale_by = math.sqrt(total / (image.width * image.height))
        width = round(image.width * scale_by)
        height = round(image.height * scale_by)

        return image.resize((width, height), Image.Resampling.LANCZOS)

    def _run_model_pipe(self, image: Image.Image, prompt: str, seed: Optional[int] = None):
        kwargs = dict(
            image=image,
            prompt=prompt,
            **self.pipe_config,
        )
        if seed:
            kwargs["generator"] = torch.Generator(device=self.device).manual_seed(seed)

        result = self.pipe(**kwargs)
        return result

    def edit_image(
        self,
        prompt_image: Image.Image,
        seed: int,
        prompt: Optional[str] = None
    ) -> Image.Image:
        """
        Edit the image using FLUX.2-klein.

        Args:
            prompt_image: The prompt image to edit.
            seed: Random seed for reproducibility.
            prompt: Optional prompt to override default prompting.

        Returns:
            The edited image.
        """
        if self.pipe is None:
            logger.error("FLUX.2-klein model is not loaded")
            raise RuntimeError("FLUX.2-klein model is not loaded")

        try:
            start_time = time.time()

            edit_prompt = prompt if prompt else self.prompting.prompt
            prepared_image = self._prepare_input_image(prompt_image)
            logger.info(f"Prompt image size: {prepared_image.size}")

            result = self._run_model_pipe(
                image=prepared_image,
                prompt=edit_prompt,
                seed=seed,
            )

            generation_time = time.time() - start_time
            image_edited = result.images[0]

            logger.success(f"Edited image generated in {generation_time:.2f}s, Size: {image_edited.size}, Seed: {seed}")

            return image_edited

        except Exception as e:
            logger.error(f"Error generating image: {e}")
            raise e

    def generate_multi_view_images(
        self,
        prompt_image: Image.Image,
        seed: int,
        views: list[str] = None
    ) -> list[Image.Image]:
        """
        Generate multi-view images from a single input image.

        Args:
            prompt_image: The input image.
            seed: Random seed for reproducibility.
            views: List of view types to generate. Options: 'left', 'right', 'back', 'original'.
                   Default: ['left', 'right', 'original']

        Returns:
            List of images for each requested view.
        """
        if views is None:
            views = ['left', 'right', 'original']

        view_prompts = {
            'left': self.settings.left_view_prompt,
            'right': self.settings.right_view_prompt,
            'back': self.settings.back_view_prompt,
        }

        images = []
        for view in views:
            if view == 'original':
                images.append(prompt_image)
            elif view in view_prompts:
                logger.info(f"Generating {view} view...")
                edited = self.edit_image(
                    prompt_image=prompt_image,
                    seed=seed,
                    prompt=view_prompts[view]
                )
                images.append(edited)
            else:
                logger.warning(f"Unknown view type: {view}, skipping...")

        return images
