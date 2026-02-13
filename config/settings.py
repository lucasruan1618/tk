from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings

config_dir = Path(__file__).parent


class Settings(BaseSettings):
    api_title: str = "404 Base Miner Service"

    # API settings
    host: str = "0.0.0.0"
    port: int = 10006

    # GPU settings
    klein_gpu: int = Field(default=0, env="KLEIN_GPU")
    trellis_gpu: int = Field(default=0, env="TRELLIS_GPU")
    dtype: str = Field(default="bf16", env="KLEIN_DTYPE")

    # Hugging Face settings
    hf_token: Optional[str] = Field(default=None, env="HF_TOKEN")

    # Trellis settings
    trellis_model_id: str = Field(default="microsoft/TRELLIS.2-4B", env="TRELLIS_MODEL_ID")

    # FLUX.2-klein Edit settings
    klein_model_id: str = Field(
        default="black-forest-labs/FLUX.2-klein-4B",
        env="KLEIN_MODEL_ID"
    )
    klein_edit_height: int = Field(default=1024, env="KLEIN_EDIT_HEIGHT")
    klein_edit_width: int = Field(default=1024, env="KLEIN_EDIT_WIDTH")
    num_inference_steps: int = Field(default=4, env="NUM_INFERENCE_STEPS")
    guidance_scale: float = Field(default=4.0, env="GUIDANCE_SCALE")
    klein_edit_prompt_path: Path = Field(
        default=config_dir.joinpath("klein_edit_prompt.json"),
        env="KLEIN_EDIT_PROMPT_PATH"
    )

    # Multi-view generation prompts
    left_view_prompt: str = Field(
        default="Show this object in left three-quarters view and make sure it is fully visible. Turn background neutral solid color contrasting with an object. Delete background details. Delete watermarks. Keep object colors. Sharpen image details",
        env="LEFT_VIEW_PROMPT"
    )
    right_view_prompt: str = Field(
        default="Show this object in right three-quarters view and make sure it is fully visible. Turn background neutral solid color contrasting with an object. Delete background details. Delete watermarks. Keep object colors. Sharpen image details",
        env="RIGHT_VIEW_PROMPT"
    )
    back_view_prompt: str = Field(
        default="Show this object from the back view and make sure it is fully visible. Turn background neutral solid color contrasting with an object. Delete background details. Delete watermarks. Keep object colors. Sharpen image details",
        env="BACK_VIEW_PROMPT"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

__all__ = ["Settings", "settings"]
