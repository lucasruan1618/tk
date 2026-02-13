"""
Pinned HuggingFace model revisions for reproducibility.

All external HuggingFace models/repos are pinned to specific commit hashes
to ensure reproducible builds as required by competition rules.
"""

HF_MODEL_REVISIONS: dict[str, str] = {
    "microsoft/TRELLIS.2-4B": "af44b45f2e35a493886929c6d786e563ec68364d",
    "microsoft/TRELLIS-image-large": "25e0d31ffbebe4b5a97464dd851910efc3002d96",
    "phunghuy159/dinov3": "58720787075743137b49d0c12cf6ad9220d7be21",
    "ZhengPeng7/BiRefNet": "26e7919b869d089e5096e898c0492898f935604c",
    "black-forest-labs/FLUX.2-klein-4B": "5e67da950fce4a097bc150c22958a05716994cea",
}


def get_revision(repo_id: str) -> str | None:
    """Look up the pinned revision for a HuggingFace repo."""
    return HF_MODEL_REVISIONS.get(repo_id)
