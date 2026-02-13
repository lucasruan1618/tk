import gc
import argparse
import asyncio
from io import BytesIO
from time import time
from PIL import Image
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
import uvicorn
from loguru import logger
from fastapi import FastAPI, UploadFile, File, APIRouter, Form
from fastapi.responses import Response, StreamingResponse
from starlette.datastructures import State

import o_voxel
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from config import settings
from modules.image_edit import KleinEditModule


def get_args() -> argparse.Namespace:
    """ Function for getting arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10006)
    return parser.parse_args()


def clean_vram() -> None:
    """ Function for cleaning VRAM. """
    gc.collect()
    torch.cuda.empty_cache()


executor = ThreadPoolExecutor(max_workers=1)


class MyFastAPI(FastAPI):
    state: State
    router: APIRouter
    version: str


@asynccontextmanager
async def lifespan(app: MyFastAPI) -> AsyncIterator[None]:
    logger.info("Loading Trellis 2 generator models ...")
    try:
        app.state.trellis_generator = Trellis2ImageTo3DPipeline.from_pretrained(settings.trellis_model_id)
        app.state.trellis_generator.to("cuda")

    except Exception as e:
        logger.exception(f"Exception during Trellis model loading: {e}")
        raise SystemExit("Trellis model failed to load → exiting server")

    # Initialize FLUX.2-klein Edit Module (required for image editing)
    try:
        logger.info("Loading FLUX.2-klein Edit model ...")
        app.state.klein_edit = KleinEditModule(settings)
        await app.state.klein_edit.startup()
    except Exception as e:
        logger.exception(f"Exception during FLUX.2-klein model loading: {e}")
        raise SystemExit("FLUX.2-klein model failed to load → exiting server")

    # Warmup with real image to pre-allocate GPU memory
    logger.info("Warming up pipeline...")
    try:
        import os
        warmup_image_path = os.path.join(os.path.dirname(__file__), "warmup_image.png")
        warmup_image = Image.open(warmup_image_path)
        _ = generation_block(warmup_image, seed=42)
        clean_vram()
        logger.info("Warmup complete.")
    except Exception as e:
        logger.warning(f"Warmup failed (non-critical): {e}")
        clean_vram()

    yield

    # Cleanup
    if app.state.klein_edit is not None:
        await app.state.klein_edit.shutdown()


app = MyFastAPI(title="404 Base Miner Service", version="0.0.0")
app.router.lifespan_context = lifespan


def generation_block(prompt_image: Image.Image, seed: int = -1):
    """
    Function for 3D data generation using FLUX.2-klein-edited image.

    Pipeline flow:
      1. FLUX.2-klein image editing (view manipulation + cleanup)
      2. Trellis2 full pipeline (shape + texture)
      3. o-voxel postprocessing
      4. GLB export
    """
    edit_prompt = "Show this object in three-quarters view and make sure it is fully visible. Turn background neutral solid color contrasting with an object. Delete background details. Delete watermarks. Keep object colors. Sharpen image details"

    t_start = time()

    # Step 1: Edit image using FLUX.2-klein
    logger.info("Editing image with FLUX.2-klein...")
    edited_image = app.state.klein_edit.edit_image(
        prompt_image=prompt_image,
        seed=seed,
        prompt=edit_prompt
    )

    t_edit = time()
    logger.debug(f"FLUX.2-klein image editing took: {(t_edit - t_start):.2f} secs.")

    # Step 2: Run Trellis2 full pipeline
    logger.info("Running Trellis2 full pipeline...")
    mesh = app.state.trellis_generator.run(
        image=edited_image,
        seed=seed,
        pipeline_type="1024_cascade",
        sparse_structure_sampler_params={
            "steps": 12,
            "guidance_strength": 7.5,
        },
        shape_slat_sampler_params={
            "steps": 12,
            "guidance_strength": 3.0,
        },
        tex_slat_sampler_params={
            "steps": 12,
            "guidance_strength": 3.0,
        },
    )[0]

    mesh.simplify()

    # Step 3: o-voxel postprocessing and GLB export
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=1000000,
        texture_size=2048,
        remesh=True,
        remesh_band=1,
        remesh_project=0.9,
        verbose=True
    )

    buffer = BytesIO()
    glb.export(buffer, extension_webp=False, file_type="glb")
    buffer.seek(0)

    t_get_model = time()
    logger.debug(f"Total Model Generation took: {(t_get_model - t_start):.2f} secs.")

    clean_vram()

    t_gc = time()
    logger.debug(f"Garbage Collection took: {(t_gc - t_get_model):.2f} secs")

    return buffer


@app.post("/generate")
async def generate_model(prompt_image_file: UploadFile = File(...), seed: int = Form()) -> Response:
    """ Generates a 3D model as GLB file using FLUX.2-klein-edited single image """

    logger.info("Task received. Prompt-Image")

    contents = await prompt_image_file.read()
    prompt_image = Image.open(BytesIO(contents))

    loop = asyncio.get_running_loop()
    buffer = await loop.run_in_executor(executor, generation_block, prompt_image, seed)
    buffer_size = len(buffer.getvalue())
    buffer.seek(0)
    logger.info(f"Task completed.")

    async def generate_chunks():
        chunk_size = 1024 * 1024  # 1 MB
        while chunk := buffer.read(chunk_size):
            yield chunk

    return StreamingResponse(
        generate_chunks(),
        media_type="application/octet-stream",
        headers={"Content-Length": str(buffer_size)}
    )


@app.get("/version", response_model=str)
async def version() -> str:
    """ Returns current endpoint version."""
    return app.version


@app.get("/health")
def health_check() -> dict[str, str]:
    """ Return if the server is alive """
    return {"status": "healthy"}


if __name__ == "__main__":
    args: argparse.Namespace = get_args()
    uvicorn.run(app, host=args.host, port=args.port, reload=False)
