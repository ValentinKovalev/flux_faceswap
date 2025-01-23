from huggingface_hub import hf_hub_download
import os
from pathlib import Path
import zipfile
from loguru import logger
import shutil


def download_models():

    HF_TOKEN = os.getenv("HF_TOKEN")

    logger.info(f"Creating directories for models")
    pulid_dir = Path("./models/pulid")
    clip_dir = Path("./models/clip")
    insightface_model_dir = Path("./models/insightface/models")
    loras_dir = Path("./models/loras")
    flux1_dir = Path("./models/unet")
    vae_dir = Path("./models/vae")

    # Create directories if they don't exist
    pulid_dir.mkdir(parents=True, exist_ok=True)
    clip_dir.mkdir(parents=True, exist_ok=True)
    insightface_model_dir.mkdir(parents=True, exist_ok=True)
    loras_dir.mkdir(parents=True, exist_ok=True)
    flux1_dir.mkdir(parents=True, exist_ok=True)
    vae_dir.mkdir(parents=True, exist_ok=True)

    # download EVA clip pt
    logger.info(f"Downloading EVA clip pt")
    hf_hub_download(
        repo_id="microsoft/LLM2CLIP-EVA02-L-14-336",
        filename="LLM2CLIP-EVA02-L-14-336.pt",
        local_dir=clip_dir.absolute().as_posix(),
        token=HF_TOKEN,
    )

    # download t5xxl_fp16 safetensors
    logger.info(f"Downloading t5xxl_fp16 safetensors")
    hf_hub_download(
        repo_id="comfyanonymous/flux_text_encoders",
        filename="t5xxl_fp16.safetensors",
        local_dir=clip_dir.absolute().as_posix(),
        token=HF_TOKEN,
    )

    # download clip_l safetensors
    logger.info(f"Downloading clip_l safetensors")
    hf_hub_download(
        repo_id="comfyanonymous/flux_text_encoders",
        filename="clip_l.safetensors",
        local_dir=clip_dir.absolute().as_posix(),
        token=HF_TOKEN,
    )

    # download insightface model
    logger.info(f"Downloading insightface model")
    hf_hub_download(
        repo_id="tau-vision/insightface-antelopev2",
        filename="antelopev2.zip",
        local_dir=insightface_model_dir.absolute().as_posix(),
        token=HF_TOKEN,
    )

    # Unzip the antelopev2.zip file
    logger.info(f"Unzipping antelopev2.zip file")
    zip_path = os.path.join(insightface_model_dir, "antelopev2.zip")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(insightface_model_dir)

    # Path(zip_path).unlink()

    # download lora(s) model
    logger.info(f"Downloading Flux-Super-Realism-LoRA model")
    hf_hub_download(
        repo_id="strangerzonehf/Flux-Super-Realism-LoRA",
        filename="super-realism.safetensors",
        local_dir=loras_dir.absolute().as_posix(),
        token=HF_TOKEN,
    )

    # download pulid model
    logger.info(f"Downloading PuLID model")
    hf_hub_download(
        repo_id="guozinan/PuLID",
        filename="pulid_flux_v0.9.1.safetensors",
        local_dir=pulid_dir.absolute().as_posix(),
        token=HF_TOKEN,
    )

    # download flux1 model
    logger.info(f"Downloading FLUX.1-dev model")
    hf_hub_download(
        repo_id="black-forest-labs/FLUX.1-dev",
        filename="flux1-dev.safetensors",
        local_dir=flux1_dir.absolute().as_posix(),
        token=HF_TOKEN,
    )

    # download vae model
    logger.info(f"Downloading vae model")
    hf_hub_download(
        repo_id="black-forest-labs/FLUX.1-dev",
        filename="vae/diffusion_pytorch_model.safetensors",
        local_dir=vae_dir.absolute().as_posix(),
        token=HF_TOKEN,
    )
    # Move vae folder contents one level up
    vae_model_path = os.path.join(vae_dir, "vae", "diffusion_pytorch_model.safetensors")
    if os.path.exists(vae_model_path):
        logger.info("Moving VAE model one level up")
        shutil.move(vae_model_path, os.path.join(vae_dir, "diffusion_pytorch_model.safetensors"))
        os.rmdir(os.path.join(vae_dir, "vae"))


if __name__ == "__main__":
    download_models()
