import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
import base64
import requests
import os
from dotenv import load_dotenv
import cv2
import mediapipe as mp
import numpy as np
import os
from pathlib import Path
from loguru import logger


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()


from nodes import NODE_CLASS_MAPPINGS

BASE_PROMPT = """Realistic skin texture, young woman's face 20-25 years old is illuminated by soft, 
        natural light, highlighting her calm and confident expression. Her smooth, fair skin contrasts with her striking,
        thick eyebrows and sharp, defined cheekbones. Her full, slightly parted lips have a natural pink hue,
        adding to her elegance. Her bright, piercing green eyes are framed by long lashes, 
        drawing attention to her gaze. The light blonde hair flows naturally around her face, 
        and a silver necklace with a winged pendant adds a touch of sophistication. Super Realism."""

def main(in_img, in_face_img, input_prompt = BASE_PROMPT, output_path = "output"):
    import_custom_nodes()
    with torch.inference_mode():
        unetloader = NODE_CLASS_MAPPINGS["UNETLoader"]()
        unetloader_63 = unetloader.load_unet(
            unet_name="flux1-dev.safetensors", weight_dtype="default"
        )

        dualcliploader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
        dualcliploader_64 = dualcliploader.load_clip(
            clip_name1="t5xxl_fp16.safetensors",
            clip_name2="clip_l.safetensors",
            type="flux",
            device="default",
        )

        loraloader = NODE_CLASS_MAPPINGS["LoraLoader"]()
        loraloader_122 = loraloader.load_lora(
            lora_name="super-realism.safetensors",
            strength_model=0.35000000000000003,
            strength_clip=0.6,
            model=get_value_at_index(unetloader_63, 0),
            clip=get_value_at_index(dualcliploader_64, 0),
        )

        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        cliptextencode_6 = cliptextencode.encode(
            text=input_prompt,
            clip=get_value_at_index(loraloader_122, 1),
        )

        vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
        vaeloader_10 = vaeloader.load_vae(
            vae_name="diffusion_pytorch_model.safetensors"
        )

        ksamplerselect = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
        ksamplerselect_16 = ksamplerselect.get_sampler(sampler_name="euler")

        randomnoise = NODE_CLASS_MAPPINGS["RandomNoise"]()
        randomnoise_25 = randomnoise.get_noise(noise_seed=random.randint(1, 2**64))

        pulidfluxmodelloader = NODE_CLASS_MAPPINGS["PulidFluxModelLoader"]()
        pulidfluxmodelloader_45 = pulidfluxmodelloader.load_model(
            pulid_file="pulid_flux_v0.9.1.safetensors"
        )

        pulidfluxevacliploader = NODE_CLASS_MAPPINGS["PulidFluxEvaClipLoader"]()
        pulidfluxevacliploader_51 = pulidfluxevacliploader.load_eva_clip()

        pulidfluxinsightfaceloader = NODE_CLASS_MAPPINGS["PulidFluxInsightFaceLoader"]()
        pulidfluxinsightfaceloader_53 = pulidfluxinsightfaceloader.load_insightface(
            provider="CUDA"
        )

        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        loadimage_54 = loadimage.load_image(image=in_face_img)

        fluxguidance = NODE_CLASS_MAPPINGS["FluxGuidance"]()
        fluxguidance_26 = fluxguidance.append(
            guidance=2.5, conditioning=get_value_at_index(cliptextencode_6, 0)
        )

        loadimage_104 = loadimage.load_image(image=in_img)

        inpaintmodelconditioning = NODE_CLASS_MAPPINGS["InpaintModelConditioning"]()
        inpaintmodelconditioning_70 = inpaintmodelconditioning.encode(
            noise_mask=True,
            positive=get_value_at_index(fluxguidance_26, 0),
            negative=get_value_at_index(fluxguidance_26, 0),
            vae=get_value_at_index(vaeloader_10, 0),
            pixels=get_value_at_index(loadimage_104, 0),
            mask=get_value_at_index(loadimage_104, 1),
        )

        basicscheduler = NODE_CLASS_MAPPINGS["BasicScheduler"]()
        applypulidflux = NODE_CLASS_MAPPINGS["ApplyPulidFlux"]()
        basicguider = NODE_CLASS_MAPPINGS["BasicGuider"]()
        samplercustomadvanced = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()

        for q in range(1):
            basicscheduler_17 = basicscheduler.get_sigmas(
                scheduler="simple",
                steps=35,
                denoise=0.4,
                model=get_value_at_index(loraloader_122, 0),
            )

            applypulidflux_62 = applypulidflux.apply_pulid_flux(
                weight=0.9,
                start_at=0,
                end_at=0.8,
                fusion="mean",
                fusion_weight_max=1,
                fusion_weight_min=0,
                train_step=1000,
                use_gray=True,
                model=get_value_at_index(loraloader_122, 0),
                pulid_flux=get_value_at_index(pulidfluxmodelloader_45, 0),
                eva_clip=get_value_at_index(pulidfluxevacliploader_51, 0),
                face_analysis=get_value_at_index(pulidfluxinsightfaceloader_53, 0),
                image=get_value_at_index(loadimage_54, 0),
                unique_id=3909983138228746112,
            )

            basicguider_47 = basicguider.get_guider(
                model=get_value_at_index(applypulidflux_62, 0),
                conditioning=get_value_at_index(inpaintmodelconditioning_70, 0),
            )

            samplercustomadvanced_92 = samplercustomadvanced.sample(
                noise=get_value_at_index(randomnoise_25, 0),
                guider=get_value_at_index(basicguider_47, 0),
                sampler=get_value_at_index(ksamplerselect_16, 0),
                sigmas=get_value_at_index(basicscheduler_17, 0),
                latent_image=get_value_at_index(inpaintmodelconditioning_70, 2),
            )

            vaedecode_49 = vaedecode.decode(
                samples=get_value_at_index(samplercustomadvanced_92, 0),
                vae=get_value_at_index(vaeloader_10, 0),
            )

            saveimage_123 = saveimage.save_images(
                filename_prefix="faceswap_res", images=get_value_at_index(vaedecode_49, 0)
            )
        return saveimage_123



def get_face_img_w_mask(in_img, save_dir='input'):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
    
    if isinstance(in_img, str):
        image = cv2.imread(in_img)
        input_name = Path(in_img).stem
    else:
        image = in_img
        input_name = 'face_mask'
        
    if image is None:
        return None
        
    height, width = image.shape[:2]

    rgba_image = np.zeros((height, width, 4), dtype=np.uint8)
    rgba_image[:,:,:3] = image
    

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    

    mask = np.ones((height, width), dtype=np.uint8) * 255
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            points = []
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                points.append([x, y])
            
            points = np.array(points, dtype=np.int32)
            hull = cv2.convexHull(points)
            

            cv2.fillConvexPoly(mask, hull, 0)

        rgba_image[:,:,3] = mask
        os.makedirs(save_dir, exist_ok=True)

        filename = f"{input_name}_w_mask.png"
        output_path = os.path.join(save_dir, filename)
        cv2.imwrite(output_path, rgba_image)
        return filename
    else:
        return None



def get_prompt_by_img_path(in_img_path, input_prompt):
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
        
    with open(in_img_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": "chatgpt-4o-latest",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "I have a different description prompt for a face, I need a description of image, essentially replacing all the entities with the ones relevant to the photo: " + input_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 1000
    }
    
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()

        result = response.json()
        generated_prompt = result["choices"][0]["message"]["content"]
        
        return generated_prompt.strip()
        
    except Exception as e:
        print(f"Error making OpenAI API request: {e}")
        return None

def faceswap_process(
    image_path="input/image2.webp",
    face_img_path="example_face.jpg",
    input_prompt=BASE_PROMPT,
    output_dir="output/"
):
    
    logger.info(f"Using image mask: {image_path}")
    logger.info(f"Using face image: {face_img_path}")

    in_img_w_mask_path = get_face_img_w_mask(image_path)
    if in_img_w_mask_path is None:
        logger.error(f"Failed to get face image with mask for {image_path}")
        return
    logger.info(f"Received face image: {in_img_w_mask_path}")
    
    logger.info(f"Receiving input prompt")
    
    input_prompt = get_prompt_by_img_path(face_img_path, BASE_PROMPT)
    logger.info(f"Input prompt: {input_prompt}")
    
    # comfyui path hack
    if face_img_path.startswith("input/"):
        face_img_path = face_img_path.split("input/")[1]
    output = main(in_img_w_mask_path, face_img_path, input_prompt, output_dir)
    output_img = output["ui"]["images"][0]["filename"]
    logger.info(f"Output image: {output_dir}/{output_img}")


if __name__ == "__main__":
    faceswap_process()

