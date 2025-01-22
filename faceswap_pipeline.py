import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch


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


def main():
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
            strength_model=0.3,
            strength_clip=0.7000000000000001,
            model=get_value_at_index(unetloader_63, 0),
            clip=get_value_at_index(dualcliploader_64, 0),
        )

        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        cliptextencode_6 = cliptextencode.encode(
            text="realistic skin texture, woman's face is illuminated by cold, dim light, highlighting her tense expression. Her skin is glistening with sweat, accentuating her sharp cheekbones and slightly parted lips, which suggest a mix of shock and fear. Faint dirt smudges and subtle abrasions on her face contribute to an atmosphere of distress and urgency.",
            clip=get_value_at_index(loraloader_122, 1),
        )

        vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
        vaeloader_10 = vaeloader.load_vae(vae_name="ae.safetensors")

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
        loadimage_54 = loadimage.load_image(
            image="clipspace/clipspace-mask-405645.2999999523.png [input]"
        )

        fluxguidance = NODE_CLASS_MAPPINGS["FluxGuidance"]()
        fluxguidance_26 = fluxguidance.append(
            guidance=3.5, conditioning=get_value_at_index(cliptextencode_6, 0)
        )

        loadimage_104 = loadimage.load_image(
            image="clipspace/clipspace-mask-2266529.4000000954.png [input]"
        )

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

        for q in range(1):
            basicscheduler_17 = basicscheduler.get_sigmas(
                scheduler="simple",
                steps=35,
                denoise=0.4,
                model=get_value_at_index(loraloader_122, 0),
            )

            applypulidflux_62 = applypulidflux.apply_pulid_flux(
                weight=0.8,
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
                unique_id=15121207877576056355,
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
            def save_image(image, filename):
                image.save(filename)
            save_image(get_value_at_index(vaedecode_49, 0), f"output_{q}.png")

if __name__ == "__main__":
    main()
