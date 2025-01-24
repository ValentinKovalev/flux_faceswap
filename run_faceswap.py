import os
import sys
from loguru import logger
import argparse

original_args = sys.argv.copy()

parser = argparse.ArgumentParser(description="Run face swap pipeline", add_help=False)
parser.add_argument(
    "--input-image", default="input/image2.webp", type=str, required=False, help="Path to input image"
)
parser.add_argument(
    "--face-image", default="input/example_face.jpg", type=str, required=False, help="Path to face image to swap"
)
parser.add_argument(
    "--prompt", type=str, required=False, help="Custom prompt for generation (optional)"
)
parser.add_argument(
    "--output",
    type=str,
    required=False,
    default="output",
    help="Output directory (default: output)",
)

args, remaining = parser.parse_known_args()

sys.argv = [sys.argv[0]] + remaining

from faceswap_pipeline import (
    faceswap_process,
    add_comfyui_directory_to_sys_path,
    add_extra_model_paths,
    import_custom_nodes,
)

if __name__ == "__main__":
    add_comfyui_directory_to_sys_path()
    add_extra_model_paths()
    import_custom_nodes()

    faceswap_process(args.input_image, args.face_image, args.prompt, args.output)
