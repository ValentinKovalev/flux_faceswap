# FaceSwap Diffusion

A project for face swapping using ComfyUI and Stable Diffusion. Supports face swapping in both single images and videos.

## Pipeline
- Receiving segmentation mask for generation with mediapipe (could be improved)
- Generating prompt using gpt VLM for face description (using base prompt as a format for better results), could be used florence-2, but gpt-4o is better
- Using PulID with Flux.1-dev SOTA model
- Using inpainting model
- Using lora for realism and better results
- Add finetuned lora on a bunch of photos at workflow level, results are below


## Installation
 - install python 3.10
 - export HF_TOKEN=your_huggingface_token
 - export OPENAI_API_KEY=your_openai_api_key
 - initialize git submodules:
 ```bash
 git submodule update --init --recursive
 ```
 - install poetry
 - ```poetry install``` using .lock file
 - ```poetry run python download_models.py```
 - download models from huggingface with ```poetry run python download_models.py```
 - ```poetry run python run_faceswap.py```

## Use your own images
- ```python run_faceswap.py --input-image absolute/path/to/image --face-image /absolute/path/to/face/image```

## Next steps:
- [ ] Replace mediapipe with something more accurate, that works with small faces and make accurate face segmentation mask
- [ ] Add and experiment with different loras for improving realism
- [ ] Using only face crop for generation and insert face crop with new face in original image
- [ ] Imporve base prompt for better face annotation
- [ ] Replace mediapipe with something more accurate
- [ ] Prod ready solution without using comfyui
## Examples

### Input Images
![Source Face](input/example1.png)
![Target Image](input/img_face01.webp)

### Faceswap Results
![Result 1](output/faceswap_res_1.png)
![Result 2](output/faceswap_res_2.png)
![Result 3](output/faceswap_res_3.png)

### Faceswap Results with lora
![Result 1 with lora](output/faceswap_res_1_lora.png)
![Result 2 with lora](output/faceswap_res_2_lora.png)
![Result 3 with lora](output/faceswap_res_3_lora.png)

### Prerequisites

- Python 3.10
- Git
- Poetry
