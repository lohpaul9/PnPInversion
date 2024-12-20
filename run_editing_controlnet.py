import os 
import numpy as np
import argparse
import json
from PIL import Image
import torch
import random

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)

from transformers import (
    AutoTokenizer,
    CLIPTextModel,
)


def setup_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

image_save_paths={
    "controlnet":"controlnet",
    }

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    pretrained_model = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model, subfolder="tokenizer", use_fast=False,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model, subfolder="text_encoder",
    )
    vae = AutoencoderKL.from_pretrained(
        pretrained_model, subfolder="vae", 
    )
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model, subfolder="unet", 
    )
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float32)
    
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.requires_grad_(False)

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        pretrained_model,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=[controlnet],
        guidance_scale=7.5,
        safety_checker=None,
        torch_dtype=torch.float32,
    )

    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device)

    return pipeline

def run_inference(instruction, image, sketch, pipeline=None):
    if not pipeline:
        pipeline = load_model()

    # Convert sketch to black and white format
    new_sketch = Image.new("RGB", sketch.size, (255, 255, 255))
    for x in range(sketch.width):
        for y in range(sketch.height):
            if sketch.getpixel((x, y)) != (0, 0, 0, 0):
                new_sketch.putpixel((x, y), (0, 0, 0))
            else:
                new_sketch.putpixel((x, y), (255, 255, 255))
    sketch = new_sketch

    setup_seed()
    torch.cuda.empty_cache()
    generator = torch.Generator(device=device)

    inference_ctx = torch.autocast("cuda")
    with inference_ctx:
        edited_image = pipeline(
            prompt=instruction,
            image=[sketch],  # Pass sketch as the control image
            # control_image=sketch,  # Explicitly pass as control_image
            num_inference_steps=30,
            generator=generator,
            guidance_scale=7.5,
            controlnet_conditioning_scale=1.0,
            guess_mode=False,
        ).images[0]
    
    return edited_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rerun_exist_images', action="store_true")
    parser.add_argument('--data_path', type=str, default="data")
    parser.add_argument('--output_path', type=str, default="output")
    parser.add_argument('--edit_category_list', nargs='+', type=str, default=["0","1","2","3","4","5","6","7","8","9"])
    parser.add_argument('--edit_method_list', nargs='+', type=str, default=["controlnet"])
    parser.add_argument('--mapping_file', type=str, default="mapping_file")
    parser.add_argument('--image_path', type=str, help="Path to the input image", default="/home/ec2-user/PnPInversion/data/annotation_images/2_add_object_80/2_natural/1_animal/221000000000.jpg")
    parser.add_argument('--sketch_path', type=str, help="Path to the sketch image", default="//home/ec2-user/PnPInversion/data/annotation_images/2_add_object_80/2_natural/1_animal/221000000000_sketch.png")
    parser.add_argument('--instruction', type=str, help="Editing instruction", default="add a colar")
    args = parser.parse_args()
    
    rerun_exist_images=args.rerun_exist_images
    data_path=args.data_path
    output_path=args.output_path
    edit_category_list=args.edit_category_list
    edit_method_list=args.edit_method_list
    mapping_file=args.mapping_file
    
    pipeline = load_model()
    
    if args.image_path and args.sketch_path and args.instruction:
        image = Image.open(args.image_path).convert("RGB")
        sketch = Image.open(args.sketch_path).convert("L")
        
        print(f"Editing single image with provided paths")
        edited_image = run_inference(args.instruction, image, sketch, pipeline)
        
        output_filename = os.path.basename(args.image_path)
        output_path = os.path.join(args.output_path, image_save_paths["controlnet"], output_filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        edited_image.save(output_path)
        print("Finished")
    else:
        with open(f"{data_path}/{mapping_file}.json", "r") as f:
            editing_instructions = json.load(f)
        
        for key, item in editing_instructions.items():
            
            if item["editing_type_id"] not in edit_category_list:
                continue
            
            image_path = os.path.join(f"{data_path}/annotation_images", item["image_path"])
            sketch_path = os.path.join(f"{data_path}/annotation_images", item["sketch_path"])
            editing_instruction = item["editing_instruction"]

            if not os.path.exists(sketch_path):
                print(f"no sketch for [{image_path}]")
                continue

            for edit_method in edit_method_list:
                present_image_save_path=image_path.replace(data_path, os.path.join(output_path,image_save_paths[edit_method]))
                if ((not os.path.exists(present_image_save_path)) or rerun_exist_images):
                    image = Image.open(image_path).convert("RGB")
                    sketch = Image.open(sketch_path).convert("L")

                    print(f"editing image [{image_path}] with [{edit_method}]")
                    edited_image = run_inference(editing_instruction, image, sketch, pipeline)
                    if not os.path.exists(os.path.dirname(present_image_save_path)):
                        os.makedirs(os.path.dirname(present_image_save_path))
                    edited_image.save(present_image_save_path)
                    
                    print(f"finish")
                    
                else:
                    print(f"skip image [{image_path}] with [{edit_method}]")