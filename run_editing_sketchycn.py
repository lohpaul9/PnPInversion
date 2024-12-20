import os 
import sys
import numpy as np
import argparse
import json
from PIL import Image
import torch
import random
import hashlib
from utils.utils import txt_draw

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

from evaluation.matrics_calculator import MetricsCalculator


def setup_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed

image_save_paths={
    "sketchy-controlnet":"sketchy-controlnet",
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
    controlnet_image = ControlNetModel.from_pretrained("lohpaul/SketchyBusinessControlNet_image", torch_dtype=torch.float32)
    controlnet_sketch = ControlNetModel.from_pretrained("lohpaul/SketchyBusinessControlNet_sketch", torch_dtype=torch.float32)
    
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet_image.requires_grad_(False)
    controlnet_sketch.requires_grad_(False)

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        pretrained_model,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=[controlnet_image, controlnet_sketch],
        guidance_scale=2,
        safety_checker=None,
        torch_dtype=torch.float32,
    )

    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device)

    return pipeline

def run_inference(instruction, image, sketch, pipeline=None, transparent_background=True):
    if not pipeline:
        pipeline = load_model()
        
    if transparent_background:
        new_sketch = Image.new("RGB", sketch.size, (255, 255, 255))
        for x in range(sketch.width):
            for y in range(sketch.height):
                if sketch.getpixel((x, y)) != (0, 0, 0, 0):
                    new_sketch.putpixel((x, y), (0, 0, 0))
                else:
                    new_sketch.putpixel((x, y), (255, 255, 255))
        sketch = new_sketch
        sketch.save("trial-sketch.png")

    seed = setup_seed()
    torch.cuda.empty_cache()
    generator = torch.Generator(device=device).manual_seed(seed)

    inference_ctx = torch.autocast("cuda")
    with inference_ctx:
        edited_image = pipeline(instruction, [[image, sketch]], num_inference_steps=20, generator=generator).images[0]
    
    return edited_image

def run_multi_inference(instruction, image, sketch, pipeline=None, transparent_background=True, num_gens=10):
    if not pipeline:
        pipeline = load_model()
        
    if transparent_background:
        new_sketch = Image.new("RGB", sketch.size, (255, 255, 255))
        for x in range(sketch.width):
            for y in range(sketch.height):
                if sketch.getpixel((x, y)) != (0, 0, 0, 0):
                    new_sketch.putpixel((x, y), (0, 0, 0))
                else:
                    new_sketch.putpixel((x, y), (255, 255, 255))
        sketch = new_sketch
        sketch.save("trial-sketch.png")

    seed = setup_seed()
    torch.cuda.empty_cache()
    final_edited_images = []
    for i in range(num_gens):
        seed_input = f"{seed}_{i}".encode('utf-8')
        hash_value = hashlib.sha256(seed_input).hexdigest()
        new_seed = int(hash_value, 16) % (2**32)

        generator = torch.Generator(device=device).manual_seed(new_seed)

        inference_ctx = torch.autocast("cuda")
        with inference_ctx:
            edited_image = pipeline(instruction, [[image, sketch]], num_inference_steps=20, generator=generator).images[0]
        
        final_edited_images.append(edited_image)
    
    sys.path.insert(0, "/evaluation")
    
    metrics_calculator = MetricsCalculator(device)
    final_edited_image = None
    final_edited_image_score = None
    curr_edited_image_score = metrics_calculator.calculate_lpips(edited_image, image, None, None)
    for edited_image in final_edited_images:
        if final_edited_image_score == None or final_edited_image_score > curr_edited_image_score:
            final_edited_image = edited_image
            final_edited_image_score = curr_edited_image_score
    
    sys.path.remove("/evaluation")
    
    return final_edited_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rerun_exist_images', action= "store_true") # rerun existing images
    parser.add_argument('--data_path', type=str, default="data") # the editing category that needed to run
    parser.add_argument('--output_path', type=str, default="output") # the editing category that needed to run
    parser.add_argument('--edit_category_list', nargs = '+', type=str, default=["0","1","2","3","4","5","6","7","8","9"]) # the editing category that needed to run
    parser.add_argument('--edit_method_list', nargs = '+', type=str, default=["sketchy-controlnet"]) # the editing methods that needed to run
    parser.add_argument('--mapping_file', type=str, default="mapping_file") # the editing methods that needed to run
    args = parser.parse_args()
    
    rerun_exist_images=args.rerun_exist_images
    data_path=args.data_path
    output_path=args.output_path
    edit_category_list=args.edit_category_list
    edit_method_list=args.edit_method_list
    mapping_file=args.mapping_file
    
    pipeline = load_model()
    
    with open(f"{data_path}/{mapping_file}.json", "r") as f:
        editing_instructions = json.load(f)
    
    for key, item in editing_instructions.items():
        
        if item["editing_type_id"] not in edit_category_list:
            continue
        
        original_prompt = item["original_prompt"].replace("[", "").replace("]", "")
        editing_prompt = item["editing_prompt"].replace("[", "").replace("]", "")
        image_path = os.path.join(f"{data_path}/annotation_images", item["image_path"])
        sketch_path = os.path.join(f"{data_path}/annotation_images", item["sketch_path"])
        editing_instruction = item["editing_instruction"]

        if not os.path.exists(sketch_path):
            print(f"no sketch for [{image_path}]")
            continue

        for edit_method in edit_method_list:
            present_image_save_path=image_path.replace(data_path, os.path.join(output_path,image_save_paths[edit_method]))
            if ((not os.path.exists(present_image_save_path)) or rerun_exist_images):
                image_instruct = txt_draw(f"instruct prompt: {editing_instruction}")
                image = Image.open(image_path).convert("RGB")
                sketch = Image.open(sketch_path).convert("RGBA")
                null_sketch = Image.new("RGB", sketch.size, (255, 255, 255))

                print(f"editing image [{image_path}] with [{edit_method}]")
                reconstructed_image = run_inference(original_prompt, image, null_sketch, pipeline=pipeline, transparent_background=False)
                edited_image = run_inference(editing_instruction, image, sketch, pipeline=pipeline, transparent_background=True)
                concatenated_image = Image.fromarray(np.concatenate((image_instruct, image, reconstructed_image, edited_image),axis=1))
                if not os.path.exists(os.path.dirname(present_image_save_path)):
                    os.makedirs(os.path.dirname(present_image_save_path))
                concatenated_image.save(present_image_save_path)
                
                print(f"finish")
                
            else:
                print(f"skip image [{image_path}] with [{edit_method}]")