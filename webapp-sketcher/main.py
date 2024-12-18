from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import shutil
import os
import io
from PIL import Image
import json
import numpy as np
import random
import torch

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


app = FastAPI()
data_dir = "../data/"
image_dir = data_dir + "/annotation_images"

# Serve static files (HTML, JS, images)
app.mount("/static", StaticFiles(directory="static"), name="static")

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

def setup_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


pipeline = load_model()

def run_inference(instruction, image, sketch, pipeline=None, transparent_background=True):
    if not pipeline:
        pipeline = load_model()
        
    if transparent_background:
        sketch = sketch.convert('L')
        sketch = sketch.point(lambda p: p > 128 and 255)

    setup_seed()
    torch.cuda.empty_cache()
    generator = torch.Generator(device=device)

    inference_ctx = torch.autocast("cuda")
    with inference_ctx:
        edited_image = pipeline(instruction, [[image, sketch]], num_inference_steps=20, generator=generator).images[0]
    
    return edited_image

# Load the mapping file
with open(f'{data_dir}/mapping_file.json', 'r') as file:
    mapping_data = json.load(file)

# Route to serve the sketching page
@app.get("/", response_class=HTMLResponse)
def get_root():
    with open("static/index.html", "r") as file:
        return HTMLResponse(content=file.read())

# New route to get the list of images with masks and editing prompts
@app.get("/images/")
async def get_images():
    prefix = "2_add_object_80"
    images_info = [
        {
            'image_path': attributes['image_path'],
            'mask': attributes['mask'],
            'editing_instruction': attributes['editing_instruction']
        }
        for attributes in mapping_data.values() if attributes['image_path'].startswith(prefix)
    ]
    # print("Filtered Images: ", images_info)
    return {"images": images_info}

# New route to serve images
@app.get("/images/{image_path:path}")
async def serve_image(image_path: str):
    full_image_path = f"{image_dir}/{image_path}"
    print("Constructed image path: ", full_image_path)  # Log the constructed path
    if not os.path.exists(full_image_path):
        print("Image not found: ", full_image_path)  # Log if the image does not exist
    else:
        print("Trying to serve image: ", full_image_path)
    return FileResponse(full_image_path)

# Modify the save_image function to save in the correct directory
@app.post("/save/")
async def save_image(file: UploadFile = File(...), image_path: str = ""):
    filename = image_path.split('.')[0]  # Get the filename without extension
    output_filename = f"{image_dir}/{filename}_sketch.png"  # Save in the same directory
    print("Saving image to: ", output_filename)
    with open(output_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"message": "File saved successfully", "filename": output_filename}

@app.post("/edit/")
async def edit_image(
    image: UploadFile = File(...), 
    sketch_image: UploadFile = File(...),
    instruction: str = Form(...)
):
    image_bytes = await image.read()
    image = Image.open(io.BytesIO(image_bytes))

    sketch_image_bytes = await sketch_image.read()
    sketch_image = Image.open(io.BytesIO(sketch_image_bytes))
    
    # edited_image = run_inference(instruction, image, sketch_image)
    
    edited_image_path = "static/result.jpg"
    # edited_image.save(edited_image_path)

    return FileResponse(edited_image_path)

# # Run this with: uvicorn main:app --reload