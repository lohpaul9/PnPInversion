from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import shutil
import os
import json

app = FastAPI()
data_dir = "../data/"
image_dir = data_dir + "/annotation_images"

# Serve static files (HTML, JS, images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the mapping file
with open(f'{data_dir}/mapping_file.json', 'r') as file:
    mapping_data = json.load(file)

# Route to serve the sketching page
@app.get("/", response_class=HTMLResponse)
def get_root():
    with open("static/index.html", "r") as file:
        return HTMLResponse(content=file.read())

# New route to get the list of images
@app.get("/images/")
async def get_images():
    prefix = "2_add_object_80"
    images = [attributes['image_path'] for attributes in mapping_data.values() if attributes['image_path'].startswith(prefix)]
    print("Filtered Images: ", images)
    return {"images": images}

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

# Run this with: uvicorn main:app --reload