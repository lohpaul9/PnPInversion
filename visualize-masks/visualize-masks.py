import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import json

def visualize_mask(mask_array, output_name, width=512, height=512):
    """
    Visualize a mask from the array format in the JSON and save it.
    
    Args:
        mask_array: List of numbers representing the mask
        output_name: Name for the output file
        width: Width of the output image (default 512)
        height: Height of the output image (default 512)
    """
    # Convert the mask array into a binary mask
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # The mask array contains pairs of numbers: position and length
    for i in range(0, len(mask_array), 2):
        if i + 1 < len(mask_array):
            pos = mask_array[i]
            length = mask_array[i + 1]
            
            # Calculate row and column
            row = pos // width
            col = pos % width
            
            # Fill the mask
            while length > 0 and row < height:
                remaining_in_row = width - col
                fill_length = min(remaining_in_row, length)
                
                mask[row, col:col + fill_length] = 255
                
                length -= fill_length
                row += 1
                col = 0
    
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, f'{output_name}.png')
    
    # Display and save the mask
    plt.figure(figsize=(10, 10))
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the figure to free memory
    
    print(f"Saved mask visualization to: {output_path}")

# Example usage for a specific image ID from your JSON
# Replace with the mask array from your JSON
mask_array = [0, 513, 1023, 2]  # This is just an example, use the actual mask array
visualize_mask(mask_array, "mask_224000000009")

# Or to process multiple masks from your JSON:
with open('/home/ec2-user/PnPInversion/data/mapping_file_sketchy.json', 'r') as f:
    data = json.load(f)

for image_id, image_data in data.items():
    visualize_mask(image_data["mask"], f"mask_{image_id}")