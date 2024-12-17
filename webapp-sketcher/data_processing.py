import json 

# Load the mapping file
with open('../data/mapping_file.json', 'r') as file:
    mapping_data = json.load(file)


def print_editing_instructions_by_category(mapping_data):
    # Dictionary to hold editing instructions by category
    editing_instructions_by_category = {}

    # Iterate through the mapping data
    for image_id, attributes in mapping_data.items():
        # Extract the category from the image path
        category = attributes['image_path'].split('/')[0]  
        editing_instruction = attributes['editing_instruction']
        
        # Initialize the category in the dictionary if not already present
        if category not in editing_instructions_by_category:
            editing_instructions_by_category[category] = []
        
        # Append the editing instruction and relevant attributes to the corresponding category
        editing_instructions_by_category[category].append({
            'image_id': image_id,
            'image_path': attributes['image_path'],
            'original_prompt': attributes['original_prompt'],
            'editing_prompt': attributes['editing_prompt'],
            'editing_instruction': editing_instruction,
            'editing_type_id': attributes['editing_type_id'],
            'blended_word': attributes['blended_word'],
            'mask_length': len(attributes['mask'])  # Store the length of the mask
        })

    # Print two full instances for each category
    for category, instances in editing_instructions_by_category.items():
        print(f"Category: {category}")
        if category != "2_add_object_80":
            continue
        for instance in instances:  # Print only the first two instances
            print(f"  Image ID: {instance['image_id']}")
            print(f"  Image Path: {instance['image_path']}")
            print(f"  Original Prompt: {instance['original_prompt']}")
            print(f"  Editing Prompt: {instance['editing_prompt']}")
            print(f"  Editing Instruction: {instance['editing_instruction']}")
            print(f"  Editing Type ID: {instance['editing_type_id']}")
            print(f"  Blended Word: {instance['blended_word']}")
            print(f"  Mask Length: {instance['mask_length']}")



def understand_mask(mask):
    # mask is a 1D array of ints
    pass

print_editing_instructions_by_category(mapping_data)

