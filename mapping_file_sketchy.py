import json

with open("data/mapping_file.json", "r") as f:
    mapping = json.load(f)

new_mapping = {}

for key, item in mapping.items():
    if item["editing_type_id"] == "2":
        new_mapping[key] = item
        new_mapping[key]["sketch_path"] = item["image_path"][:-4] + "_sketch.png"

with open("data/mapping_file_sketchy.json", 'w') as f:
    json.dump(new_mapping, f, indent=4)