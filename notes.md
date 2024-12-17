We need different environments for everything

For the p2p requirements, we also need to run:
pip install torch --upgrade

Running command:
time python run_editing_[filename].py         --output_path output         --edit_category_list 0 2         --edit_method_list “[method_name]"

time python run_editing_masactrl.py         --output_path output         --edit_category_list 0 2         --edit_method_list “ddim+masactrl"