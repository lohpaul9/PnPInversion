We need different environments for everything

For the p2p requirements, we also need to run:
pip install torch --upgrade

Running command:
time python run_editing_[filename].py         --output_path output         --edit_category_list 0 2         --edit_method_list "[method_name]"

time python run_editing_masactrl.py         --output_path output         --edit_category_list 0 2         --edit_method_list "ddim+masactrl"
time python run_editing_sketchycn.py         --output_path output         --edit_category_list 2         --edit_method_list "sketchy-controlnet"        --mapping_file mapping_file_sketchy

python evaluation/evaluate.py --metrics "structure_distance" "psnr_unedit_part" "lpips_unedit_part" "mse_unedit_part" "ssim_unedit_part" "clip_similarity_source_image" "clip_similarity_target_image" "clip_similarity_target_image_edit_part" --result_path evaluation_result.csv --edit_category_list 2 --tgt_methods 1_ddim+p2p 1_ddim+masactrl 1_directinversion+pnp 9_sketchy-controlnet

python evaluation/evaluate.py --metrics "lpips" "lpips_unedit_part" "ssim" "clip_similarity_target_image" "clip_similarity_target_image_edit_part" --result_path evaluation_result.csv --edit_category_list 2 --tgt_methods 1_ddim+p2p 1_ddim+masactrl 1_directinversion+pnp 9_sketchy-controlnet
