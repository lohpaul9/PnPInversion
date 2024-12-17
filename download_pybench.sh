# # Define the Google Drive file link
file_id="1qNNAbkd90qGTgwKdMppMpMU8Cegbj8bF"
file_id_path = "https://drive.google.com/file/d/1qNNAbkd90qGTgwKdMppMpMU8Cegbj8bF/view?usp=sharing"

# pip install gdown

destination="PIEbench.zip"
gdown --id $file_id -O $destination


# Unzip the downloaded file
unzip ${destination}