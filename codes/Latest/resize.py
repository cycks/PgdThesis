import os
from PIL import Image

def resize_images_in_folder(source_folder, destination_folder, new_size=(256, 256)):
    """
    Resizes all images in a folder and its subfolders and saves them to a new folder.

    Args:
        source_folder (str): The path to the folder containing the original images.
        destination_folder (str): The path to the folder where resized images will be saved.
        new_size (tuple): The desired dimensions (width, height) for the resized images.
    """
    if not os.path.exists(source_folder):
        print(f"Error: The source folder '{source_folder}' does not exist.")
        return

    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)
    
    # Supported image extensions
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
    
    print(f"Starting image resizing process from '{source_folder}' to '{destination_folder}'...")

    # Walk through the source folder, including all subfolders
    for root, dirs, files in os.walk(source_folder):
        # Create the corresponding subfolder structure in the destination folder
        relative_path = os.path.relpath(root, source_folder)
        destination_path = os.path.join(destination_folder, relative_path)
        os.makedirs(destination_path, exist_ok=True)
        
        for file in files:
            # Check if the file is an image
            if file.lower().endswith(image_extensions):
                source_image_path = os.path.join(root, file)
                destination_image_path = os.path.join(destination_path, file)
                
                try:
                    # Open the image using Pillow
                    with Image.open(source_image_path) as img:
                        # Resize the image to the new dimensions
                        resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
                        
                        # Save the resized image to the new location
                        resized_img.save(destination_image_path)
                        print(f"Resized and saved: {source_image_path} -> {destination_image_path}")
                except Exception as e:
                    print(f"Could not resize '{source_image_path}': {e}")
    
    print("\nImage resizing process completed.")

# Define your source and destination folders based on the prompt
source_folder_path = './komnet'
destination_folder_path = './komnet2'

# Run the function to resize images
resize_images_in_folder(source_folder_path, destination_folder_path)

# aws s3 cp cycks_with_padding.py s3://bucket-fulani
# cp -r ./newpgd/komnet ./modifiedPGD
# aws s3 cp ~/home/sikolia/Desktop/PgdThesis/codes/newpgd/CorrectedCode/cycks_with_padding.py s3://bucket-fulani
# aws s3 cp s3://bucket-fulani/cycks_with_padding.py   /modifiedPGD