# ---------------------------------------------------------------
# Script name : bulk_blur_images.py
# Created by  : Enda Stockwell
# ---------------------------------------------------------------
# Description:
# Apply Gaussian blur to directory of images and save
# ---------------------------------------------------------------


from PIL import Image 
from PIL import ImageFilter 
import os

# Open dataset directory
data_directory = "C:\\Users\\Enda\\Desktop\\2023 College\\Masters\\predictions\\commons_data"


# For every image in directory 
for entry in os.scandir(data_directory): 
    # Open Image
    img = Image.open(entry.path)
    # Apply a Gaussian blur
    blur_img = img.filter(ImageFilter.GaussianBlur(30))

    # Find filename 
    (name, extension) = os.path.splitext(entry.path)

    # Save new blurred file
    blur_img.save(name + 'blur.jpg')


