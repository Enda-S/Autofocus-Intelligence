
# ---------------------------------------------------------------
# Script name : load_multiple_images.py
# Created by  : Enda Stockwell
# ---------------------------------------------------------------
# Description:
# Testing code used to load multiple images from folder.
# ---------------------------------------------------------------

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import glob
import os

# Load multiple images
images = []

def load_images_from_folder(folder):
    for file in os.listdir(folder):
        img = cv.imread(os.path.join(folder, file))
        if img is not None:
            images.append(img)
    return images


load_images_from_folder("data\\focal_sweep_images")

height = int(images[0].shape[1])
width = int(images[0].shape[0])

for img in images: 
    img = cv.resize(img, (width//2, height//6), interpolation=cv.INTER_CUBIC)
    cv.imshow(str([img]), img)


final_img = np.concatenate(images, axis=1)
#cv.imshow("image", resized_images)


cv.waitKey(0)
