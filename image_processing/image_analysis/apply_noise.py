# ---------------------------------------------------------------
# Script name : apply_noise.py
# Created by  : Enda Stockwell
# ---------------------------------------------------------------
# Description:
# Apply noise to image  
#    Used here to apply noise to reconvolved image to replicate noise
# ---------------------------------------------------------------

import cv2 as cv
import numpy as np
from skimage.restoration import estimate_sigma
from skimage import color

def estimate_noise(img):
    return estimate_sigma(img, multichannel=True, average_sigmas=True)

image = cv.imread('data\\captures\\edited\\convolution\\reconvolved7.jpg')
image = color.rgb2gray(image)

mean = 0
var = 0.0005
sigma = var ** 0.5
gaussian = np.random.normal(mean, sigma, (image.shape[0],image.shape[1])) 

noisy_image = image + gaussian

cv.imwrite("data\\captures\\edited\\convolution\\noisy_reconvolved7.jpg", noisy_image*255)
