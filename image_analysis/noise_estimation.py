# ---------------------------------------------------------------
# Script name : noise_estimation.py
# Created by  : Enda Stockwell
# ---------------------------------------------------------------
# Description:
# Estimate noise of given images   
#    Used here for comparison between reconvolved and orignals
# ---------------------------------------------------------------

import cv2 as cv
import numpy as np
from skimage.restoration import estimate_sigma
from skimage import color
def estimate_noise(img):
    return estimate_sigma(img, channel_axis=-1, average_sigmas=True)


image1 = cv.imread('data\\captures\\test_focal_distance_1000.jpg')
image2 = cv.imread('data\\captures\\test_focal_distance_10.jpg')
image3 = cv.imread('data\\captures\\edited\\convolution\\reconvolved7.jpg')
image3 = color.rgb2gray(image3)


noise_score1 = estimate_noise(image1)
noise_score2 = estimate_noise(image2)
noise_score3 = estimate_noise(image3)

print(noise_score1)
print(noise_score2)
print(noise_score3)