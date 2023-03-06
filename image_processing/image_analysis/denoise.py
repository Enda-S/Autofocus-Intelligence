# ---------------------------------------------------------------
# Script name : deblur.py
# Created by  : Enda Stockwell
# ---------------------------------------------------------------
# Description:
# Denoise input image
# ---------------------------------------------------------------

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d as conv2
from skimage import color, restoration

image = cv.imread('data\\captures\\test_focal_distance_1000.jpg')
sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharpen = cv.filter2D(image, -1, sharpen_kernel)


# resized = cv.resize(sharpen, (1000, 750), interpolation=cv.INTER_CUBIC)
# cv.imshow('Resized', resized)

# cv.waitKey()


# Denoising
dst = cv.fastNlMeansDenoisingColored(sharpen,None,10,10,7,21)



cv.imwrite("data\\captures\\test_focal_distance_1000_denoise.jpg", dst)
