# ---------------------------------------------------------------
# Script name : convolution.py
# Created by  : Enda Stockwell
# ---------------------------------------------------------------
# Description:
# Convolution of 2 images 
#   Used here to convolute in-focus image with PSF to blur
# ---------------------------------------------------------------

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d as conv2
from skimage import color
from scipy import signal

image = cv.imread('data\\captures\\test_focal_distance_10.jpg')
psf = cv.imread('data\\captures\\edited\\deconvolved.jpg')

#image = cv.resize(image, (300, 200), interpolation=cv.INTER_CUBIC)
psf = cv.resize(psf, (30, 30), interpolation=cv.INTER_CUBIC)

image = color.rgb2gray(image)
psf = color.rgb2gray(psf)/30


#cv.imshow('psf', psf)
#cv.imshow('image1', image)

plt.imshow(psf)
plt.show()

kernel = np.ones((5,5),np.float32)/25

# from skimage import util 
# psf = util.invert(psf)/25
#print(psf)

reconvolved = cv.filter2D(image, -1, psf)
#reconvolved = signal.convolve2d(image, psf, boundary='symm', mode='same')


#cv.imshow('reconvolved', reconvolved)
cv.imwrite("data\\captures\\edited\\reconvolved.jpg", reconvolved*255)


cv.waitKey(0)