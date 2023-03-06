# ---------------------------------------------------------------
# Script name : convolution_testing.py
# Created by  : Enda Stockwell
# ---------------------------------------------------------------
# Description:
# Further testing of convolution between PSF and images
#   Varying PSF image sizes by resizing the produced PSF image
# ---------------------------------------------------------------

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d as conv2
from skimage import color, restoration
from scipy import signal
image = cv.imread('data\\captures\\test_focal_distance_10.jpg')
psf = cv.imread('data\\captures\\edited\\deconvolved.jpg')

#image = cv.resize(image, (300, 200), interpolation=cv.INTER_CUBIC)
psf1 = cv.resize(psf, (3, 3), interpolation=cv.INTER_CUBIC)
psf2 = cv.resize(psf, (5, 5), interpolation=cv.INTER_CUBIC)
psf3 = cv.resize(psf, (10, 10), interpolation=cv.INTER_CUBIC)
psf4 = cv.resize(psf, (25, 25), interpolation=cv.INTER_CUBIC)
psf5 = cv.resize(psf, (100, 100), interpolation=cv.INTER_CUBIC)
psf6 = cv.resize(psf, (300, 300), interpolation=cv.INTER_CUBIC)
psf7 = cv.resize(psf, (500, 500), interpolation=cv.INTER_CUBIC)
psf8 = cv.resize(psf, (750, 750), interpolation=cv.INTER_CUBIC)
psf9 = cv.resize(psf, (1000, 1000), interpolation=cv.INTER_CUBIC)
psf10 = cv.resize(psf, (1500, 1500), interpolation=cv.INTER_CUBIC)
psf11 = psf

psf1 = color.rgb2gray(psf1)
psf2 = color.rgb2gray(psf2)
psf3 = color.rgb2gray(psf3)/5
psf4 = color.rgb2gray(psf4)/20
psf5 = color.rgb2gray(psf5)/400
psf6 = color.rgb2gray(psf6)/3000
psf7 = color.rgb2gray(psf7)/7000
psf8 = color.rgb2gray(psf8)/13500
psf9 = color.rgb2gray(psf9)/30000
psf10 = color.rgb2gray(psf10)/70000
psf11 = color.rgb2gray(psf11)/90000

image = color.rgb2gray(image)


reconvolved1 = cv.filter2D(image, -1, psf1)
reconvolved2 = cv.filter2D(image, -1, psf2)
reconvolved3 = cv.filter2D(image, -1, psf3)
reconvolved4 = cv.filter2D(image, -1, psf4)
reconvolved5 = cv.filter2D(image, -1, psf5)
reconvolved6 = cv.filter2D(image, -1, psf6)
reconvolved7 = cv.filter2D(image, -1, psf7)
reconvolved8 = cv.filter2D(image, -1, psf8)
reconvolved9 = cv.filter2D(image, -1, psf9)
reconvolved10 = cv.filter2D(image, -1, psf10)
reconvolved11 = cv.filter2D(image, -1, psf11)


cv.imwrite("data\\captures\\edited\\convolution\\reconvolved1.jpg", reconvolved1*255)
cv.imwrite("data\\captures\\edited\\convolution\\reconvolved2.jpg", reconvolved2*255)
cv.imwrite("data\\captures\\edited\\convolution\\reconvolved3.jpg", reconvolved3*255)
cv.imwrite("data\\captures\\edited\\convolution\\reconvolved4.jpg", reconvolved4*255)
cv.imwrite("data\\captures\\edited\\convolution\\reconvolved5.jpg", reconvolved5*255)
cv.imwrite("data\\captures\\edited\\convolution\\reconvolved6.jpg", reconvolved6*255)
cv.imwrite("data\\captures\\edited\\convolution\\reconvolved7.jpg", reconvolved7*255)
cv.imwrite("data\\captures\\edited\\convolution\\reconvolved8.jpg", reconvolved8*255)
cv.imwrite("data\\captures\\edited\\convolution\\reconvolved9.jpg", reconvolved9*255)
cv.imwrite("data\\captures\\edited\\convolution\\reconvolved10.jpg", reconvolved10*255)
cv.imwrite("data\\captures\\edited\\convolution\\reconvolved11.jpg", reconvolved10*255)


cv.waitKey(0)