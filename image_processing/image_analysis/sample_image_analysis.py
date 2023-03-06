# ---------------------------------------------------------------
# Script name : sample_image_analysis.py
# Created by  : Enda Stockwell
# ---------------------------------------------------------------
# Description:
# Analysis of collimator produced focal sweep
#
#   Applies Canny filters
#   Calculates variance and shows differences between original and canny
# ---------------------------------------------------------------

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Load images
img1 = cv.imread('data\\focal_sweep_images\\ZPos_0.07.bmp')
img2 = cv.imread('data\\focal_sweep_images\\ZPos_0.142.bmp')
img3 = cv.imread('data\\focal_sweep_images\\ZPos_0.222.bmp')

# Find image height & width 
width = int(img2.shape[0])
height = int(img2.shape[1])
print ("Width: ", width, "  Height: ", height, "(Pixels)")

# Resize images (lower resolution for ease of use)
img1 = cv.resize(img1, (480, 360), interpolation=cv.INTER_CUBIC)
img2 = cv.resize(img2, (480, 360), interpolation=cv.INTER_CUBIC)
img3 = cv.resize(img3, (480, 360), interpolation=cv.INTER_CUBIC)

# Cocatinate images and lable each
concat_images = np.concatenate((img1, img2, img3), axis=1)
cv.putText(concat_images, '3 images of different focal distance', (480, 25), cv.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255), 1)
cv.putText(concat_images, 'Image 1, 0.07um', (0, 200), cv.FONT_HERSHEY_TRIPLEX, 0.75, (255, 255, 255), 1)
cv.putText(concat_images, 'Image 2, 0.142um', (480, 200), cv.FONT_HERSHEY_TRIPLEX, 0.75, (255, 255, 255), 1)
cv.putText(concat_images, 'Image 3, 0.222um', (960, 200), cv.FONT_HERSHEY_TRIPLEX, 0.75, (255, 255, 255), 1)

cv.imshow('Images Concatinated', concat_images)

# Apply canny filter to images
canny1 = cv.Canny(img1, 125, 175)
canny2 = cv.Canny(img2, 125, 175)
canny3 = cv.Canny(img3, 125, 175)

# cv.imshow('Canny1', canny1)
# cv.imshow('Canny2', canny2)
# cv.imshow('Canny3', canny3)

# Cocatinate filtered images and lable each
concat_canny_images = np.concatenate((canny1, canny2, canny3), axis=1)
cv.putText(concat_canny_images, 'Same 3 with Canny filter applied', (480, 25), cv.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255), 1)
cv.putText(concat_canny_images, 'Image 1, 0.07um', (0, 200), cv.FONT_HERSHEY_TRIPLEX, 0.75, (255, 255, 255), 1)
cv.putText(concat_canny_images, 'Image 2, 0.142um', (480, 200), cv.FONT_HERSHEY_TRIPLEX, 0.75, (255, 255, 255), 1)
cv.putText(concat_canny_images, 'Image 3, 0.222um', (960, 200), cv.FONT_HERSHEY_TRIPLEX, 0.75, (255, 255, 255), 1)

cv.imshow('Edge Cascased Concatinated', concat_canny_images)


gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
gray3 = cv.cvtColor(img3, cv.COLOR_BGR2GRAY)

var1 = np.var(img1)
var2 = np.var(img2)
var3 = np.var(img3)

var_grey1 = np.var(gray1)
var_grey2 = np.var(gray2)
var_grey3 = np.var(gray3)

var_coef_1 = (np.sqrt(var1)/cv.meanStdDev(img1)[0])
var_coef_2 = (np.sqrt(var2)/cv.meanStdDev(img2)[0])
var_coef_3 = (np.sqrt(var3)/cv.meanStdDev(img3)[0])

print(cv.meanStdDev(img1))

print("Variance of images in color\n ", "Image 1: ", var1, "Image 2: ", var2, "Image 3: ", var3)
print("\nVariance of images in greyscale\n ", "Image 1: ", var_grey1, "Image 2: ", var_grey2, "Image 3: ", var_grey3)

# print("\n Coefficient of variance in color\n ", "Image 1: ", var_coef_1, "Image 2: ", var_coef_1, "Image 3: ", var_coef_1)


cv.waitKey(0)
