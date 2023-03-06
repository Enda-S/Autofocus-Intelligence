# ---------------------------------------------------------------
# Script name : variance_of_laplacian.py
# Created by  : Enda Stockwell
# ---------------------------------------------------------------
# Description:
# Find variance of laplacian of sample focal sweep images
#   Captured on iPhone
# ---------------------------------------------------------------

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img1 = cv.imread('data\Focus_Sweep_images\image_1.jpeg')
img2 = cv.imread('data\Focus_Sweep_images\image_2.jpeg')
img3 = cv.imread('data\Focus_Sweep_images\image_3.jpeg')
img4 = cv.imread('data\Focus_Sweep_images\image_4.jpeg')
img5 = cv.imread('data\Focus_Sweep_images\image_5.jpeg')

height = int(img1.shape[1])
width = int(img1.shape[0])

resized1 = cv.resize(img1, (width//6, height//4), interpolation=cv.INTER_CUBIC)
resized2 = cv.resize(img2, (width//6, height//4), interpolation=cv.INTER_CUBIC)

# cv.imshow('Resized Image 1', resized1)
# cv.imshow('Resized Image 2', resized2)


# Blurring photo
blur1 = cv.GaussianBlur(resized1, (3, 3), 0)
blur2 = cv.GaussianBlur(resized2, (3, 3), 0)
# cv.imshow('Blur', blur)

# Converting to grayscale
gray1 = cv.cvtColor(blur1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(blur2, cv.COLOR_BGR2GRAY)


# Canny filter (edge detection) 
# canny1 = cv.Canny(blur1, 125, 175)
# canny2 = cv.Canny(blur2, 125, 175)
#cv.imshow('Canny of grescaled, blurred image', canny)

# Apply Laplace function
dst1 = cv.Laplacian(gray1, cv.CV_16S, 3)
dst2 = cv.Laplacian(gray2, cv.CV_16S, 3)

# converting back to uint8
abs_dst1 = cv.convertScaleAbs(dst1)
abs_dst2 = cv.convertScaleAbs(dst2)

cv.imshow("Image 1 with laplace applied", abs_dst1)
cv.imshow("Image 2 with laplace applied", abs_dst2)



# Compute the Laplacian of the image and then return the focus
# measure, which is simply the variance of the Laplacian

# print("Image 1 = shortest focal distance (0)\nImage 5 = longest focal distance (inf)")
print("Varience of Laplacian (image 1):", cv.Laplacian(img1, cv.CV_64F).var())
print("Varience of Laplacian 2:", cv.Laplacian(img2, cv.CV_64F).var())
print("Varience of Laplacian 3:", cv.Laplacian(img3, cv.CV_64F).var())
print("Varience of Laplacian 4:", cv.Laplacian(img4, cv.CV_64F).var())
print("Varience of Laplacian 5:", cv.Laplacian(img5, cv.CV_64F).var())


cv.waitKey(0)
