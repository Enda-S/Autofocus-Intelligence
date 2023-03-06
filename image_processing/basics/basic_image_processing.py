# ---------------------------------------------------------------
# Script name : basic_image_processing.py
# Created by  : Enda Stockwell
# ---------------------------------------------------------------
# Description:
# Basic image processing using openCV
# ---------------------------------------------------------------

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Load image
img = cv.imread('data\\lichtenstein.png')

# Resize image
img = cv.resize(img, (480, 360), interpolation=cv.INTER_CUBIC)
cv.imshow('Image', img)

# Converting to greyscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', gray)

# Apply a laplace
laplace = cv.Laplacian(img, cv.CV_64F)
cv.imshow('Laplacian', laplace)

# Blur an image
blur = cv.GaussianBlur(img, (3, 3), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)

# Edge Cascade
canny = cv.Canny(blur, 125, 175)
cv.imshow('Canny', canny)

# Resize
resized = cv.resize(img, (500,500), interpolation=cv.INTER_CUBIC)
cv.imshow('Resized', resized)

# Cropping
cropped = img[50:300, 50:300]
cv.imshow('Cropped', cropped)


cv.waitKey(0)