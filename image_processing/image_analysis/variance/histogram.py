# ---------------------------------------------------------------
# Script name : variance_of_laplacian.py
# Created by  : Enda Stockwell
# ---------------------------------------------------------------
# Description:
# Plot histograms
#   Compares two images, one blurry one sharp. Captured on iPhone
# ---------------------------------------------------------------

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img1 = cv.imread('data\Focus_Sweep_images\image_1.jpeg')

img2 = cv.imread('data\Focus_Sweep_images\image_2.jpeg')

# Histogram from image 1
plt.subplot(1, 2, 1)
plt.imshow(img1)
plt.title("Image 1")


# Histogram from image 2
plt.subplot(1, 2, 2)
plt.imshow(img2)
plt.title("Image 2")


plt.show()

height = int(img1.shape[1])
width = int(img1.shape[0])

resized1 = cv.resize(img1, (width//6, height//4), interpolation=cv.INTER_CUBIC)
resized2 = cv.resize(img2, (width//6, height//4), interpolation=cv.INTER_CUBIC)

# Blurring photo
blur1 = cv.GaussianBlur(resized1, (3, 3), 0)
blur2 = cv.GaussianBlur(resized2, (3, 3), 0)
# cv.imshow('Blur', blur)


# Converting to grayscale
gray1 = cv.cvtColor(blur1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(blur2, cv.COLOR_BGR2GRAY)


# compute a grayscale histogram
hist1 = cv.calcHist([gray1], [0], None, [256], [0, 256])
hist2 = cv.calcHist([gray2], [0], None, [256], [0, 256])

# Plot histograms
plt.figure(0)

# Histogram from image 1
plt.subplot(2, 1, 1)
plt.plot(hist1)
plt.title("Grayscale Histogram 1")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")

# Histogram from image 2
plt.subplot(2, 1, 2)
plt.plot(hist2)
plt.title("Grayscale Histogram 2")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")

plt.xlim([0, 256])
plt.show()