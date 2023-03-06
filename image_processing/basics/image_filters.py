# ---------------------------------------------------------------
# Script name : image_filters.py
# Created by  : Enda Stockwell
# ---------------------------------------------------------------
# Description:
# Apply Laplace and various Sobel filters to image 
# ---------------------------------------------------------------

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Load image
img = cv.imread('data\\ZPos_0.07.bmp')

# Laplace transform
lap = cv.Laplacian(img, cv.CV_64F, ksize=3)
lap = np.uint8(np.absolute(lap))

# Sobel filters
sobelX = cv.Sobel(img, cv.CV_64F, 1, 0)
sobelY = cv.Sobel(img, cv.CV_64F, 0, 1)
sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))

sobelCombined = cv.bitwise_or(sobelX, sobelY)

# Canny filter
edges = cv.Canny(img, 100, 200)

# Plot images nicely
titles = ['Original Image', 'Laplacian', 'SobelX', 'SobelY', 'Sobel Combined', 'Canny']
images = [img, lap, sobelX, sobelY, sobelCombined, edges]

for i in range(6):
    plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])


plt.show()