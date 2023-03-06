# ---------------------------------------------------------------
# Script name : image_histograms.py
# Taken from  : https://docs.opencv.org/
# ---------------------------------------------------------------
# Description:
# Plot image color histograms
# ---------------------------------------------------------------

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Load image
img = cv.imread('data\lichtenstein.png')
cv.imshow('Castle', img)

# Show image histogram
plt.figure(0)
plt.hist(img.ravel(), 256, [0,256])
plt.title("Total Image histogram")

color = ('b','g','r')

# Show image RGB histograms
plt.figure(1)

for i,col in enumerate(color):
    histr = cv.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])

# Convert to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Grayscale image', gray)

# Plot greyscale histogram
plt.figure(2)
plt.hist(gray.ravel(), 256, [0,256])
plt.title("Grayscale image histogram")

# Blur image to 2 extents
blur1 = cv.GaussianBlur(gray, (3, 3), cv.BORDER_DEFAULT)
cv.imshow('Blurred image (1)', blur1)

blur2 = cv.GaussianBlur(gray, (25, 25), cv.BORDER_DEFAULT)
cv.imshow('Blurred image (2)', blur2)

# Plot histogram of blurred images
plt.figure(3)
plt.hist(blur1.ravel(), 256, [0,256])
plt.title("Blurred image histogram (1)")
plt.figure(4)
plt.hist(blur2.ravel(), 256, [0,256])
plt.title("Blurred image histogram (2)")

# Plot overlapping histogram of the original grayscale image and the two blurred
plt.style.use('seaborn-deep')
plt.figure(5)
plt.hist([gray.ravel(), blur1.ravel(), blur2.ravel()], 256, [0,256], label=['image', 'Blur 1', 'Blur 2'])
plt.legend(loc='upper right')

plt.figure(6)
# Create a mask
mask = np.zeros(gray.shape[:2], np.uint8)
mask[100:300, 100:400] = 255
masked_img = cv.bitwise_and(gray,gray,mask = mask)

# Calculate histogram with mask and without mask
# Check third argument for mask
hist_full = cv.calcHist([gray],[0], None,[256],[0,256])
hist_mask = cv.calcHist([gray],[0], mask,[256],[0,256])
plt.subplot(221), plt.imshow(gray, 'gray')
plt.subplot(222), plt.imshow(mask,'gray')
plt.subplot(223), plt.imshow(masked_img, 'gray')
plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
plt.xlim([0,256])


plt.show()

cv.waitKey(0)