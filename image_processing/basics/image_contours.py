# ---------------------------------------------------------------
# Script name : image_contours.py
# Created by  : Enda Stockwell
# ---------------------------------------------------------------
# Description:
# Display image contours using a thresholded Canny filter
#   and prints total contours detected
# ---------------------------------------------------------------

import cv2 as cv

# Load image
img = cv.imread('data\lichtenstein.png')
cv.imshow('Original Image', img)

# Convert to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Apply Gaussian blur
blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)

# Apply Canny filter
canny = cv.Canny(blur, 125, 175)
cv.imshow('Canny', canny)

# Thresholding  (converts image to binary values)
ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
cv.imshow('Threshold', thresh)

# Finds contours
contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

# Prints number of contours
print("Total contours detected:", len(contours))

cv.waitKey(0)