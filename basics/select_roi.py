# ---------------------------------------------------------------
# Script name : select_roi.py
# Created by  : Enda Stockwell
# ---------------------------------------------------------------
# Description:
# Select and crop to a Region Of Interest (ROI) from image
# ---------------------------------------------------------------

import cv2 as cv

img = cv.imread('data\\lichtenstein.png', 1)


# Select ROI function
roi = cv.selectROI(img)

# Print rectangle points of selected roi
print(roi)

# Crop selected roi from raw image
roi_cropped = img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]

# Show cropped image
cv.imshow("ROI", roi_cropped)


cv.waitKey(0)