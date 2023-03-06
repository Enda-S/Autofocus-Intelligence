import cv2 as cv
import numpy as np

# Load blank image (zeros)
blank = np.zeros((500, 500, 3), dtype = 'uint8')
cv.imshow('Blank', blank)


# Make image a certain color
blank[50:75, 50:75] = 0,255,255
cv.imshow('Yellow', blank)


# Draw rectangle
# cv.rectangle(blank, (0,0), (250, 500), (0, 255, 0), thickness = cv.FILLED)
# cv.imshow('Rectangle', blank)


# Draw circle
# cv.circle(blank, (blank.shape[1]//2, blank.shape[0]//2), 40, (0, 0, 255), thickness =-1)
# cv.imshow('Circle', blank)


# Draw line
# cv.line(blank, (0,0), (250, 250), (0, 255, 0), thickness = 3)
# cv.imshow('Line', blank)


# Write text
cv.putText(blank, 'Test text', (0, 225), cv.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255), 2)
cv.imshow('Text', blank)


cv.waitKey(0)
