# ---------------------------------------------------------------
# Script name : segmented_variance_of_laplacian.py
# Created by  : Enda Stockwell
# ---------------------------------------------------------------
# Description:
# Find variance of laplacian of sample image
#   Where the image is broken into different segments
#
# Useful in explaining how the variance of laplacian works on different textures
# ---------------------------------------------------------------

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Load image
img = cv.imread('data\lichtenstein.png')
cv.imshow('Lichtenstein_img_processing_test', img)

# Laplace transform of image
laplace = cv.Laplacian(img, cv.CV_64F)
cv.imshow('Laplacian', laplace)

# Blur image to different extents
blur1 = cv.GaussianBlur(img, (3, 3), cv.BORDER_DEFAULT)
blur2 = cv.GaussianBlur(img, (7, 7), cv.BORDER_DEFAULT)
blur3 = cv.GaussianBlur(img, (25, 25), cv.BORDER_DEFAULT)

# Segments the image into 9 equally sized squares.
A = img[0:img.shape[0]//3, 0:img.shape[1]//3]
B = img[0:img.shape[0]//3, img.shape[1]//3:(2*img.shape[1])//3]
C = img[0:img.shape[0]//3, (2*img.shape[1])//3:(3*img.shape[1])//3]
D = img[img.shape[0]//3:(2*img.shape[0])//3, 0:img.shape[1]//3]
E = img[img.shape[0]//3:(2*img.shape[0])//3, img.shape[1]//3:(2*img.shape[1])//3]
F = img[img.shape[0]//3:(2*img.shape[0])//3, (2*img.shape[1])//3:(3*img.shape[1])//3]
G = img[(2*img.shape[0])//3:(3*img.shape[0])//3, 0:img.shape[1]//3]
H = img[(2*img.shape[0])//3:img.shape[0], img.shape[1]//3:(2*img.shape[1])//3]
I = img[(2*img.shape[0])//3:img.shape[0], (2*img.shape[1])//3:(3*img.shape[1])//3]

# Label each segment and display it
cv.putText(A, 'A', (125, 25), cv.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255), 1)
cv.imshow('A', A)
cv.putText(B, 'B', (125, 25), cv.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255), 1)
cv.imshow('B', B)
cv.putText(C, 'C', (125, 25), cv.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255), 1)
cv.imshow('C', C)
cv.putText(D, 'D', (125, 25), cv.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255), 1)
cv.imshow('D', D)
cv.putText(E, 'E', (125, 25), cv.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255), 1)
cv.imshow('E', E)
cv.putText(F, 'F', (125, 25), cv.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255), 1)
cv.imshow('F', F)
cv.putText(G, 'G', (125, 25), cv.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255), 1)
cv.imshow('G', G)
cv.putText(H, 'H', (125, 25), cv.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255), 1)
cv.imshow('H', H)
cv.putText(I, 'I', (125, 25), cv.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255), 1)
cv.imshow('I', I)

# Calculates the variance of the laplace transformed image
print("Varience of Laplacian A:", cv.Laplacian(A, cv.CV_64F).var())
print("Varience of Laplacian B:", cv.Laplacian(B, cv.CV_64F).var())
print("Varience of Laplacian C:", cv.Laplacian(C, cv.CV_64F).var())
print("Varience of Laplacian D:", cv.Laplacian(D, cv.CV_64F).var())
print("Varience of Laplacian E:", cv.Laplacian(E, cv.CV_64F).var())
print("Varience of Laplacian F:", cv.Laplacian(F, cv.CV_64F).var())
print("Varience of Laplacian G:", cv.Laplacian(G, cv.CV_64F).var())
print("Varience of Laplacian H:", cv.Laplacian(H, cv.CV_64F).var())
print("Varience of Laplacian I:", cv.Laplacian(I, cv.CV_64F).var())

# concat_images = np.concatenate((A, B, C, D, E, F, G, H, I), axis=1)

# total variance
tot = (cv.Laplacian(A, cv.CV_64F).var()+cv.Laplacian(B, cv.CV_64F).var()+cv.Laplacian(C, cv.CV_64F).var()+cv.Laplacian(D, cv.CV_64F).var()+cv.Laplacian(E, cv.CV_64F).var()+cv.Laplacian(F, cv.CV_64F).var()+cv.Laplacian(G, cv.CV_64F).var()+cv.Laplacian(H, cv.CV_64F).var()+cv.Laplacian(I, cv.CV_64F).var())
weighted_tot = (0.5*(cv.Laplacian(A, cv.CV_64F).var()+cv.Laplacian(B, cv.CV_64F).var()+cv.Laplacian(C, cv.CV_64F).var()+cv.Laplacian(D, cv.CV_64F).var()+2*(cv.Laplacian(E, cv.CV_64F).var())+cv.Laplacian(F, cv.CV_64F).var()+cv.Laplacian(G, cv.CV_64F).var()+cv.Laplacian(H, cv.CV_64F).var()+cv.Laplacian(I, cv.CV_64F).var()))

avg = tot//9
weighted_avg = weighted_tot//9

print("total: ", tot, "\nweighted total: ", weighted_tot, "\naverage: ", avg, "\nweighted average: ", weighted_avg)

cv.waitKey(0)