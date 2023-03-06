# ---------------------------------------------------------------
# Script name : ROI_box_MTF.py
# Created by  : Enda Stockwell
# ---------------------------------------------------------------
# Description:
# Used to find area for MTF ROI
# ---------------------------------------------------------------

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d as conv2
from skimage import color, restoration

image = cv.imread('data\\captures\\test_focal_distance_1000.jpg')
image2 = cv.imread('data\\captures\\test_focal_distance_10.jpg')


# Start coordinate, here (5, 5)
# represents the top left corner of rectangle
start_point = (1526, 1055)
  
# Ending coordinate, here (220, 220)
# represents the bottom right corner of rectangle
end_point = (1788, 1200)
  
# Blue color in BGR
color = (255, 0, 0)
  
# Line thickness of 2 px
thickness = 2
  
# Draw a rectangle 
image = cv.rectangle(image, start_point, end_point, color, thickness)
  
# Displaying the image 
plt.imshow(image) 
plt.show()