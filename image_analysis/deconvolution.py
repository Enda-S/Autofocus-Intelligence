# ---------------------------------------------------------------
# Script name : deconvolution.py
# Created by  : Enda Stockwell
# ---------------------------------------------------------------
# Description:
# Deconvolution of 2 images using Richardson-Lucy
#   Used here to produce PSF 
# ---------------------------------------------------------------

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d as conv2
from skimage import color, restoration

image = cv.imread('data\\captures\\test_focal_distance_1000.jpg')
image2 = cv.imread('data\\captures\\test_focal_distance_10.jpg')


im1 = color.rgb2gray(image)

im2 = color.rgb2gray(image2)


height, width = im1.shape 
im1 = cv.resize(im1, (width//3, height//3), interpolation=cv.INTER_CUBIC)
im2 = cv.resize(im2, (width//3, height//3), interpolation=cv.INTER_CUBIC)


# Restore Image using Richardson-Lucy algorithm
# (Find PSF)
deconvolved = restoration.richardson_lucy(im1, im2, num_iter=30)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5))
plt.gray()

for a in (ax[0], ax[1]):
       a.axis('off')

ax[0].imshow(im1)
ax[0].set_title('Original Data')

ax[1].imshow(deconvolved, vmin=im1.min(), vmax=im1.max())
ax[1].set_title('Deconvolution using\nRichardson-Lucy')


fig.subplots_adjust(wspace=0.02, hspace=0.2,
                    top=0.9, bottom=0.05, left=0, right=1)
plt.show()


cv.imwrite("data\\captures\\edited\\deconvolved.jpg", 255*deconvolved)
