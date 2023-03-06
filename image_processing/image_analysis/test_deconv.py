# ---------------------------------------------------------------
# Script name : test_deconv.py
# Created by  : Enda Stockwell
# ---------------------------------------------------------------
# Description:
# Further testing of image restoration using RL deconvolution
# ---------------------------------------------------------------


import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d as conv2
from skimage import color, restoration

image = cv.imread('data\\captures\\test_focal_distance_1000.jpg')
image = color.rgb2gray(image)


psf = cv.imread('data\\captures\\edited\\deconvolved.jpg')

psf = cv.resize(psf, (500, 500), interpolation=cv.INTER_CUBIC)
psf = color.rgb2gray(psf)/7000


# convolved = cv.filter2D(image, -1, psf)



# Restore Image using Richardson-Lucy algorithm
deconvolved = restoration.richardson_lucy(image, psf, num_iter=30)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5))
plt.gray()

for a in (ax[0], ax[1]):
       a.axis('off')

ax[0].imshow(image)
ax[0].set_title('Original Data')

ax[1].imshow(deconvolved, vmin=image.min(), vmax=image.max())
ax[1].set_title('Restoration using\nRichardson-Lucy')


fig.subplots_adjust(wspace=0.02, hspace=0.2, top=0.9, bottom=0.05, left=0, right=1)
plt.show()


cv.imwrite("data\\captures\\edited\\test1.jpg", 255*deconvolved)
