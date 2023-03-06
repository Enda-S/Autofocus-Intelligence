# ---------------------------------------------------------------
# Script name : plot_MTF.py
# Created by  : Enda Stockwell
# ---------------------------------------------------------------
# Description:
# Plot MTF from full image
# ---------------------------------------------------------------

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from findiff import FinDiff
import scipy.fftpack
from numpy import diff

# Load image
img = cv.imread('data\\captures\\test_focal_distance_1000.jpg', 0)

img = img[1050:1200,1500:1800]

rows, cols = img.shape

plt.imshow(img)

# Loop through rows/cols, storing the intensity of each pixel 
pixel_intensity = []
for i in range(rows):
    for j in range(cols):
        pixel_intensity.append(img[i,j])

# Sort intensity from small to large
pixel_intensity = np.sort(pixel_intensity)

# Normalize pixel intensity from 0 -> 1
normalized_pixel_intensity = (pixel_intensity - np.min(pixel_intensity))/np.ptp(pixel_intensity)

# Standardize the distance from edge, with 0 at the center
x = np.linspace(0, len(pixel_intensity), len(pixel_intensity))
standardized_distance = (x - x.mean())/(x.std())

# Plot the normalized pixel intensity
plt.subplot(2,2,1)
plt.scatter(standardized_distance, normalized_pixel_intensity)
plt.title("Edge Spread Function")
plt.xlabel("Distance... (Standardised)")
plt.ylabel("Pixel Intensity")

# Fit a sigmoid to the data
xdata = standardized_distance
ydata = normalized_pixel_intensity

def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return (y)

p0 = [max(ydata), np.median(xdata), 1, min(ydata)]
popt, pcov = curve_fit(sigmoid, xdata, ydata,p0, method='dogbox')

x = np.linspace(-2, 2, 5000)
y = sigmoid(x, *popt)


# Plot the data with the sigmoid fitted
plt.subplot(2,2,2)
plt.plot(xdata, ydata, 'o', label='data')
plt.title("Fitting curve to data points")
plt.plot(x,y, label='Sigmoid fit to data')
plt.legend(loc='best')


# Find discrete derivative

# method 1.
dx = 0.1
dy = diff(y)/dx
dydx = diff(y)/diff(x)
x2 = np.linspace(-2, 2, 4999)

# method 2.
d_dx = FinDiff(0,1)
df_dx = d_dx(y)


# Plot the derivative (LSF)
plt.subplot(2,2,3)
plt.title("Line-Spread Function")
plt.plot(x, df_dx)


# Find magnitude of the Fourier Transform of the LSF 
N = 2000
T = 1.0 / 800.0
yf = scipy.fftpack.fft(df_dx)
xf = np.linspace(-1, 1.0/(2.0*T), N//2)

# Plot the MTF
plt.subplot(2,2,4)
plt.title("MTF (fft|dy/dx|)")
# plt.plot(xf, 2.0/N * np.abs(yf[:N//2]))
plt.plot(xf, np.abs(yf[:N//2]))
plt.xlim(-1, 5)


plt.show()
cv.waitKey(0)