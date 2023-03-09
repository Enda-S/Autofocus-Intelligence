# ---------------------------------------------------------------
# Script name : autofocus_contrast.py
# Created by  : Enda Stockwell
# Adapted from : www.arducam.com/
# ---------------------------------------------------------------
# Description:
# Sweep focal range whilst measuring contrast.
#   If contrast starts rolling off => Jump back to maximum and capture.
#
# Mimics standard contrast detection autofocus systems.
# ---------------------------------------------------------------

import cv2
import numpy as np
import os
import time
import smbus
import matplotlib.pyplot as plt
from subprocess import run

# Initialize Picamera and smbus for Arducam's I2C focus control
bus = smbus.SMBus(0)
try:
    import picamera
    from picamera.array import PiRGBArray
except:
    sys.exit(0)
    
def focusing(val):
    # Initialize focal distance
    value = (val << 4) & 0x3ff0
    data1 = (value >> 8) & 0x3f
    data2 = value & 0xf0
    print("focus value: {}".format(val))
    time.sleep(0.01)
    cmd = run(["i2cset", "-y", "0", "0x0c", str(data1), str(data2)])
            

    return(cmd.returncode)

def sobel(img):
    # Apply Sobel
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img_sobel = cv2.Sobel(img_gray,cv2.CV_16U,1,1)
    return cv2.mean(img_sobel)[0]

def laplacian(img):
    # Apply Laplacian
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img_sobel = cv2.Laplacian(img_gray,cv2.CV_16U)
    return cv2.mean(img_sobel)[0]
    

# Calculate contrast	
def calculation(camera):
    # Get raw camera data
    rawCapture = PiRGBArray(camera) 
    # Capture frame
    camera.capture(rawCapture,format="bgr", use_video_port=True)
    # Convert captured image into an array
    image = rawCapture.array
    # Clear stream for the next frame
    rawCapture.truncate(0)
    
    return laplacian(image)
    
    
    
if __name__ == "__main__":
    # Open camera
    camera = picamera.PiCamera()
    camera.start_preview(fullscreen=False, window = (1240, 0, 640, 480))
    camera.resolution = (640, 480)
    # Wait
    time.sleep(0.1)
    
    print("Start focusing")
    
    # Variables needed for logging
    max_index = 10
    max_value = 0.0
    last_value = 0.0
    dec_count = 0
    focal_distance = 10
    val_list = []
    focal_list = []
        

    while True:
        # Adjust focus
        return_val = focusing(focal_distance)
        
		# Check for errors writing to i2c, try again
        if return_val !=0:
            print("ERROR Writing to I2c.\nTrying again..\n")
            focusing(focal_distance)

        # Take image and calculate image clarity
        val = calculation(camera)
        
        # Append recorded contrast to list
        val_list.append(val)
        focal_list.append(focal_distance)
        
        print("focus score " + str(val))
        print("\n")

        #Find the maximum image clarity
        if val > max_value:
            max_index = focal_distance
            max_value = val
            
        # If the image clarity starts to decrease
        if val < last_value:
            dec_count += 1
        else:
            dec_count = 0

        # If the image clarity is reduced by six consecutive frames
        if dec_count > 6:
            # Stop sweeping
            break
        last_value = val
        
        # Increase the focal distance
        focal_distance += 15
        if focal_distance > 1000:
            break

	# Return focus to the maximum contrast result recorded
    return_val = focusing(max_index)
    
    # Check for errors writing to i2c, try again
    if return_val !=0:
        print("ERROR Writing to I2c.\nTrying again..\n")
        focusing(max_index)
    
    time.sleep(.5)

    camera.resolution = (3280,2464)
    # Save image to file.
    camera.capture("highest_contrast_capture.jpg")
    print("max index = %d,max value = %lf" % (max_index, max_value))

    
    camera.stop_preview()
    camera.close()

    
    x = np.arange(0, len(val_list), 1)
    
    plt.plot(focal_list, val_list)
    plt.title("Contrast measurement vs. focal distance")
    plt.xlabel("Focal Distance")
    plt.ylabel("Image Quality")
    plt.show()
