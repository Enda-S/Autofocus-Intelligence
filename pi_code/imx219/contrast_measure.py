# ---------------------------------------------------------------
# Script name : contrast_measure.py
# Created by  : Enda Stockwell
# Adapted from : www.arducam.com/
# ---------------------------------------------------------------
# Description:
# Sweeps entire focal range whilst measuring contrast.
#   Plot the recorded contrast detected and jump to maximum.
#
# Includes advanced error checking of I2c writing
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
    #time.sleep(0.5)
    print("focus value: {}".format(val))
    #bus.write_byte_data(0x0c,data1,data2)
    #os.system("i2cset -y 0 0x0c %d %d" % (data1,data2))
    time.sleep(0.01)
    cmd = run(["i2cset", "-y", "0", "0x0c", str(data1), str(data2)])
            
    #if cmd.returncode !=0:
    #    print("ERROR Writing to I2c.\nTrying again..\n")
    #    cmd = run(["i2cset", "-y", "0", "0x0c", str(data1), str(data2)])

    return(cmd.returncode)
    
        
def sobel(img):
    # Apply sobel  
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img_sobel = cv2.Sobel(img_gray,cv2.CV_16U,1,1)
    return cv2.mean(img_sobel)[0]

def laplacian(img):
    # Apply laplacian  
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img_sobel = cv2.Laplacian(img_gray,cv2.CV_16U)
    return cv2.mean(img_sobel)[0]
    

def calculation(camera):
    # Calculate contrast
    rawCapture = PiRGBArray(camera) 
    camera.capture(rawCapture,format="bgr", use_video_port=True)
    image = rawCapture.array
    rawCapture.truncate(0)
    return laplacian(image)
    
    
if __name__ == "__main__":
    # Open camera
    camera = picamera.PiCamera()
    camera.start_preview(fullscreen=False, window = (1240, 0, 640, 480))
    camera.resolution = (640, 480)
    # Wait
    time.sleep(0.01)
    print("Start focusing")
    
    # Variables needed for logging
    max_index = 10
    max_value = 0.0
    focal_distance = 0
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
        
        # Append result to list
        val_list.append(val)
        focal_list.append(focal_distance)
        
        # Print to user
        print("focus score " + str(val))
        print("\n")

        # Find the maximum image clarity
        if val > max_value:
            max_index = focal_distance
            max_value = val
            
        
        #Increase the focal distance
        focal_distance += 5
        # If we reach maximum focal distance
        if focal_distance > 1000:
            #leave loop
            break

    # Return focus to the maximum contrast result recorded
    return_val = focusing(max_index)
        
    # Check for errors writing to i2c, try again
    if return_val !=0:
        print("ERROR Writing to I2c.\nTrying again..\n")
        focusing(max_index)
    
    # Wait 1 second
    time.sleep(1)

    # Set camera resolution to maximum
    camera.resolution = (3280,2464)

    # Capture image and save to file.
    camera.capture("test.jpg")

    # Print details to user
    print("max index = %d,max value = %lf" % (max_index, max_value))
    #while True:
    #   time.sleep(1)
    
    # Close camera
    camera.stop_preview()
    camera.close()
    
    # Plot the contrast measurement on the Y-axis, against focal distance on X
    x = np.arange(0, len(val_list), 1)
    
    plt.plot(focal_list, val_list)
    plt.title("Focal Sweep vs. Image Quality")
    plt.xlabel("Focal Distance")
    plt.ylabel("Contrast")
    plt.show()