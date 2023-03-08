# ---------------------------------------------------------------
# Script name : focal_sweep.py
# Created by  : Enda Stockwell
# Adapted from : www.arducam.com/ and www.picamera.readthedocs.io/
# ---------------------------------------------------------------
# Description:
# Manual focus adjustment from user input using Pygame.
#   Allows for capturing of single jpeg+raw images
#       and capturing of focal sweep.
# ---------------------------------------------------------------

import os
import time
import sys
import threading
import pygame,sys
from pygame.locals import *
from time import ctime, sleep
import numpy as np

# Initialize Picamera and smbus for Arducam's I2C focus control
import smbus
bus = smbus.SMBus(0)
try:
    import picamera
    from picamera.array import PiRGBArray
except:
    sys.exit(0)

# Define number of images to sweep
images = 100

# Open pygame window
pygame.init()
screen=pygame.display.set_mode((320,240),0,32)
pygame.key.set_repeat(100)


def runFocus():
    # Init focal distance
    temp_val = 512

    # Define focal step
    step = round(1000/images)
    
    # Open camera
    camera = picamera.PiCamera()
    camera.start_preview(fullscreen=False, window = (1240, 0, 640, 480))

    # Set camera resolution
    camera.resolution = (640, 480)
    running = True

    # Pygame loop
    while running:
        for event in pygame.event.get():
            # If user presses a key
            if event.type ==KEYDOWN:

                # Print current focal distance
                print (temp_val)
                
                # If Key pressed was up
                if event.key == K_UP:
                    # Print details to user
                    print ('UP')
                    # If not at maximum
                    if temp_val < 1000:
                        # Increment focus
                        temp_val += 10
                    else:
                        # At maximum, don't increment
                        temp_val = temp_val
                        
                    # Write data to I2c to control the Arducam focus motor
                    value = (temp_val<<4) & 0x3ff0
                    dat1 = (value>>8)&0x3f
                    dat2 = value & 0xf0
                    os.system("i2cset -y 0 0x0c %d %d" % (dat1,dat2))
                    
                # If Key pressed was Down   
                elif event.key==K_DOWN:
                    # Print details to user
                    print ('DOWN')
                    # If at minimum
                    if temp_val <12 :
                        # Don't decrement
                        temp_val = temp_val
                    else:
                        # Otherwise decrement
                        temp_val -= 10

                    # Write data to I2c to control the Arducam focus motor
                    value = (temp_val<<4) & 0x3ff0
                    dat1 = (value>>8)&0x3f
                    dat2 = value & 0xf0
                    os.system("i2cset -y 0 0x0c %d %d" % (dat1,dat2))
                    
                # If Key pressed was escape   
                elif event.key==K_ESCAPE:
                    # Print exit details to user
                    print ('Exiting')
                    # Disable running variable window
                    running = False
                    # Close pygame window
                    pygame.quit()
                    # Wait
                    time.sleep(0.5)
                    # Disable camera preview
                    camera.stop_preview()
                    # Close camera
                    camera.close()

                # If Key pressed was S   
                elif event.key==K_s:
                    # Print sweeping details to user
                    print ('Sweeping')
                    # Disable camera preview
                    camera.stop_preview()
                    # Wait
                    time.sleep(0.5)
                    # Increase resolution to max for high quality capture
                    camera.resolution = (3280,2464)
                    # Set focal distance to 0
                    temp_val = 0
                    # Write data to I2c to control the Arducam focus motor
                    value = (temp_val<<4) & 0x3ff0
                    dat1 = (value>>8)&0x3f
                    dat2 = value & 0xf0
                    os.system("i2cset -y 0 0x0c %d %d" % (dat1,dat2))
                    
                    # Loop through number of images to capture
                    for image in range(images):
                        # Stepping through the focal distance
                        if temp_val < 1000:
                            # Adding the step to each increment of focal distance
                            temp_val += step
                        else:
                            temp_val = temp_val

                        # Sleep between focal distance changing
                        time.sleep(0.1)

                        # Write data to I2c to control the Arducam focus motor
                        value = (temp_val<<4) & 0x3ff0
                        dat1 = (value>>8)&0x3f
                        dat2 = value & 0xf0
                        os.system("i2cset -y 0 0x0c %d %d" % (dat1,dat2))
                        
                        # Capture image and save to file
                        camera.capture("test_images/test_focal_distance_" + str(temp_val) + ".jpg")
                        print("Image taken at focal distance " + str(temp_val))
                        
                    # Re-load preview with lower resolution
                    camera.start_preview(fullscreen=False, window = (1240, 0, 640, 480))
                    camera.resolution = (640, 480)

                    print("done")
                    
                # If Key pressed was C 
                elif event.key==K_c:
                    # Print capture details to user
                    print ('Capturing image')
                    # Disable camera preview
                    camera.stop_preview()
                    # Capture raw YUV data
                    camera.capture('test_images/raw/yuv_image.data', 'yuv')
                    # Change camera settings
                    camera.iso = 100
                    camera.awb_mode = "off"
                    # Wait
                    time.sleep(2)
                    # Capture jpg+raw image
                    camera.capture('test_images/raw/bayer_image.jpg', 'jpeg', bayer=True)
                    
                    # Capture raw bayer data from camera
                    with picamera.array.PiBayerArray(camera) as stream:
                        camera.capture(stream, 'jpeg', bayer=True)
                        # Demosaic data and write to output (just use stream.array if you
                        # want to skip the demosaic step)
                        output = (stream.demosaic() >> 2).astype(np.uint8)
                        with open('test_images/raw/bayer_image.data', 'wb') as f:
                            output.tofile(f)
                    
                    print ('Finished Capturing')

                    # Re-load preview with lower resolution
                    camera.start_preview(fullscreen=False, window = (1240, 0, 640, 480))
                    camera.resolution = (640, 480)
    
if __name__ == "__main__":
 runFocus()
