# ---------------------------------------------------------------
# Script name : capture_data.py
# Created by  : Enda Stockwell
# ---------------------------------------------------------------
# Description:
# Vary focal distance from min->max and capture 500 images
#
# Used in dataset generation
# ---------------------------------------------------------------

import os
import time
import sys
import threading
import pygame,sys
from pygame.locals import *
from time import ctime, sleep
import numpy as np
from subprocess import run
import smbus

# Initialize Picamera and smbus for Arducam's I2C focus control
bus = smbus.SMBus(0)
try:
    import picamera
    from picamera.array import PiRGBArray
except:
    sys.exit(0)

# Declare number of images to capture
images = 500

# Initialize pygame window
pygame.init()
screen=pygame.display.set_mode((320,240),0,32)
pygame.key.set_repeat(100)

def runFocus():
    temp_val = 512

    step = 3
    # Open camera
    camera = picamera.PiCamera()
    
    camera.start_preview(fullscreen=False, window = (1240, 0, 640, 480))
    #set camera resolution to 640x480
    camera.resolution = (640, 480)

    running = True

    while running:
        for event in pygame.event.get():
            if event.type ==KEYDOWN:
                print (temp_val)
                if event.key==K_ESCAPE:
                    print ('Exiting')
                    running = False
                    pygame.quit()
                    time.sleep(0.2)
                    camera.stop_preview()
                    camera.close()
                    
                elif event.key==K_s:
                    print ('Sweeping')
                    camera.stop_preview()
                    time.sleep(0.2)
                    
                    camera.resolution = (3280, 2464)
                    #camera.resolution = (1920, 1080)
                    camera.shutter_speed = 33000
                    temp_val = 0
                    value = (temp_val<<4) & 0x3ff0
                    dat1 = (value>>8)&0x3f
                    dat2 = value & 0xf0
                    
                    
                    cmd = run(["i2cset", "-y", "0", "0x0c", str(dat1), str(dat2)])
            
                    if cmd.returncode !=0:
                        print("ERROR Writing to I2c.\nTrying again..\n")
                        cmd = run(["i2cset", "-y", "0", "0x0c", str(dat1), str(dat2)])
                    
                    camera.capture("image_capturing/current_capture/focal_distance_" + str(temp_val) + ".jpg", format='jpeg', bayer=True)
                    print("Image taken at focal distance " + str(temp_val))
                                       
                    
                    for image in range(images):
                        if temp_val < 1000:
                            temp_val += step
                        else:
                            break
                        time.sleep(0.01)
                        value = (temp_val<<4) & 0x3ff0
                        dat1 = (value>>8)&0x3f
                        dat2 = value & 0xf0
                        cmd = run(["i2cset", "-y", "0", "0x0c", str(dat1), str(dat2)])
            
                        if cmd.returncode !=0:
                            print("ERROR Writing to I2c.\nTrying again..\n")
                            cmd = run(["i2cset", "-y", "0", "0x0c", str(dat1), str(dat2)])
                        
                        #save image to file.
                        camera.capture("image_capturing/current_capture/focal_distance_" + str(temp_val) + ".jpg", format='jpeg', bayer=True)
                        print("Image taken at focal distance " + str(temp_val))
                        
                        
                    camera.start_preview(fullscreen=False, window = (1240, 0, 640, 480))
                    #set camera resolution to 640x480(Small resolution for faster speeds.)
                    camera.resolution = (640, 480)
                    print ('Finished Capturing')
                    
                    
                elif event.key==K_c:
                    camera.resolution = (3280, 2464)
                    #camera.shutter_speed = 1200000
                    time.sleep(2)
                    print ('Capturing test image')
                    camera.capture('image_capturing/test_capture/test.jpeg', 'jpeg')
                    camera.resolution = (640, 480)

    
if __name__ == "__main__":
 runFocus()


