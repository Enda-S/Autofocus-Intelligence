# ---------------------------------------------------------------
# Script name : focal_sweep.py
# Created by  : Enda Stockwell
# Adapted from : www.arducam.com/
# ---------------------------------------------------------------
# Description:
# Focal sweep user entry code written with V4L2 support
#   Built for the IMX519 using libcamera instead of Picamera
# ---------------------------------------------------------------

import os
import time
import sys
import threading
import pygame,sys
from pygame.locals import *
from time import ctime, sleep
import cv2

#os.system("libcamera-hello -t 0")


images = 10
pygame.init()
screen=pygame.display.set_mode((320,240),0,32)
pygame.key.set_repeat(100)

def runFocus():
    temp_val = 0
    value = (temp_val<<4) & 0x3ff0
    dat1 = (value>>8)&0x3f
    dat2 = value & 0xf0
    os.system("v4l2-ctl -c focus_absolute=%d -d /dev/v4l-subdev1" % temp_val)
                    

    step = round(4095/images)
    #open camera
    #os.system("libcamera-hello -t 0 --width 640 --height 480")

    print("test")
    running = True

    while running:
        for event in pygame.event.get():
            if event.type ==KEYDOWN:
                print (temp_val)
                if event.key == K_UP:
                    print ('UP')
                    if temp_val < 4050:
                        temp_val += 50
                    else:
                        temp_val = temp_val
                        
                    value = (temp_val<<4) & 0x3ff0
                    dat1 = (value>>8)&0x3f
                    dat2 = value & 0xf0
                    os.system("v4l2-ctl -c focus_absolute=%d -d /dev/v4l-subdev1" % temp_val)
                    
                    
                elif event.key==K_DOWN:
                    print ('DOWN')
                    if temp_val <50 :
                        temp_val = temp_val
                    else:
                        temp_val -= 50
                    
                    value = (temp_val<<4) & 0x3ff0
                    dat1 = (value>>8)&0x3f
                    dat2 = value & 0xf0
                    
                    os.system("v4l2-ctl -c focus_absolute=%d -d /dev/v4l-subdev1" % temp_val)
                    

                elif event.key==K_ESCAPE:
                    print ('Exiting')
                    running = False
                    pygame.quit()
                    time.sleep(0.5)

                    
                elif event.key==K_s:
                    print ('Sweeping')
                    time.sleep(0.5)
                    
                    temp_val = 0
                    value = (temp_val<<4) & 0x3ff0
                    dat1 = (value>>8)&0x3f
                    dat2 = value & 0xf0
                    
                    for image in range(images):
                        if temp_val < 4095:
                            temp_val += step
                        else:
                            temp_val = temp_val
                        
                        value = (temp_val<<4) & 0x3ff0
                        dat1 = (value>>8)&0x3f
                        dat2 = value & 0xf0
                        os.system("v4l2-ctl -c focus_absolute=%d -d /dev/v4l-subdev1" % temp_val)
                        cv2.imwrite("test.jpg", frame)
                        os.system("libcamera-jpeg -o test.jpg")
                        
                        #save image to file.
                        print("Image taken at focal distance " + str(temp_val))
                        
                        

    
if __name__ == "__main__":
 runFocus()
