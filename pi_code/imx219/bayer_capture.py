# ---------------------------------------------------------------
# Script name : bayer_capture.py
# Created by  : Enda Stockwell
# ---------------------------------------------------------------
# Description:
# Capture images in jpeg+raw format
#
# ---------------------------------------------------------------

import time
import picamera
import numpy as np

with picamera.PiCamera() as camera:
    # Let the camera warm up for a couple of seconds
    time.sleep(2)
    
    camera.capture("testb.jpg", format='jpeg', bayer=True)
    time.sleep(0.1)
    camera.capture("testn.jpg", format='jpeg')
