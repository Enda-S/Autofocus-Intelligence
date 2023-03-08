# ---------------------------------------------------------------
# Script name : load_camera.py
# Created by  : Enda Stockwell
# Adapted from : www.arducam.com/
# ---------------------------------------------------------------
# Description:
# Run before focal_sweep.py to enable camera
# ---------------------------------------------------------------

import cv2
cap = cv2.VideoCapture(3)


while True:
    ret, frame = cap.read()
    cv2.imshow("Arducam", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break