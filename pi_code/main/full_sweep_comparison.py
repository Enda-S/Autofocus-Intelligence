# ---------------------------------------------------------------
# Script name : full_sweep_comparison.py
# Created by  : Enda Stockwell
# Adapted from : www.arducam.com/ and www.tensorflow.org
# ---------------------------------------------------------------
# Description:
# Sweep entire focal range whilst measuring contrast and predicting focal score.
#
# Plots both contrast measured, and AI prediction vs. focal distance.
# Both normalized 0->1 for convenience.
# ---------------------------------------------------------------

import cv2
import numpy as np
import os
import time
import smbus
import matplotlib.pyplot as plt
import tflite_runtime as tf
from tflite_runtime.interpreter import Interpreter 
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
    time.sleep(0.01)
    # Write to I2C bus to change focal distance on Arducam sensor
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
    
    image = cv2.resize(image, (256, 256))
    return laplacian(image)
    

# Make AI prediction
def prediction(interpreter, camera):
	# Get raw camera data
    rawCapture = PiRGBArray(camera) 
    # Capture frame
    camera.capture(rawCapture, format="bgr", use_video_port=True)
    # Convert captured image into an array
    image = rawCapture.array
    # Clear stream for the next frame
    rawCapture.truncate(0)
    
	# resize captured image to (256,256)
    img = cv2.resize(image, (256, 256))

    
    #Preprocess the image to required size and cast
    input_shape = input_details[0]['shape']
    input_tensor= np.array(np.expand_dims(img,0), dtype = np.float32)
    input_index = interpreter.get_input_details()[0]["index"]
    interpreter.set_tensor(input_index, input_tensor)
    interpreter.invoke()
    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
	# Make the prediction
    pred = np.squeeze(output_data)
    highest_pred_loc = np.argmax(pred)
    
    output_probs = softmax(output_data)
    pred_label = np.argmax(output_probs)
    classed_name = str(class_names[pred_label])
    confidence = (100 * np.max(output_probs))
    
    return classed_name, confidence

# Define softmax function as tflite does not include 
def softmax(x):
    return(np.exp(x)/np.exp(x).sum())

# Function to normalize results to [0,1]
def normalize_array(array):
	normalized = np.interp(array, (array.min(), array.max()), (0,1))
	return(normalized)


if __name__ == "__main__":
    
	# Load model
	model_path = "converted_model.tflite"
	interpreter = Interpreter(model_path)
	print("Model Loaded Successfully.")

	# Allocate tensors
	interpreter.allocate_tensors()
	_, height, width, _ = interpreter.get_input_details()[0]['shape']
	print("Image Shape (", width, ",", height, ")")

	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()

	class_names = ['blurry', 'focussed']
	
	# Open camera
	camera = picamera.PiCamera()
	camera.start_preview(fullscreen=False, window = (1240, 0, 640, 480))
	camera.resolution = (256, 256)
	time.sleep(.1)

	# Variables needed for logging
	max_index = 10
	max_value = 0.0
	last_value = 0.0
	dec_count = 0
	focal_distance = 10
	val_list_ai = []
	val_list_cont = []

	focal_list = []
		

	while True:
		time.sleep(0.5)
		# Adjust focus
		return_val = focusing(focal_distance)
		
		# Check for errors writing to i2c, try again
		if return_val !=0:
			focusing(focal_distance)

		# Calculate image contrast
		classname, score = prediction(interpreter, camera)
        
		# Predict image score
		val = calculation(camera)
		
    	# Check if image was predicted as sharp or blurry and store accordingly
		if classname == "blurry":
        	# This score would normally be between [50, 100]
        	#We multiply it by -1 and subtract 50 set it between [0,-50]
			val_list_ai.append((score*-1)+50)
		else:
            # This score would normally be between [50, 100]
        	# We subtract 50 set it between [0,50]
			val_list_ai.append(score-50)
                        
		# Store results in lists
		val_list_cont.append(val)
		focal_list.append(focal_distance)
		
		
		# Increase the focal distance
		focal_distance += 15
		if focal_distance > 1000:
			break


	time.sleep(.1)
    # Close camera
	camera.stop_preview()
	camera.close()

	# Convert result lists to arrays and normalize to [0,1]
	ai_array = normalize_array(np.array(val_list_ai))
	cont_array = normalize_array(np.array(val_list_cont))
    
	# Plot results
	x = np.arange(0, len(ai_array), 1)

	plt.subplot(2,1,1)
	plt.plot(focal_list, ai_array)
	plt.title("AI Prediction vs. focal distance")
	plt.xlabel("Focal Distance")
	plt.ylabel("Image Quality")
	
	plt.subplot(2,1,2)
	plt.plot(focal_list, cont_array)
	plt.title("Contrast measurement vs. focal distance")
	plt.xlabel("Focal Distance")
	plt.ylabel("Image Quality")
	
	
	plt.show()
