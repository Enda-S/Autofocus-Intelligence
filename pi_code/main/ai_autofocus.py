# ---------------------------------------------------------------
# Script name : ai_autofocus.py
# Created by  : Enda Stockwell
# Adapted from : www.arducam.com/ and tensorflow.org
# ---------------------------------------------------------------
# Description:
# Sweep focal range whilst predicting focal score.
#   If score starts rolling off => Jump back to maximum and capture.
#
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

	# Print to user initialized focal distance
    print("focus value: {}".format(val))

	# Wait before setting focal distance
    time.sleep(0.01)
    
    # Write to the i2c bus
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
	# Wait
	time.sleep(.1)
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
		classname, score = prediction(interpreter, camera)
		
		# Append recorded contrast to list
		if classname == "blurry":
			# If blurry we will multiply the confidence score by -1
			print("focus score " + str(score*-1))
			val_list.append(score*-1)
		else:
			print("focus score " + str(score))
			val_list.append(score)
		
		focal_list.append(focal_distance)
		
		
		print("predicted class " + classname)
		print("\n")
		
		#Find the maximum image clarity
		if score > max_value:
			max_index = focal_distance
			max_value = score
			
		# If the image clarity starts to decrease
		if score < last_value:
			dec_count += 1
		else:
			dec_count = 0

		# If the image clarity is reduced by six consecutive frames
		if dec_count > 6:
			# Stop sweeping
			break
		last_value = score
		
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
	time.sleep(1)

	camera.resolution = (3280,2464)
	# Save image to file.
	camera.capture("highest_prediction_capture.jpg")
	print("max index = %d,max value = %lf" % (max_index, max_value))

	# Close camera
	camera.stop_preview()
	camera.close()


	# Plot swept focal range and AI predicted score
	x = np.arange(0, len(val_list), 1)

	plt.plot(focal_list, val_list)
	plt.title("AI Prediction vs. focal distance")
	plt.xlabel("Focal Distance")
	plt.ylabel("Image Quality")
	plt.show()
