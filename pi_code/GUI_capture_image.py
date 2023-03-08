# ---------------------------------------------------------------
# Script name : GUI_capture_image.py
# Created by  : Enda Stockwell
# Adapted from : www.tensorflow.org and www.arducam.com/
# ---------------------------------------------------------------
# Description:
# TKinter GUI built to predict and measure contrast from an image
#   The image is to be captured from a live camera stream
#       Using Picamera on raspberry Pi 4
#
# Loads and predicts from a tensorflow lite model
# ---------------------------------------------------------------

import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
import tflite_runtime as tf
from tflite_runtime.interpreter import Interpreter 
import os
import sys
import numpy as np
import smbus

# Load Picamera and bus for i2c control
bus = smbus.SMBus(0)
try:
    import picamera
    from picamera.array import PiRGBArray
except:
    print("ERROR: Failed to connect to PiCamera")
    sys.exit(0)

# Load model
model_path = "converted_model.tflite"

# Load the model
interpreter = Interpreter(model_path)
print("Model Loaded Successfully.")

# Allocate tensors
interpreter.allocate_tensors()
_, height, width, _ = interpreter.get_input_details()[0]['shape']
print("Image Shape (", width, ",", height, ")")

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ['blurry', 'focussed']


# Start camera preview window
camera = picamera.PiCamera()
camera.start_preview(fullscreen=False, window = (1240, 0, 640, 480))
camera.resolution = (640, 480)
camera.framerate= 24

rawCapture = PiRGBArray(camera, size=camera.resolution)

# Distance between each focal distance step
focal_step = 50

# Function to step focal distance UP
def up_focus():
    # Get step value
    global focal_step
    # Print details to user
    print ('UP', focal_step)

    # If not at maximum
    if focal_step < 1000:
        # Step by 50
        focal_step += 50
    else:
        # Else at maximum, dont step
        focal_step = focal_step
    
    # Write data to I2c to control the Arducam focus motor
    value = (focal_step<<4) & 0x3ff0
    dat1 = (value>>8)&0x3f
    dat2 = value & 0xf0
    os.system("i2cset -y 0 0x0c %d %d" % (dat1,dat2))
        

# Function to step focal distance DOWN
def down_focus():
    # Get step value
    global focal_step
    # Print details to user
    print ('DOWN', focal_step)
    
    # If at minimum
    if focal_step <50 :
        # Dont step
        focal_step = focal_step
    else:
        # Else step
        focal_step -= 50
    
    # Write data to I2c to control the Arducam focus motor
    value = (focal_step<<4) & 0x3ff0
    dat1 = (value>>8)&0x3f
    dat2 = value & 0xf0
    os.system("i2cset -y 0 0x0c %d %d" % (dat1,dat2))


# Sobel filter
def apply_sobel(img):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    return sobel

# Canny filter
def apply_canny(img):
    canny = cv2.Canny(img, 100, 200)
    return canny

# Laplacian filter
def apply_laplacian(img):
    # Apply Laplacian filter
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    return laplacian

# Measures contrast of image
def calculate_contrast(image):
    # Method 1 (Variance of Laplacian)
    contrast = cv2.Laplacian(image, cv2.CV_64F).var()


    # # Method 2 std_dev/mean
    # # Calculate standard deviation of pixel intensities
    # std_dev = np.std(image)
    
    # #Calculate contrast as standard deviation divided by mean
    # mean = np.mean(image)
    
    # contrast = std_dev / mean


    # # Method 3 (precentage contrast)
    # # Calculate maximum and minimum pixel values
    # max_val = int(np.max(image))
    # min_val = int(np.min(image))
    # print("max,", max_val)
    # print("min,", min_val)
    # # Calculate percentage contrast
    # if min_val ==0 and max_val==0:
    #     percent_contrast = 0
    # else:
    #     percent_contrast = (max_val - min_val) / ((max_val + min_val) / 2) * 50

    # Display the measurement result on GUI
    result_2_label.config(text=f"Contrast Measurement: \n%.2f" %(contrast), width=40,
            height=5, font=('Times New Roman', 15))


# Define function to update the image
def update_image():

    # Apply filters if selected
    if sobel_var.get():
        # Convert to greyscale
        img_grey =  cv2.cvtColor(orig_img, cv2.COLOR_RGB2GRAY)
        # Apply Sobel
        img = apply_sobel(img_grey)
        # Extrapolate a 3rd chanel needed for AI prediction
        pred_img = np.repeat(img[..., np.newaxis], 3, axis=-1)
        # Calculate contrast
        calculate_contrast(img)
        
    elif canny_var.get():
        # Convert to greyscale
        img_grey =  cv2.cvtColor(orig_img, cv2.COLOR_RGB2GRAY)
        # Apply Canny
        img = apply_canny(img_grey)
        # Extrapolate a 3rd chanel needed for AI prediction
        pred_img = np.repeat(img[..., np.newaxis], 3, axis=-1)
        # Calculate contrast
        calculate_contrast(img)
    
    elif laplacian_var.get():
        # Convert to greyscale
        img_grey =  cv2.cvtColor(orig_img, cv2.COLOR_RGB2GRAY)
        # Apply Laplacian
        img = apply_laplacian(img_grey)
        # Extrapolate a 3rd chanel needed for AI prediction
        pred_img = np.repeat(img[..., np.newaxis], 3, axis=-1)
        # Add the absolute value of the minimum intensity value to all pixel values
        img_shifted = img + abs(np.min(img))
        # Calculate contrast
        calculate_contrast(img_shifted)

    # No filter selected           
    else:
        # Convert to greyscale
        img_grey =  cv2.cvtColor(orig_img, cv2.COLOR_RGB2GRAY)
        img = orig_img
        pred_img = orig_img
        # Calculate contrast
        calculate_contrast(img_grey)
    
    
    # Classify/predict on the input image
    classify_image(interpreter, pred_img)

    # Display the image in the GUI
    img = Image.fromarray(img)
    img_tk = ImageTk.PhotoImage(img)
    img_label.configure(image=img_tk)
    img_label.image = img_tk


# Define softmax function as tflite does not include 
def softmax(x):
    return(np.exp(x)/np.exp(x).sum())


# Classify image function
def classify_image(interpreter, image, top_k=1):

    #Preprocess the image to required size and cast
    input_shape = input_details[0]['shape']
    input_tensor= np.array(np.expand_dims(image,0), dtype = np.float32)
    input_index = interpreter.get_input_details()[0]["index"]
    interpreter.set_tensor(input_index, input_tensor)
    interpreter.invoke()
    output_details = interpreter.get_output_details()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    pred = np.squeeze(output_data)
    
    highest_pred_loc = np.argmax(pred)
    
    output_probs = softmax(output_data)
    pred_label = np.argmax(output_probs)
    #score = softmax(pred[0])
    classed_name = class_names[pred_label]
    result_label.config(text=f"AI Prediction: \nThis image most likely belongs to class " + str(classed_name) + " with a %.2f" % (100 * np.max(output_probs)) + " percent confidence.", font=('Times New Roman', 15))


# Define function to capture an image and update the GUI
def capture_image():

    # Capture using Picamera
    camera.capture(rawCapture, format="rgb")
    # Convert image to array
    img = rawCapture.array
    # Truncate - needed to clear stream for next frame 
    rawCapture.truncate(0)

    # Load and resize the image
    img = cv2.resize(img, (256, 256))

    # Store the original image
    global orig_img
    orig_img = img.copy()

    #Calculate contrast of image
    calculate_contrast(orig_img)
    
    #Classify/predict on image
    classify_image(interpreter, orig_img)

    # Display the image in the GUI
    img_array = Image.fromarray(orig_img)
    img_tk = ImageTk.PhotoImage(img_array)
    img_label.configure(image=img_tk)
    img_label.image = img_tk



# Create the GUI
root = tk.Tk()
root.title('Autofocus Intelligence')
root.geometry("800x600")

# Create a frame for the capture image button
cap_button_frame = tk.Frame(root)
cap_button_frame.pack()

# Create a frame for the filter buttons
button_frame = tk.Frame(root)
button_frame.pack()

# Create a frame for the up button
up_button_frame = tk.Frame(root)
up_button_frame.pack()

up_button = tk.Button(up_button_frame, text="UP", command=up_focus, font=('Arial', 15, 'bold'))
up_button.pack(side="left", padx=5, pady=5)

# Create a frame for the down button
down_button_frame = tk.Frame(root)
down_button_frame.pack()

down_button = tk.Button(down_button_frame, text="DOWN",  command=down_focus, font=('Arial', 15, 'bold'))
down_button.pack(side="left", padx=5, pady=5)

# Create a button to capture an image
capture_button = tk.Button(cap_button_frame, text="Capture Image", command=capture_image, font=('Arial', 15, 'italic'))
capture_button.pack(side="left", padx=5, pady=5)

# Create a label for the image
img_label = tk.Label(root)
img_label.pack()

# Create button and variable for applying Sobel
sobel_var = tk.BooleanVar(value=False)
sobel_button = tk.Checkbutton(button_frame, text="Sobel Filter", variable=sobel_var, command=update_image, font=('Arial', 15, 'bold'))
sobel_button.pack(side="left", padx=5, pady=5)

# Create button and variable for applying Canny
canny_var = tk.BooleanVar(value=False)
canny_button = tk.Checkbutton(button_frame, text="Canny Filter", variable=canny_var, command=update_image, font=('Arial', 15, 'bold'))
canny_button.pack(side="left", padx=5, pady=5)

# Create button and variable for applying Laplacian
laplacian_var = tk.BooleanVar(value=False)
laplacian_button = tk.Checkbutton(button_frame, text="Laplacian Filter", variable=laplacian_var, command=update_image, font=('Arial', 15, 'bold'))
laplacian_button.pack(side="left", padx=5, pady=5)

# Create a label to display the AI predicted result
result_label = tk.Label(root, text='AI Predicted Sharpness Score: ',  font=('Times New Roman', 15))
result_label.pack()

# Create a label to display the Contrast measurement result
result_2_label = tk.Label(root, text='Contrast Measurement: ', font=('Times New Roman', 15))
result_2_label.pack()



# Start the GUI
root.mainloop()
    
