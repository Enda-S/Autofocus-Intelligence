# ---------------------------------------------------------------
# Script name : GUI.py
# Created by  : Enda Stockwell
# Adapted from : www.tensorflow.org and www.arducam.com/
# ---------------------------------------------------------------
# Description:
# TKinter GUI built to predict and measure contrast from an image.
#   The image is to be uploaded locally from filesystem.
#
# Loads and predicts from a Keras Tensorflow model.
# ---------------------------------------------------------------

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np

import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('C:\\Users\\Enda\\Desktop\\2023 College\\Masters\\Tensorflow\\data\\models\\my_model copy-best.h5')

model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])
class_names = ['blurry', 'focussed']

# Define filter functions
def apply_sobel(img):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    return sobel

def apply_canny(img):
    canny = cv2.Canny(img, 100, 200)
    return canny

def apply_laplacian(img):
    # Apply Laplacian filter
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    return laplacian

def calculate_contrast(image):
    # Code a button or list to select contrast score
    # Method 1 (laplacian variance)
    
    contrast = cv2.Laplacian(image, cv2.CV_64F).var()

    # Method 2 std_dev/mean
    # # calculate standard deviation of pixel intensities
    # std_dev = np.std(image)
    
    # # calculate contrast as standard deviation divided by mean
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

    result_2_label.config(text=f"Contrast Measurement: \n%.2f" %(contrast), width=40,
            height=5, font=('Times New Roman', 15))

# Define function to update the image
def update_image():
    # Apply filters if selected
    if sobel_var.get():
        img_grey =  cv2.cvtColor(orig_img, cv2.COLOR_RGB2GRAY)
        img = apply_sobel(img_grey)
        # extrapolate a 3rd chanel
        pred_img = np.repeat(img[..., np.newaxis], 3, axis=-1)
        calculate_contrast(img)
        
    elif canny_var.get():
        img_grey =  cv2.cvtColor(orig_img, cv2.COLOR_RGB2GRAY)
        img = apply_canny(img_grey)
        pred_img = np.repeat(img[..., np.newaxis], 3, axis=-1)
        calculate_contrast(img)
    
    elif laplacian_var.get():
        img_grey =  cv2.cvtColor(orig_img, cv2.COLOR_RGB2GRAY)
        img = apply_laplacian(img_grey)
        pred_img = np.repeat(img[..., np.newaxis], 3, axis=-1)
        
        # Add the absolute value of the minimum intensity value to all pixel values
        img_shifted = img + abs(np.min(img))

        calculate_contrast(img_shifted)
    else:
        img_grey =  cv2.cvtColor(orig_img, cv2.COLOR_RGB2GRAY)
        img = orig_img
        pred_img = orig_img
        calculate_contrast(img_grey)

    predict_image(pred_img)

    # Display the image in the GUI
    img = Image.fromarray(img)
    img_tk = ImageTk.PhotoImage(img)
    img_label.configure(image=img_tk)
    img_label.image = img_tk

    

def predict_image(input_img):
    # Preprocess the image and make a prediction
    #img = np.array(input_img)
    #img = img / 255.0
    #img = np.expand_dims(img, axis=0)

    
    img_array = tf.keras.utils.img_to_array(input_img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    
    prediction = model.predict(img_array)

    #predicted_classes = prediction.argmax(axis=1)

    score = tf.nn.softmax(prediction[0])

    #sharpness = prediction[0][0]
    # Display the result
    result_label.config(text=f"AI Prediction: \nThis image most likely belongs to class " + str(class_names[np.argmax(score)]) + " with a %.2f" % (100 * np.max(score)) + " percent confidence.", font=('Times New Roman', 15))


# Define function to select an image and update the GUI
def select_image():
    # Open a file dialog to select an image
    file_path = filedialog.askopenfilename()

    # Load and resize the image
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))

    # Store the original image
    global orig_img
    orig_img = img.copy()

    predict_image(orig_img)
    calculate_contrast(orig_img)

    # Display the image in the GUI
    img_array = Image.fromarray(orig_img)
    img_tk = ImageTk.PhotoImage(img_array)
    img_label.configure(image=img_tk)
    img_label.image = img_tk




# Create the GUI
root = tk.Tk()
root.title('Autofocus Intelligence')
root.geometry("800x600")

# Create a frame for the select image button
sel_button_frame = tk.Frame(root)
sel_button_frame.pack()

# Create a frame for the filter buttons
button_frame = tk.Frame(root)
button_frame.pack()


# Create a button to select an image
select_button = tk.Button(sel_button_frame, text="Select Image", command=select_image, font=('Arial', 15, 'italic'))
select_button.pack(side="left", padx=5, pady=5)

# Create buttons for applying filters
sobel_var = tk.BooleanVar(value=False)
sobel_button = tk.Checkbutton(button_frame, text="Sobel Filter", variable=sobel_var, command=update_image, font=('Arial', 15, 'bold'))
sobel_button.pack(side="left", padx=5, pady=5)

canny_var = tk.BooleanVar(value=False)
canny_button = tk.Checkbutton(button_frame, text="Canny Filter", variable=canny_var, command=update_image, font=('Arial', 15, 'bold'))
canny_button.pack(side="left", padx=5, pady=5)

laplacian_var = tk.BooleanVar(value=False)
laplacian_button = tk.Checkbutton(button_frame, text="Laplacian Filter", variable=laplacian_var, command=update_image, font=('Arial', 15, 'bold'))
laplacian_button.pack(side="left", padx=5, pady=5)


# Create a label for the image
img_label = tk.Label(root)
img_label.pack()


# Create a label to display the result
result_label = tk.Label(root, text='AI Predicted Sharpness Score: ',  font=('Times New Roman', 15))
result_label.pack()

# Create a label to display the result
result_2_label = tk.Label(root, text='Contrast Measurement: ', font=('Times New Roman', 15))
result_2_label.pack()

# Start the GUI
root.mainloop()
