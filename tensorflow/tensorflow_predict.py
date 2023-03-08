# ---------------------------------------------------------------
# Script name : tensorflow_predict.py
# Created by  : Enda Stockwell
# Adapted from: tensorflow.org
# ---------------------------------------------------------------
# Description:
#   Make predictions on unseen testing data
# ---------------------------------------------------------------

import tensorflow as tf
import numpy as np
import os
import csv


# Set the batch size image dimensions
batch_size = 15
img_height = 256
img_width = 256

# Set directory for prediction dataset
predict_directory = "Tensorflow\\data\\unseen_data\\commons_images"

# Set class names
class_names = ['blurry', 'focussed']
print(class_names)

# Load previously fitted model
model = tf.keras.models.load_model('Tensorflow\\data\\models\\my_model.h5')

# Compile model
model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])


# Show the model architecture
model.summary()

# Declare list of scores
score_vals = []

# load all images from directory into a list
for file in os.listdir(predict_directory):

  img_path = os.path.join(predict_directory, file)
  img = tf.keras.utils.load_img(
  img_path, target_size=(img_height, img_width)
  )

  filename = file

  # Normalize the image
  #processed_image = np.array(img, dtype="float") / 255.0

  # Convert image to array
  img_array = tf.keras.utils.img_to_array(img)

  # Add a batch dimension 
  img_array = tf.expand_dims(img_array, 0) # Create a batch

  # Predict on the unseen
  prediction = model.predict(img_array)
  predicted_classes = prediction.argmax(axis=1)
  print(predicted_classes)
  score = tf.nn.softmax(prediction[0])

  print(
  "This image most likely belongs to {} with a {:.2f} percent confidence."
  .format(class_names[np.argmax(score)], 100 * np.max(score))
  )

  # Append predictions list
  score_vals.append([filename, class_names[np.argmax(score)], 100 * np.max(score)])


# Store predictions in a CSV
with open("Tensorflow\\data\\predictions\\predictions.csv","w+") as my_csv:
  csvWriter = csv.writer(my_csv, delimiter=',')
  csvWriter.writerows(score_vals)


