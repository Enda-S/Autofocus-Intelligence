# ---------------------------------------------------------------
# Script name : tensorflow_classify.py
# Created by  : Enda Stockwell
# Adapted from: tensorflow.org
# ---------------------------------------------------------------
# Description:
# Running a Keras network to train and validate on our captured data
# ---------------------------------------------------------------

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from random import shuffle
from tensorflow import keras
from PIL import ImageFilter 

# Set directory for training Dataset
directory = "Tensorflow\\data\\training_data\\objects"

# Set the batch size image dimensions
batch_size = 15
img_height = 256
img_width = 256

# Split data into training/validation split from directory
train_ds = tf.keras.utils.image_dataset_from_directory(
  directory,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


val_ds = tf.keras.utils.image_dataset_from_directory(
  directory,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Find class names from directory setup
class_names = train_ds.class_names
print(class_names)


# Configure dataset for PERFORMANCE
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# Normalize the image data between 0->1
normalization_layer = tf.keras.layers.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]

# Should prove normalization:
print(np.min(first_image), np.max(first_image))


# Create Keras model
num_classes = 2

model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.Dropout(0.2, input_shape=(60,)),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.Dropout(0.2, input_shape=(60,)),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])

# Define model opimization
opt = keras.optimizers.Adam(learning_rate=1e-05)

# Compile model
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=opt, metrics=['accuracy'])

# Fit the model and save history
epochs = 2
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# Save the model
model.save('Tensorflow\\data\\models\\my_model_2.h5')

# Plot Accuracy/ Validation Accuracy/ Loss / Validation Loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# Run a small test on unseen data to make sure it worked

# Load image from URL
sharp_stopsign_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9d/Stop_sign_light_red.svg/512px-Stop_sign_light_red.svg.png?20211116183705"
sharp_stopsign_path = tf.keras.utils.get_file('In Focus stopsign', origin=sharp_stopsign_url)

img = tf.keras.utils.load_img(
    sharp_stopsign_path, target_size=(img_height, img_width)
)

# Convert image to array
img_array = tf.keras.utils.img_to_array(img)

# Add a batch dimension 
img_array = tf.expand_dims(img_array, 0)

# Predict on the unseen image
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

# Print results
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
# Show image
plt.imshow(img)
plt.show()



# Apply a blur to the earlier image
blur_img = img.filter(ImageFilter.GaussianBlur(20))

# Convert image to array
img_array_2 = tf.keras.utils.img_to_array(blur_img)

# Add a batch dimension 
img_array_2 = tf.expand_dims(img_array_2, 0)

# Predict on the unseen image
predictions_2 = model.predict(img_array_2)
score_2 = tf.nn.softmax(predictions_2[0])

# Print results
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score_2)], 100 * np.max(score_2))
)
# Show image
plt.imshow(blur_img)
plt.show()