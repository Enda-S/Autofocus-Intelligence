# ---------------------------------------------------------------
# Script name : conver_model_tflite.py
# Created by  : Enda Stockwell
# ---------------------------------------------------------------
# Description:
# Convert a .h5 Keras model into a .tflite model
#   To be used on the Pi running Tensorflow lite
# ---------------------------------------------------------------


import tensorflow as tf

# Load model
model = tf.keras.models.load_model('models\my_model.h5')

# Convert model to tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save new tflite model
open("converted_model.tflite", "wb").write(tflite_model)