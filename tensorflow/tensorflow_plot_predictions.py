# ---------------------------------------------------------------
# Script name : tensorflow_plot_predictions.py
# Created by  : Enda Stockwell
# ---------------------------------------------------------------
# Description:
#   Plot the predictions produced by the keras network on unseen data
# ---------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv
from matplotlib import pyplot as plt

# Load the predictions CSV into a pandas dataframe
df = pd.read_csv(r'Tensorflow\\data\\predictions\\predictions.csv',  names=["filename", "determination", "score"])

# Store the variables accordingly
determination = df.loc[:, "determination"]
score = df.loc[:, "score"]
filename = df.loc[:, "filename"]

# Declare lists
formatted_array = []
filename_array = []
var_array =[]

# For each image
for item in range(len(df)):
    filename_array.append(filename[item])
    
    # Read the image
    img = cv.imread("Tensorflow\\data\\unseen_data\\commons_images\\" + filename[item], 1)

    # Resize the image
    img = cv.resize(img, (256, 256), interpolation = cv.INTER_AREA)

    # Store the variance(laplacian(image))
    var_array.append(cv.Laplacian(img, cv.CV_64F).var())

    # Check if image was predicted as sharp or blurry
    if determination[item] == "blurry":
        formatted_array.append(score[item]*-1)
    else:
        formatted_array.append(score[item])


# Plot the predictions vs. variance(laplacian(image))
fig, ax = plt.subplots()
plt.title("AI Prediction Confidence vs. Contrast Measured", fontsize=24)
plt.ylabel("CNN confidenece", fontweight='bold', fontsize = 18)
plt.xlabel("Variance of Laplacian", fontweight='bold',  fontsize = 18)
ax.scatter(var_array, formatted_array, color = 'green')


# Label each point of scatter plot with file name
# for i, txt in enumerate(filename_array):
#     ax.annotate(txt, (var_array[i], formatted_array[i]))

plt.show()


