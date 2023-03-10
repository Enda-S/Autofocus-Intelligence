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
import numpy as np 

# Load the predictions CSV into a pandas dataframe
df = pd.read_csv(r'Tensorflow\\data\\predictions\\predictions.csv',  names=["filename", "determination", "score"])

# Store the variables accordingly
determination = df.loc[:, "determination"]
score = df.loc[:, "score"]
filename = df.loc[:, "filename"]

# Declare lists
ai_list = []
filename_list = []
contrast_list =[]

# For each image
for item in range(len(df)):
    filename_list.append(filename[item])
    
    # Read the image
    img = cv.imread("Tensorflow\\data\\unseen_data\\commons_images\\" + filename[item], 1)

    # Resize the image
    img = cv.resize(img, (256, 256), interpolation = cv.INTER_AREA)

    # Store the variance(laplacian(image))
    contrast_list.append(cv.Laplacian(img, cv.CV_64F).var())

    # Check if image was predicted as sharp or blurry and store accordingly
    if determination[item] == "blurry":
        # This score would normally be between [50, 100]
        # We multiply it by -1 and subtract 50 set it between [0,-50]
        ai_list.append((score[item]*-1)+50)
    else:
        # This score would normally be between [50, 100]
        # We subtract 50 set it between [0,50]
        ai_list.append(score[item]-50)

    
# Function to normalize results to [0,1]
def normalize_array(array):
	normalized = np.interp(array, (array.min(), array.max()), (0,1))
	return(normalized)


ai_array_normalized = normalize_array(np.array(ai_list))
contrast_array_normalized = normalize_array(np.array(contrast_list))

# Plot the predictions vs. variance(laplacian(image))
fig, ax = plt.subplots()
plt.title("AI Prediction Confidence vs. Contrast Measured", fontsize=24)
plt.ylabel("CNN confidenece", fontweight='bold', fontsize = 18)
plt.xlabel("Variance of Laplacian", fontweight='bold',  fontsize = 18)
ax.scatter(contrast_array_normalized, ai_array_normalized, color = 'green')


# Label each point of scatter plot with file name
for i, txt in enumerate(filename_list):
     ax.annotate(txt, (contrast_array_normalized[i], ai_array_normalized[i]))

plt.show()


