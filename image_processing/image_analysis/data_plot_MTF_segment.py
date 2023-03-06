# ---------------------------------------------------------------
# Script name : data_plot_MTF_segment.py
# Created by  : Enda Stockwell
# ---------------------------------------------------------------
# Description:
# Plot the MTF scores across multiple images on one ROI
#   Using Valeo sample images + data
# ---------------------------------------------------------------

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Load txt file with MTF scores
df = []

# Concatinate all data from files
data1 = pd.read_csv('data\\focal_sweep_data\MTF_ZPos_0.054.txt', sep="\t", header=[2],  on_bad_lines='skip')
data2 = pd.read_csv('data\\focal_sweep_data\MTF_ZPos_0.062.txt', sep="\t", header=[2],  on_bad_lines='skip')
data3 = pd.read_csv('data\\focal_sweep_data\MTF_ZPos_0.07.txt', sep="\t", header=[2],  on_bad_lines='skip')
data4 = pd.read_csv('data\\focal_sweep_data\MTF_ZPos_0.078.txt', sep="\t", header=[2],  on_bad_lines='skip')
data5 = pd.read_csv('data\\focal_sweep_data\MTF_ZPos_0.086.txt', sep="\t", header=[2],  on_bad_lines='skip')
data6 = pd.read_csv('data\\focal_sweep_data\MTF_ZPos_0.094.txt', sep="\t", header=[2],  on_bad_lines='skip')
data7 = pd.read_csv('data\\focal_sweep_data\MTF_ZPos_0.102.txt', sep="\t", header=[2],  on_bad_lines='skip')
data8 = pd.read_csv('data\\focal_sweep_data\MTF_ZPos_0.11.txt', sep="\t", header=[2],  on_bad_lines='skip')
data9 = pd.read_csv('data\\focal_sweep_data\MTF_ZPos_0.118.txt', sep="\t", header=[2],  on_bad_lines='skip')
data10 = pd.read_csv('data\\focal_sweep_data\MTF_ZPos_0.126.txt', sep="\t", header=[2],  on_bad_lines='skip')
data11 = pd.read_csv('data\\focal_sweep_data\MTF_ZPos_0.134.txt', sep="\t", header=[2],  on_bad_lines='skip')
data12 = pd.read_csv('data\\focal_sweep_data\MTF_ZPos_0.142.txt', sep="\t", header=[2],  on_bad_lines='skip')
data13 = pd.read_csv('data\\focal_sweep_data\MTF_ZPos_0.15.txt', sep="\t", header=[2],  on_bad_lines='skip')
data14 = pd.read_csv('data\\focal_sweep_data\MTF_ZPos_0.158.txt', sep="\t", header=[2],  on_bad_lines='skip')
data15 = pd.read_csv('data\\focal_sweep_data\MTF_ZPos_0.166.txt', sep="\t", header=[2],  on_bad_lines='skip')
data16 = pd.read_csv('data\\focal_sweep_data\MTF_ZPos_0.174.txt', sep="\t", header=[2],  on_bad_lines='skip')
data17 = pd.read_csv('data\\focal_sweep_data\MTF_ZPos_0.182.txt', sep="\t", header=[2],  on_bad_lines='skip')
data18 = pd.read_csv('data\\focal_sweep_data\MTF_ZPos_0.19.txt', sep="\t", header=[2],  on_bad_lines='skip')
df.append(data1)
df.append(data2)
df.append(data3)
df.append(data4)
df.append(data5)
df.append(data6)
df.append(data7)
df.append(data8)
df.append(data9)
df.append(data10)
df.append(data11)
df.append(data12)
df.append(data13)
df.append(data14)
df.append(data15)
df.append(data16)
df.append(data17)
df.append(data18)

# Arrays for holding spatial frequencies & MTF scores
spatial_frequency = []
MTF = []

# Loop through each column, specifying Region Of Interest (ROI) as "MTF.X"
# Appends MTF list with 1 region accross all focal distances
for x in range (0, 18):
    MTF.append(df[x].loc[:, ["MTF.1"]])

y_axis = []

# Loop through all focal distances, appending each with 1 specific mtf score / Spatial frequency step? per focal distance
for x in range(0, 18):
    y_axis.append(MTF[x][1:2])

# Convert y_axis to numpy array, and flatten
y_axis = np.array(y_axis)
y_axis = y_axis.flatten()

print(y_axis)

# Linearly space x_axis for each focal distance
x_axis = np.linspace(0, 18, 18)

# Plot data
plt.figure(0)
plt.plot(x_axis, y_axis)
plt.xlabel("Focal Distance (normalized)")
plt.ylabel("MTF score")
plt.title("Middle Centre - LEFT region")
plt.show(block=True)