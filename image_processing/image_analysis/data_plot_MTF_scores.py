# ---------------------------------------------------------------
# Script name : data_plot_MTF_scores.py
# Created by  : Enda Stockwell
# ---------------------------------------------------------------
# Description:
# Plot the MTF scores across multiple images on multiple ROI's
#   Using Valeo sample images + data
# ---------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_appended = []
data_files = ["0.038", "0.046", "0.054", "0.062", "0.078", "0.086", "0.094", "0.102", "0.11", "0.118", 
              "0.126", "0.134", "0.142", "0.15", "0.158", "0.166", "0.174", "0.182", "0.19", "0.198", 
              "0.206", "0.214", "0.222", "0.23"]

# Append datasets into a list
for i in range(len(data_files)):
    data_appended.append(pd.read_csv("data\\focal_sweep_data\MTF_ZPos_" + data_files[i] +".txt", sep="\t",  on_bad_lines='skip'))

# Concatinate to singular dataframe
df = pd.concat(data_appended, axis=1, keys= data_files)

# Drop all NA columns
df = df.dropna(axis=1)

# Code to view column names
# for col in df.columns:
#     print(col)

# This plot filters each of the same segment into one
MC_Top = df.filter(like = 'MC.Top')
MC_Top = MC_Top.tail(-2)
MC_Left = df.filter(like = 'MC.Left')
MC_Left = MC_Left.tail(-2)
MC_Bottom = df.filter(like = 'MC.Bottom')
MC_Bottom = MC_Bottom.tail(-2)
MC_Right = df.filter(like = 'MC.Right')
MC_Right = MC_Right.tail(-2)
TL_Top = df.filter(like = 'TL.Top')
TL_Top = TL_Top.tail(-2)
TL_Left = df.filter(like = 'TL.Left')
TL_Left = TL_Left.tail(-2)
TL_Bottom = df.filter(like = 'TL.Bottom')
TL_Bottom = TL_Bottom.tail(-2)
TL_Right = df.filter(like = 'TL.Right')
TL_Right = TL_Right.tail(-2)
TR_Top = df.filter(like = 'TR.Top')
TR_Top = TR_Top.tail(-2)
TR_Left = df.filter(like = 'TR.Left')
TR_Left = TR_Left.tail(-2)
TR_Bottom = df.filter(like = 'TR.Bottom')
TR_Bottom = TR_Bottom.tail(-2)
TR_Right = df.filter(like = 'TR.Right')
TR_Right = TR_Right.tail(-2)
BR_Top = df.filter(like = 'BR.Top')
BR_Top = BR_Top.tail(-2)
BR_Left = df.filter(like = 'BR.Left')
BR_Left = BR_Left.tail(-2)
BR_Bottom = df.filter(like = 'BR.Bottom')
BR_Bottom = BR_Bottom.tail(-2)
BR_Right = df.filter(like = 'BR.Right')
BR_Right = BR_Right.tail(-2)
BL_Top = df.filter(like = 'BL.Top')
BL_Top = BL_Top.tail(-2)
BL_Left = df.filter(like = 'BL.Left')
BL_Left = BL_Left.tail(-2)
BL_Bottom = df.filter(like = 'BL.Bottom')
BL_Bottom = BL_Bottom.tail(-2)
BL_Right = df.filter(like = 'BL.Right')
BL_Right = BL_Right.tail(-2)

# This block turns each segment into floating point number
# And sets the x and y axis for plotting

# try changing the ".iloc[2]" to something like ".iloc[16]". why is this possible? 
# Changing the spatial frequency?
y_axis_0 = pd.to_numeric(MC_Top.iloc[2], downcast="float")
x_axis_0 = np.linspace(0, MC_Top.shape[1], MC_Top.shape[1])
y_axis_1 = pd.to_numeric(MC_Left.iloc[2], downcast="float")
x_axis_1 = np.linspace(0, MC_Left.shape[1], MC_Left.shape[1])
y_axis_2 = pd.to_numeric(MC_Bottom.iloc[2], downcast="float")
x_axis_2 = np.linspace(0, MC_Bottom.shape[1], MC_Bottom.shape[1])
y_axis_3 = pd.to_numeric(MC_Right.iloc[2], downcast="float")
x_axis_3 = np.linspace(0, MC_Right.shape[1], MC_Right.shape[1])
y_axis_4 = pd.to_numeric(TL_Top.iloc[2], downcast="float")
x_axis_4 = np.linspace(0, TL_Top.shape[1], TL_Top.shape[1])
y_axis_5 = pd.to_numeric(TL_Left.iloc[2], downcast="float")
x_axis_5 = np.linspace(0, TL_Left.shape[1], TL_Left.shape[1])
y_axis_6 = pd.to_numeric(TL_Bottom.iloc[2], downcast="float")
x_axis_6 = np.linspace(0, TL_Bottom.shape[1], TL_Bottom.shape[1])
y_axis_7 = pd.to_numeric(TL_Right.iloc[2], downcast="float")
x_axis_7 = np.linspace(0, TL_Right.shape[1], TL_Right.shape[1])
y_axis_8 = pd.to_numeric(TR_Top.iloc[2], downcast="float")
x_axis_8 = np.linspace(0, TR_Top.shape[1], TR_Top.shape[1])
y_axis_9 = pd.to_numeric(TR_Left.iloc[2], downcast="float")
x_axis_9 = np.linspace(0, TR_Left.shape[1], TR_Left.shape[1])
y_axis_10 = pd.to_numeric(TR_Bottom.iloc[2], downcast="float")
x_axis_10 = np.linspace(0, TR_Bottom.shape[1], TR_Bottom.shape[1])
y_axis_11 = pd.to_numeric(TR_Right.iloc[2], downcast="float")
x_axis_11 = np.linspace(0, TR_Right.shape[1], TR_Right.shape[1])
y_axis_12 = pd.to_numeric(BR_Top.iloc[2], downcast="float")
x_axis_12 = np.linspace(0, BR_Top.shape[1], BR_Top.shape[1])
y_axis_13 = pd.to_numeric(BR_Left.iloc[2], downcast="float")
x_axis_13 = np.linspace(0, BR_Left.shape[1], BR_Left.shape[1])
y_axis_14 = pd.to_numeric(BR_Bottom.iloc[2], downcast="float")
x_axis_14 = np.linspace(0, BR_Bottom.shape[1], BR_Bottom.shape[1])
y_axis_15 = pd.to_numeric(BR_Right.iloc[2], downcast="float")
x_axis_15 = np.linspace(0, BR_Right.shape[1], BR_Right.shape[1])
y_axis_16 = pd.to_numeric(BL_Top.iloc[2], downcast="float")
x_axis_16 = np.linspace(0, BL_Top.shape[1], BL_Top.shape[1])
y_axis_17 = pd.to_numeric(BL_Left.iloc[2], downcast="float")
x_axis_17 = np.linspace(0, BL_Left.shape[1], BL_Left.shape[1])
y_axis_18 = pd.to_numeric(BL_Bottom.iloc[2], downcast="float")
x_axis_18 = np.linspace(0, BL_Bottom.shape[1], BL_Bottom.shape[1])
y_axis_19 = pd.to_numeric(BL_Right.iloc[2], downcast="float")
x_axis_19 = np.linspace(0, BL_Right.shape[1], BL_Right.shape[1])

# This block plots the data one by one (a loop would be nice but...)
plt.figure(0)
plt.subplot(4, 5, 1)
plt.plot(x_axis_0, y_axis_0)
plt.xlabel("Focal distance (Normalized)")
plt.ylabel("MTF score")
plt.title("MC_Top")

#plt.figure(1)
plt.subplot(4, 5, 2)
plt.plot(x_axis_1, y_axis_1)
plt.xlabel("Focal distance (Normalized)")
plt.ylabel("MTF score")
plt.title("MC_Left")

#plt.figure(2)
plt.subplot(4, 5, 3)
plt.plot(x_axis_2, y_axis_2)
plt.xlabel("Focal distance (Normalized)")
plt.ylabel("MTF score")
plt.title("MC_Bottom")

#plt.figure(3)
plt.subplot(4, 5, 4)
plt.plot(x_axis_3, y_axis_3)
plt.xlabel("Focal distance (Normalized)")
plt.ylabel("MTF score")
plt.title("MC_Right")

#plt.figure(4)
plt.subplot(4, 5, 5)
plt.plot(x_axis_4, y_axis_4)
plt.xlabel("Focal distance (Normalized)")
plt.ylabel("MTF score")
plt.title("TL_Top")

#plt.figure(5)
plt.subplot(4, 5, 6)
plt.plot(x_axis_5, y_axis_5)
plt.xlabel("Focal distance (Normalized)")
plt.ylabel("MTF score")
plt.title("TL_Left")

#plt.figure(6)
plt.subplot(4, 5, 7)
plt.plot(x_axis_6, y_axis_6)
plt.xlabel("Focal distance (Normalized)")
plt.ylabel("MTF score")
plt.title("TL_Bottom")

#plt.figure(7)
plt.subplot(4, 5, 8)
plt.plot(x_axis_7, y_axis_7)
plt.xlabel("Focal distance (Normalized)")
plt.ylabel("MTF score")
plt.title("TL_Right")

#plt.figure(4)
plt.subplot(4, 5, 9)
plt.plot(x_axis_8, y_axis_8)
plt.xlabel("Focal distance (Normalized)")
plt.ylabel("MTF score")
plt.title("TR_Top")

#plt.figure(5)
plt.subplot(4, 5, 10)
plt.plot(x_axis_9, y_axis_9)
plt.xlabel("Focal distance (Normalized)")
plt.ylabel("MTF score")
plt.title("TR_Left")

#plt.figure(6)
plt.subplot(4, 5, 11)
plt.plot(x_axis_10, y_axis_10)
plt.xlabel("Focal distance (Normalized)")
plt.ylabel("MTF score")
plt.title("TR_Bottom")

#plt.figure(7)
plt.subplot(4, 5, 12)
plt.plot(x_axis_11, y_axis_11)
plt.xlabel("Focal distance (Normalized)")
plt.ylabel("MTF score")
plt.title("TR_Right")

#plt.figure(8)
plt.subplot(4, 5, 13)
plt.plot(x_axis_12, y_axis_12)
plt.xlabel("Focal distance (Normalized)")
plt.ylabel("MTF score")
plt.title("BR_Top")

#plt.figure(9)
plt.subplot(4, 5, 14)
plt.plot(x_axis_13, y_axis_13)
plt.xlabel("Focal distance (Normalized)")
plt.ylabel("MTF score")
plt.title("BR_Left")

#plt.figure(10)
plt.subplot(4, 5, 15)
plt.plot(x_axis_14, y_axis_14)
plt.xlabel("Focal distance (Normalized)")
plt.ylabel("MTF score")
plt.title("BR_Bottom")

#plt.figure(11)
plt.subplot(4, 5, 16)
plt.plot(x_axis_15, y_axis_15)
plt.xlabel("Focal distance (Normalized)")
plt.ylabel("MTF score")
plt.title("BR_Right")

#plt.figure(12)
plt.subplot(4, 5, 17)
plt.plot(x_axis_16, y_axis_16)
plt.xlabel("Focal distance (Normalized)")
plt.ylabel("MTF score")
plt.title("BL_Top")

#plt.figure(13)
plt.subplot(4, 5, 18)
plt.plot(x_axis_17, y_axis_17)
plt.xlabel("Focal distance (Normalized)")
plt.ylabel("MTF score")
plt.title("BL_Left")

#plt.figure(14)
plt.subplot(4, 5, 19)
plt.plot(x_axis_18, y_axis_18)
plt.xlabel("Focal distance (Normalized)")
plt.ylabel("MTF score")
plt.title("BL_Bottom")

#plt.figure(15)
plt.subplot(4, 5, 20)
plt.plot(x_axis_19, y_axis_19)
plt.xlabel("Focal distance (Normalized)")
plt.ylabel("MTF score")
plt.title("BL_Right")


# This block plots each curve on a singular 3d plot
ax = plt.figure(2).add_subplot(projection='3d')
ax.plot(x_axis_0, y_axis_0, zs=0, zdir='y', label='MC_Top')
ax.plot(x_axis_1, y_axis_1, zs=1, zdir='y', label='MC_Left')
ax.plot(x_axis_2, y_axis_2, zs=2, zdir='y', label='MC_Bottom')
ax.plot(x_axis_3, y_axis_3, zs=3, zdir='y', label='MC_Right')
ax.plot(x_axis_4, y_axis_4, zs=4, zdir='y', label='TL_Top')
ax.plot(x_axis_5, y_axis_5, zs=5, zdir='y', label='TL_Left')
ax.plot(x_axis_6, y_axis_6, zs=6, zdir='y', label='TL_Bottom')
ax.plot(x_axis_7, y_axis_7, zs=7, zdir='y', label='TL_Right')
ax.plot(x_axis_8, y_axis_8, zs=4, zdir='y', label='TR_Top')
ax.plot(x_axis_9, y_axis_9, zs=5, zdir='y', label='TR_Left')
ax.plot(x_axis_10, y_axis_10, zs=6, zdir='y', label='TR_Bottom')
ax.plot(x_axis_11, y_axis_11, zs=7, zdir='y', label='TR_Right')
ax.plot(x_axis_12, y_axis_12, zs=8, zdir='y', label='BR_Top')
ax.plot(x_axis_13, y_axis_12, zs=9, zdir='y', label='BR_Left')
ax.plot(x_axis_14, y_axis_14, zs=10, zdir='y', label='BR_Bottom')
ax.plot(x_axis_15, y_axis_15, zs=11, zdir='y', label='BR_Right')
ax.plot(x_axis_16, y_axis_16, zs=12, zdir='y', label='BL_Top')
ax.plot(x_axis_17, y_axis_17, zs=13, zdir='y', label='BL_Left')
ax.plot(x_axis_18, y_axis_18, zs=14, zdir='y', label='BL_Bottom')
ax.plot(x_axis_19, y_axis_19, zs=15, zdir='y', label='BL_Right')
plt.xlabel("Focal distance (Normalized)")
plt.ylabel("Region")
ax.set_zlabel('MTF')
plt.title("3D plot of MTF/Region/ZPos")
#plt.legend(loc='best')


plt.show(block=True)
