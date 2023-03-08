# ---------------------------------------------------------------
# Script name : bayer_retrieve.py
# Created by  : Enda Stockwell
# Adapted from : https://pypi.org/project/picamraw/
# ---------------------------------------------------------------
# Description:
# Retrieve images captured in jpeg+raw format
# ---------------------------------------------------------------

from picamraw import PiRawBayer, PiCameraVersion
from matplotlib import pyplot as plt
import numpy as np

raw_bayer = PiRawBayer(
    filepath='/home/enda/Desktop/test.jpeg',  # A JPEG+RAW file, e.g. an image captured using raspistill with the "--raw" flag
    camera_version=PiCameraVersion.V2,
    sensor_mode=0
)
raw_bayer.bayer_array   # A 16-bit 2D numpy array of the bayer data
raw_bayer.bayer_order   # A `BayerOrder` enum that describes the arrangement of the R,G,G,B pixels in the bayer_array
raw_bayer.to_rgb()      # A 16-bit 3D numpy array of bayer data collapsed into RGB channels (see docstring for details).
raw_bayer.to_3d()       # A 16-bit 3D numpy array of bayer data split into RGB channels (see docstring for details).

plt.imshow((raw_bayer.to_rgb() * 255).astype(np.uint8))
print(raw_bayer.bayer_array )
plt.show()