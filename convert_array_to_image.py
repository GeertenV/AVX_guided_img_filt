import numpy as np
from numpy import genfromtxt
from PIL import Image



red_vals = genfromtxt("imagedata/red_output.txt",dtype="int",delimiter=" ", unpack=False)
red_vals = np.reshape(red_vals, (-1, 512))
green_vals = genfromtxt("imagedata/green_output.txt",dtype="int",delimiter=" ", unpack=False)
green_vals = np.reshape(green_vals, (-1, 512))
blue_vals = genfromtxt("imagedata/blue_output.txt",dtype="int",delimiter=" ", unpack=False)
blue_vals = np.reshape(blue_vals, (-1, 512))

width=512
height=512
data = np.arange(width * height, dtype=np.int64).reshape((height, width))
img_data = np.empty((height, width, 3), dtype=np.uint8)
img_data[:, :, 0] = red_vals
img_data[:, :, 1] = green_vals
img_data[:, :, 2] = blue_vals
image = Image.fromarray(img_data)

image.save("image.bmp")
