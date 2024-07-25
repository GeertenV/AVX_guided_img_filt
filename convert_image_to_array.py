import os
import sys
import numpy as np
from PIL import Image

np.set_printoptions(threshold=sys.maxsize)

img = Image.open('imagedata/cave-noflash.bmp')
arr = np.array(img)
str_r = np.array_str(arr[:,:,0].flatten(),max_line_width=10000000).replace('[', '').replace(']', '').replace('  ', ' ').replace('  ', ' ')
str_g = np.array_str(arr[:,:,1].flatten(),max_line_width=10000000).replace('[', '').replace(']', '').replace('  ', ' ').replace('  ', ' ')
str_b = np.array_str(arr[:,:,2].flatten(),max_line_width=10000000).replace('[', '').replace(']', '').replace('  ', ' ').replace('  ', ' ')

with open('imagedata/cave-noflash_red.txt', 'a') as f:
    f.write(str_r)
    
with open('imagedata/cave-noflash_green.txt', 'a') as f:
    f.write(str_g)
    
with open('imagedata/cave-noflash_blue.txt', 'a') as f:
    f.write(str_b)

img = Image.open('imagedata/cave-flash.bmp')
arr = np.array(img)
str_r = np.array_str(arr[:,:,0].flatten(),max_line_width=10000000).replace('[', '').replace(']', '').replace('  ', ' ').replace('  ', ' ')
str_g = np.array_str(arr[:,:,1].flatten(),max_line_width=10000000).replace('[', '').replace(']', '').replace('  ', ' ').replace('  ', ' ')
str_b = np.array_str(arr[:,:,2].flatten(),max_line_width=10000000).replace('[', '').replace(']', '').replace('  ', ' ').replace('  ', ' ')

with open('imagedata/cave-flash_red.txt', 'a') as f:
    f.write(str_r)
    
with open('imagedata/cave-flash_green.txt', 'a') as f:
    f.write(str_g)
    
with open('imagedata/cave-flash_blue.txt', 'a') as f:
    f.write(str_b)
