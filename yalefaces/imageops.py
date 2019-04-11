import numpy as np
from PIL import Image

img = Image.open('subject01.centerlight.jpg')
arr = np.array(img) # 640x480x4 array
print (arr.shape, arr.dtype)
print(arr)
