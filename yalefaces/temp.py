import numpy as np 
import cv2 as cv
import matplotlib.pyplot as plt 

img = cv.imread("filename.jppg", 0)

dim = np.shape(img)

rows, cols = dim

total_pixels = rows * cols	

my_arr = np.zeros((10, total_pixels), dtype=np.uint8)


## -------------------subplots-----------
figure, arr = plt.imshow(10, 50)
while (i < img_count):
	arr[0, 0].imshow(img, cmap='gray')