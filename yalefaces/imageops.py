import numpy as np
from PIL import Image

img = Image.open('subject01.centerlight.jpg')
arr = np.array(img) # 640x480x4 array
print(arr)
print (arr.shape, arr.dtype)

print('\n Before converting image to face vector ARRAY of image is: \n',arr)

arr.shape = (233*220, 1)
print('\n Face vector \n',arr)

imgMean = np.mean(arr)
#mean of array values of an Image
print(imgMean)

# arr = np.reshape(1,5)
# print("reshaping the array", arr)

# print("\nravel() : ", arr.ravel())
# line 22
# img_array = cv2.imread(os.path.join(path,img))
# img_array.resize(1,233*220)

# line 53

# plt.imshow(resized_avg_faceVector)
# plt.show()