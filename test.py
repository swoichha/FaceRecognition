from PIL import Image
import os

# Crops the image and saves it as "new_filename"
# def crop_image(img, crop_area, new_filename):
#     cropped_image = img.crop(crop_area)
#     cropped_image.save(new_filename)

# The x, y coordinates of the areas to be cropped. (x1, y1, x2, y2)
# crop_area = [(100, 10, 320, 243)]

# path = 'E:\project\Face Recognition'
#
# for i, crop_area in enumerate(crop_areas):
#     filename = os.path.splitext(image_name)[0]
#     ext = os.path.splitext(image_name)[1]
#     new_filename = filename + str(i) + ext

#     crop_image(img, crop_area, new_filename)


for f in os.listdir('.'):
        if f.endswith('jpg'):
                img = Image.open(f)
                fileName, ext = os.path.splitext(f)
                new_filename = fileName + str(f) + ext
                cropped_image = img.crop((100, 10, 320, 243))
                cropped_image.save(new_filename)

#to plot values import matplotlib.pyplot as plt
plt.plot(x,y,'ro')
plt.show()               

# crop_image(i, crop_area, new_filename)
        # print(ext)
        # i.save('cropped/{}.jpg'.format(fn))
        # im.save('subject01.jpg')