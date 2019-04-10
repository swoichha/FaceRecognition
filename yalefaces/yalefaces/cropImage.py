
from PIL import Image
import os

for f in os.listdir('.'):
        if f.endswith('jpg'):
                img = Image.open(f)
                fileName, ext = os.path.splitext(f)
                new_filename = fileName + ext
                cropped_image = img.crop((100, 10, 320, 243))
                cropped_image.save(new_filename)

