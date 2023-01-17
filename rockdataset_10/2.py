import os
import shutil

with open('ImageSets/Segmentation/test.txt', 'r') as f:
    lines = f.read().splitlines()
    for ii, line in enumerate(lines):
        image = os.path.join('JPEGImages/', line + '.jpg')
        new_path = 'test' + '/' + line + '.jpg'
        shutil.copyfile(image, new_path)
