import os
import re

import cv2
import numpy as np


def loader(path, dim):
    size = len([name for name in os.listdir(path) if name.endswith(".JPG") or name.endswith(".jpg")])
    pictures = np.zeros((size, dim[0], dim[1], dim[2]))

    for filename in os.listdir(path):
        if filename.endswith(".JPG") or filename.endswith(".jpg"):
            index = re.findall(r'\d+', filename)
            pictures[int(index[0]) - 1, :,:,:] = cv2.imread(path +"/"+filename)
    return pictures
