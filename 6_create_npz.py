"""
Aim to convert ".jpg" to "Frame.npz"
"""

import os
import glob
import json
import cv2
import numpy as np
from tqdm import tqdm

jpg_path = "/data1/gaoxiang_cong/EyeLipCropper-master/test/mouth_color/*"
all_image = glob.glob(jpg_path)

for i in all_image:
    name_fold = i.split("/")[-1]
    print('\033[36mCropping mouth file: {} ...\033[0m'.format(name_fold))
    filename1 = os.path.join("/data1/gaoxiang_cong/EyeLipCropper-master/test/File_npz_mouth_color", "{}.npz".format(name_fold))
    image_path = os.path.join(i, "*")
    every_image = glob.glob(image_path)
    every_image.sort()
    sequence = []
    for e_ima in tqdm(every_image):
        a = cv2.imread(e_ima, cv2.IMREAD_GRAYSCALE)
        sequence.append(a)
    np.savez_compressed(filename1, data=np.array(sequence))



