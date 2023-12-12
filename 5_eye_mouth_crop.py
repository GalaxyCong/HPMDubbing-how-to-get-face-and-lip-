'''
Crop eye (left, right), mouth ROIs based on our cropper
'''
from __future__ import absolute_import, division, print_function
import argparse
import os
import numpy as np
from skimage import io
from tqdm import tqdm
from cropper.eye_cropper import crop_eye_image
from cropper.mouth_cropper import crop_mouth_image
import glob
import json

def parse_args():
    parser = argparse.ArgumentParser(description='crop eye and mouth regions')

    # common arguments
    parser.add_argument('--images-path', type=str, default='/data/conggaoxiang/TalkNet-ASD-main/Chem_face/SUkBTHfKqf4-face-SUkBTHfKqf4-015/pyframes',
                        help='[COMMON] the input frames path')
    parser.add_argument('--landmarks-path', type=str, default='./test/landmarks',
                        help='[COMMON] the input 68 landmarks path')

    # eyes cropping arguments
    parser.add_argument('--boxes-path', type=str, default='./test/boxes',
                        help='[EYE] the input bounding boxes path')
    parser.add_argument('--eye-width', type=int, default=60,
                        help='[EYE] width of cropped eye ROIs')
    parser.add_argument('--eye-height', type=int, default=48,
                        help='[EYE] height of cropped eye ROIs')
    parser.add_argument('--face-roi-width', type=int, default=300,
                        help='[EYE] maximize this argument until there is a warning message')
    parser.add_argument('--face-roi-height', type=int, default=300,
                        help='[EYE] maximize this argument until there is a warning message')
    parser.add_argument('--left-eye-path', type=str, default='./test/left_eye',
                        help='[EYE] the output left eye images path')
    parser.add_argument('--right-eye-path', type=str, default='./test/right_eye',
                        help='[EYE] the output right eye images path')

    # mouth cropping arguments
    parser.add_argument('--mean-face', type=str, default='./cropper/20words_mean_face.npy',
                        help='[MOUTH] mean face pathname')
    parser.add_argument('--mouth-width', type=int, default=96,
                        help='[MOUTH] width of cropped mouth ROIs')
    parser.add_argument('--mouth-height', type=int, default=96,
                        help='[MOUTH] height of cropped mouth ROIs')
    parser.add_argument('--start-idx', type=int, default=48,
                        help='[MOUTH] start of landmark index for mouth')
    parser.add_argument('--stop-idx', type=int, default=68,
                        help='[MOUTH] end of landmark index for mouth')
    parser.add_argument('--window-margin', type=int, default=12,
                        help='[MOUTH] window margin for smoothed_landmarks')

    parser.add_argument('--mouth-path', type=str, default='./test/mouth',
                        help='[MOUTH] the output mouth images path')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # all_path = "/data/conggaoxiang/TalkNet-ASD-main/Chem_face/*"
    all_path = "/data/conggaoxiang/EyeLipCropper-master/test/landmarks/*"
    path_list = glob.glob(all_path)
    uninc_error = []
    for i in path_list:
        i = "/data/conggaoxiang/EyeLipCropper-master/test/landmarks_v1/Inside@Joy-face-Inside@Joy_00_0681_00"
        nname = i.split("/")[-1]
        args.landmarks_path = os.path.join('./test_V2C/landmarks', nname)
        args.boxes_path = os.path.join('./test_V2C/boxes', nname)
        args.images_path = os.path.join("/data/conggaoxiang/TalkNet-ASD-main/face_all/all/face_all", nname, "pyframes")
        args.mouth_path = os.path.join('./test_V2C/mouth', nname)
        print('\033[36mCropping mouth images ...\033[0m')
        os.makedirs(args.mouth_path, exist_ok=True)
        crop_mouth_image(args.images_path,
                         args.landmarks_path,
                         args.mouth_path,
                         np.load(args.mean_face),
                         crop_width=args.mouth_width,
                         crop_height=args.mouth_height,
                         start_idx=args.start_idx,
                         stop_idx=args.stop_idx,
                         window_margin=args.window_margin)
    json_str = json.dumps(uninc_error)
    with open('/data/conggaoxiang/EyeLipCropper-master/test/uninc.json', 'w') as json_file:
        json_file.write(json_str)

if __name__ == '__main__':
    main()
