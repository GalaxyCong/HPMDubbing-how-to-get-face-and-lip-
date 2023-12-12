'''
Align faces to generate 68 landmarks and bounding boxes
'''

from __future__ import absolute_import, division, print_function

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import json
import face_alignment
import numpy as np
import glob
from joblib import Parallel, delayed

def parse_args():
    parser = argparse.ArgumentParser(
        description='align faces with `https://github.com/1adrianb/face-alignment`')
    parser.add_argument('--images-path', type=str,
                        default='/data1/gaoxiang_cong/TalkNet-ASD-main/Chem_face/SUkBTHfKqf4-face-SUkBTHfKqf4-015/pyframes', help='the input frames path')
    parser.add_argument('--landmarks-path', type=str,
                        default='./test/landmarks', help='the output 68 landmarks path')
    parser.add_argument('--boxes-path', type=str,
                        default='./test/boxes', help='the output bounding boxes path')

    parser.add_argument('--device', type=str, default='cuda:0',
                        help='cpu or gpu cuda device')
    parser.add_argument('--log-path', type=str, default='./test_V2C/logs',
                        help='logging when there are no faces detected')
    args = parser.parse_args()
    return args


def read(i):
    args = parse_args()
    nname = i.split("/")[-1]
    args.landmarks_path = os.path.join('./test_V2C/landmarks', nname)
    args.boxes_path = os.path.join('./test_V2C/boxes', nname)
    args.images_path = os.path.join(i, "pyframes")
    os.makedirs(args.landmarks_path, exist_ok=True)
    os.makedirs(args.boxes_path, exist_ok=True)
    fan = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._2D, device=args.device, flip_input=False)
    preds = fan.get_landmarks_from_directory(
        args.images_path, return_bboxes=True)
    if len(glob.glob(os.path.join(i, 'pyframes/*'))) == len(preds.items()):
        for image_file, (landmark, _, box) in preds.items():
            if not box:
                os.makedirs(args.log_path, exist_ok=True)
                with open(os.path.join(args.log_path, 'log.txt'), 'a') as logger:
                    logger.write(os.path.abspath(image_file) + '\n')
                continue
            landmark = np.array(landmark)[0]
            box = np.array(box)[0, :4]
            npy_file_name = os.path.splitext(
                os.path.basename(image_file))[0] + '.npy'
            image_landmark_path = os.path.join(args.landmarks_path, npy_file_name)
            image_box_path = os.path.join(args.boxes_path, npy_file_name)
            np.save(image_landmark_path, landmark)
            np.save(image_box_path, box)

def main():
    all_path = "/data1/gaoxiang_cong/TalkNet-ASD-main/face_all/all/face_all/*"
    path_list = glob.glob(all_path)
    Parallel(n_jobs=20, verbose=1)(
        delayed(read)(wav) for wav in path_list
    )

if __name__ == '__main__':
    main()
