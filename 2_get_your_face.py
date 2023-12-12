"""
# Date: 2022/9/04
# Log: 1) Input .avi file, which processed by "demo_v2c.py"
#         path  "/data1/gaoxiang_cong/TalkNet-ASD-main/demo/Ralph@Ralph_00_0014_00/pyframes/0000001.jpg"
#     2) Use face-model, to get bbox[], and save the .jpg of the cropped image
#     3) Save, we save the output image in "outface" fold
#     path "/data1/gaoxiang_cong/TalkNet-ASD-main/outface/Ralph@Ralph_00_0014_00/0000001-face{1}.jpg"
"""

import sys, time, os, tqdm, torch, argparse, glob, subprocess, warnings, cv2, pickle, numpy, pdb, math, \
    python_speech_features
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from scipy import signal
from shutil import rmtree
from joblib import Parallel, delayed
from scipy.io import wavfile
from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score, f1_score

from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

from model.faceDetector.s3fd import S3FD
from talkNet import talkNet
from PIL import Image
import json

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description="TalkNet Demo or Columnbia ASD Evaluation")
parser.add_argument('--videoName', type=str, default="001", help='Demo video name')
parser.add_argument('--videoFolder', type=str, default="demo", help='Path for inputs, tmps and outputs')
parser.add_argument('--pretrainModel', type=str, default="pretrain_TalkSet.model",
                    help='Path for the pretrained TalkNet model')
parser.add_argument('--nDataLoaderThread', type=int, default=10, help='Number of workers')
parser.add_argument('--facedetScale', type=float, default=0.25,
                    help='Scale factor for face detection, the frames will be scale to 0.25 orig')
parser.add_argument('--minTrack', type=int, default=10, help='Number of min frames for each shot')
parser.add_argument('--numFailedDet', type=int, default=10,
                    help='Number of missed detections allowed before tracking is stopped')
parser.add_argument('--minFaceSize', type=int, default=1, help='Minimum face size in pixels')
parser.add_argument('--cropScale', type=float, default=0.40, help='Scale bounding box')
parser.add_argument('--start', type=int, default=0, help='The start time of the video')
parser.add_argument('--duration', type=int, default=0,
                    help='The duration of the video, when set as 0, will extract the whole video')
parser.add_argument('--evalCol', dest='evalCol', action='store_true', help='Evaluate on Columnbia dataset')
parser.add_argument('--colSavePath', type=str, default="/data08/col", help='Path for inputs, tmps and outputs')
args = parser.parse_args()
if os.path.isfile(args.pretrainModel) == False:  # Download the pretrained model
    Link = "1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea"
    cmd = "gdown --id %s -O %s" % (Link, args.pretrainModel)
    subprocess.call(cmd, shell=True, stdout=None)
if args.evalCol == True:
    args.videoName = 'col'
    args.videoFolder = args.colSavePath
    args.savePath = os.path.join(args.videoFolder, args.videoName)
    args.videoPath = os.path.join(args.videoFolder, args.videoName + '.mp4')
    args.duration = 0
    if os.path.isfile(args.videoPath) == False:  # Download video
        link = 'https://www.youtube.com/watch?v=6GzxbrO0DHM&t=2s'
        cmd = "youtube-dl -f best -o %s '%s'" % (args.videoPath, link)
        output = subprocess.call(cmd, shell=True, stdout=None)
    if os.path.isdir(args.videoFolder + '/col_labels') == False:  # Download label
        link = "1Tto5JBt6NsEOLFRWzyZEeV6kCCddc6wv"
        cmd = "gdown --id %s -O %s" % (link, args.videoFolder + '/col_labels.tar.gz')
        subprocess.call(cmd, shell=True, stdout=None)
        cmd = "tar -xzvf %s -C %s" % (args.videoFolder + '/col_labels.tar.gz', args.videoFolder)
        subprocess.call(cmd, shell=True, stdout=None)
        os.remove(args.videoFolder + '/col_labels.tar.gz')
else:
    args.savePath = os.path.join(args.videoFolder, args.videoName)
def inference_video(args):
    DET = S3FD(device='cuda')
    flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
    flist.sort()
    no_face_diction = []
    d = 0
    for fidx, fname in enumerate(flist):
        image = cv2.imread(fname)
        imageNumpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = DET.detect_faces(imageNumpy, conf_th=0.9, scales=[args.facedetScale])
        for fffidx, bbox in enumerate(bboxes):
            if bbox is not None:
                face_image = Image.open(os.path.join("/data1/gaoxiang_cong/TalkNet-ASD-main", str(flist[fidx])))
                face_image = face_image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
                if face_image is not None:
                    face_image.save(os.path.join(args.savePath, fname.split("/")[-1]))
                else:
                    print(args.savePath.split('/')[-2])
            else:
                no_face_diction.append(fname.split("Chem_frame/")[-1])
                print("Can't detect face on: ", args.savePath.split('/')[-2])
    if len(no_face_diction)!=0:
        json_str = json.dumps(no_face_diction)
        with open('/data1/gaoxiang_cong/TalkNet-ASD-main/chem_output_face/{}.json'.format(flist[0].split("/")[-2]),
                  'w') as json_file:
            json_file.write(json_str)

def read(i):
    args.pyframesPath = i
    args.savePath = os.path.join('/data1/gaoxiang_cong/TalkNet-ASD-main/Chem_buwan/Chem_face',
                                 i.split("/")[-1], "pyframes")
    if os.path.exists(args.savePath):
        rmtree(args.savePath)
    os.makedirs(args.savePath, exist_ok=True)
    inference_video(args)

def main():
    last_path = "/data1/gaoxiang_cong/TalkNet-ASD-main/Chem_buwan/Chem_frame/*"

    last = glob.glob(last_path)

    Parallel(n_jobs=4, verbose=1)(
        delayed(read)(wav) for wav in last
    )


if __name__ == '__main__':
    main()

