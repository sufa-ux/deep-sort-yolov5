from yolov5.utils.general import check_img_size
from utils_ds.draw import draw_boxes
import pandas as pd
import argparse
import numpy as np
import cv2
import torch.backends.cudnn as cudnn
from main import VideoTracker


parser = argparse.ArgumentParser()
# input and output
parser.add_argument('--input_path', type=str, default='./MOT16-03.mp4', help='source')  # file/folder, 0 for webcam
parser.add_argument('--save_path', type=str, default='output/', help='output folder')  # output folder
parser.add_argument("--frame_interval", type=int, default=2)
parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--save_txt', default='output/predict/', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

# camera only
parser.add_argument("--display", action="store_true")
parser.add_argument("--display_width", type=int, default=800)
parser.add_argument("--display_height", type=int, default=600)
parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")

# YOLO-V5 parameters
parser.add_argument('--weights', type=str, default='yolov5/weights/yolov5s.pt', help='model.pt path')
parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
parser.add_argument('--classes', nargs='+', type=int, default=[0], help='filter by class')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', help='augmented inference')

# deepsort parameters
parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")

args = parser.parse_args()
args.img_size = check_img_size(args.img_size)

with VideoTracker(args) as vdo_trk:
    # cap = cv2.VideoCapture(0)
    # while True:
    #     _, im = cap.read()
    #     if im is None:
    #         break
    #     outputs, yt, st = vdo_trk.image_track(im)
    #     print(len(outputs))
    #     if len(outputs) > 0:
    #         num=len(outputs)
    #         bbox_xyxy = outputs[:, :4]
    #         identities = outputs[:, -1]
    #         img=draw_boxes(im,bbox_xyxy,identities)
    #         cv2.imshow('kk',img)
    #         key=cv2.waitKey(1)
    #     else:
    #         cv2.imshow('kk',im)
    #         key=cv2.waitKey(1)
    vdo_trk.test()