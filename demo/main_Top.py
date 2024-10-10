import time
import sys
import os
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.backends import cudnn
import cv2
from math import sqrt
from ultralytics import YOLO

Object_detection = YOLO('/home/aibig30/PCI-AS/demo/weights/AC_DETECTION.pt')
class_list = ['Adult', 'Child']

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Top view transformation and distance calculation.')
    parser.add_argument('--gpu',
                        dest='gpu_id', help='GPU device id to use [0], set -1 to use CPU',
                        default=0, type=int)
    parser.add_argument('--cam',
                        dest='cam_id', help='Camera device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--snapshot',
                        dest='snapshot', help='Name of model snapshot.',
                        default='/home/aibig30/PCI-AS/demo/weights/cmu.pth',
                        type=str)
    parser.add_argument('--save_viz',
                        dest='save_viz', help='Save images with pose cube.',
                        default=False, type=bool)
    parser.add_argument('--video_name',
                        dest='video_name', help='Name of the video file (without extension).',
                        default='240712_007_C2',
                        type=str)
    args = parser.parse_args()
    return args

def get_perspective_transform_matrix(src_points, dst_points):
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    return matrix

def calculate_distance(p1, p2):
    return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

if __name__ == '__main__':
    args = parse_args()
    cudnn.enabled = True
    gpu = args.gpu_id
    if (gpu < 0):
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:%d' % gpu)

    print('Loading data.')

    video_base_path = '/home/aibig30/PCI-AS/demo/videos/'
    output_base_path = '/home/aibig30/PCI-AS/demo/outputs/'
    
    video_name = args.video_name
    video_path = os.path.join(video_base_path, video_name + '.mp4')
    output_video_path = os.path.join(output_base_path, video_name + '_output.mp4')
    output_txt_path = os.path.join(output_base_path, video_name + '_output.txt')
    
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    size = (int(width), int(height))
    out = cv2.VideoWriter(output_video_path, fourcc, 15, size)
    
    src_points = np.float32([[847, 291], [1980, 110], [4000, 1443], [2000, 3720]])
    dst_points = np.float32([[0, 0], [3840, 0], [3840, 3840], [0, 3840]])
    matrix = get_perspective_transform_matrix(src_points, dst_points)
    
    with open(output_txt_path, 'w') as results_file:
        results_file.write("frame,distance,adult_speed,child_speed\n")

        previous_adult_center = None
        previous_child_center = None
        previous_frame_time = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            (h, w, c) = frame.shape
            top_view_frame = cv2.warpPerspective(frame, matrix, (3840, 3840))

            adult_center = None
            child_center = None

            object_detections = Object_detection.track(frame)[0]

            for data in object_detections.boxes.data.tolist():
                if len(data) == 7:
                    confidence = float(data[5])
                    if confidence < 0.75:
                        continue

                    x_min, y_min, x_max, y_max = int(data[0]), int(data[1]), int(data[2]), int(data[3])
                    class_id = int(data[6])
                    label = class_list[class_id]
                    
                    if label == 'Adult':
                        adult_center = ((x_min + x_max) / 2, y_max)
                    elif label == 'Child':
                        child_center = ((x_min + x_max) / 2, y_max)

            if adult_center is not None and child_center is not None:
                distance = calculate_distance(adult_center, child_center)
            else:
                distance = None

            adult_speed = None
            child_speed = None

            if previous_frame_time is not None:
                time_elapsed = frame_time - previous_frame_time

                if previous_adult_center is not None and adult_center is not None:
                    adult_speed = calculate_distance(previous_adult_center, adult_center) / time_elapsed

                if previous_child_center is not None and child_center is not None:
                    child_speed = calculate_distance(previous_child_center, child_center) / time_elapsed

            previous_adult_center = adult_center
            previous_child_center = child_center
            previous_frame_time = frame_time
            
            results_file.write(f"{frame_number},{distance},{adult_speed},{child_speed}\n")

            out.write(top_view_frame)

        out.release()
