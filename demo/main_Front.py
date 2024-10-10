import time
import sys
import os
import argparse
from pathlib import Path
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())
import numpy as np
import torch
import torch.nn as nn
from torch.backends import cudnn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from PIL import Image
from model import RepNet6D
import util
import cv2
from math import cos, sin
from ultralytics import YOLO

Head_detection = YOLO('./weights/HEAD_DETECTION.pt')
Object_detection = YOLO('./weights/AC_DETECTION.pt')
class_list = ['Adult', 'Child']

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Head pose estimation using the 6DRepNet.')
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
                        default='240712_007_C7',
                        type=str)
    args = parser.parse_args()
    return args

transformations = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

if __name__ == '__main__':
    args = parse_args()
    cudnn.enabled = True
    gpu = args.gpu_id
    if (gpu < 0):
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:%d' % gpu)
    snapshot_path = args.snapshot
    model = RepNet6D(backbone_name='RepVGG-B1g4',
                     backbone_file='',
                     deploy=True,
                     pretrained=False)

    print('Loading data.')
    print('model', model)

    def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size=400):
        pitch = pitch * np.pi / 180
        yaw = -(yaw * np.pi / 180)
        roll = roll * np.pi / 180

        if tdx is not None and tdy is not None:
            tdx = tdx
            tdy = tdy
        else:
            height, width = img.shape[:2]
            tdx = width / 2
            tdy = height / 2

        x1 = size * (cos(yaw) * cos(roll)) + tdx
        y1 = size * (cos(pitch) * sin(roll) + cos(roll)
                     * sin(pitch) * sin(yaw)) + tdy

        x2 = size * (-cos(yaw) * sin(roll)) + tdx
        y2 = size * (cos(pitch) * cos(roll) - sin(pitch)
                     * sin(yaw) * sin(roll)) + tdy

        x3 = size * (sin(yaw)) + tdx
        y3 = size * (-cos(yaw) * sin(pitch)) + tdy

        cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 2)
        cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)

        direction_angle = np.arctan2(y3 - tdy, x3 - tdx) * 180 / np.pi
        return img, direction_angle, x3, y3

    saved_state_dict = torch.load(os.path.join(
        snapshot_path), map_location=None if torch.cuda.is_available() else 'cpu')

    if 'model_state_dict' in saved_state_dict:
        model.load_state_dict(saved_state_dict['model_state_dict'])
    else:
        model.load_state_dict(saved_state_dict)
    model.to(device)
    model.eval()

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
    WHITE = (255, 255, 255)
    
    with open(output_txt_path, 'w') as results_file:
        results_file.write("frame,adult_pitch,adult_yaw,adult_roll,child_pitch,child_yaw,child_roll,adult_angle,child_angle,adult_tdx,adult_tdy,child_tdx,child_tdy,adult_x3,adult_y3,child_x3,child_y3\n")

        with torch.no_grad():
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                (h, w, c) = frame.shape

                adult_pitch, adult_yaw, adult_roll = None, None, None
                child_pitch, child_yaw, child_roll = None, None, None
                adult_angle, child_angle = None, None
                adult_tdx, adult_tdy, child_tdx, child_tdy = None, None, None, None
                adult_x3, adult_y3, child_x3, child_y3 = None, None, None, None

                object_detections = Object_detection.track(frame)[0]

                for data in object_detections.boxes.data.tolist():
                    if len(data) == 7:
                        confidence = float(data[5])
                        if confidence < 0.75:
                            continue

                        x_min, y_min, x_max, y_max = int(data[0]), int(
                            data[1]), int(data[2]), int(data[3])
                        class_id = int(data[6])
                        label = class_list[class_id]

                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 1)
                        cv2.putText(frame, f'{label}: {round(confidence, 2)}%', (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    
                        obj_frame = frame[y_min:y_max, x_min:x_max]
                        head_detections = Head_detection(obj_frame)[0]

                        for head_data in head_detections.boxes.data.tolist():
                            head_confidence = float(head_data[4])
                            if head_confidence < 0.75:
                                continue

                            head_x_min, head_y_min, head_x_max, head_y_max = int(head_data[0]), int(
                                head_data[1]), int(head_data[2]), int(head_data[3])

                            cv2.rectangle(frame, (x_min + head_x_min, y_min + head_y_min), (x_min + head_x_max, y_min + head_y_max), WHITE, 1)
                            cv2.putText(frame, str(round(head_confidence, 2)) + '%', (x_min + head_x_min, y_min + head_y_min), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

                            bbox_width = abs(head_x_max - head_x_min)
                            bbox_height = abs(head_y_max - head_y_min)

                            head_x_min = max(0, head_x_min-int(0.2*bbox_height))
                            head_y_min = max(0, head_y_min-int(0.2*bbox_width))
                            head_x_max = head_x_max+int(0.2*bbox_height)
                            head_y_max = head_y_max+int(0.2*bbox_width)

                            img = obj_frame[head_y_min:head_y_max, head_x_min:head_x_max]
                            img = Image.fromarray(img)
                            img = img.convert('RGB')
                            img = transformations(img)
                            img = torch.Tensor(img[None, :]).to(device)

                            start = time.time()
                            R_pred = model(img)
                            end = time.time()
                            print('Head pose estimation: %2f ms' % ((end - start)*1000.))
                            euler = util.compute_euler_angles_from_rotation_matrices(
                                R_pred)*180/np.pi
                            p_pred_deg = euler[:, 0].cpu()
                            y_pred_deg = euler[:, 1].cpu()
                            r_pred_deg = euler[:, 2].cpu()

                            tdx = x_min + head_x_min + int(.5 * (head_x_max - head_x_min))
                            tdy = y_min + head_y_min + int(.5 * (head_y_max - head_y_min))

                            frame, direction_angle, x3, y3 = draw_axis(frame, y_pred_deg[0], p_pred_deg[0], r_pred_deg[0], tdx, tdy, size=bbox_width)
                            
                            if label == 'Adult':
                                adult_pitch, adult_yaw, adult_roll = p_pred_deg.item(), y_pred_deg.item(), r_pred_deg.item()
                                adult_angle = direction_angle
                                adult_tdx, adult_tdy = tdx, tdy
                                adult_x3, adult_y3 = x3, y3
                            elif label == 'Child':
                                child_pitch, child_yaw, child_roll = p_pred_deg.item(), y_pred_deg.item(), r_pred_deg.item()
                                child_angle = direction_angle
                                child_tdx, child_tdy = tdx, tdy
                                child_x3, child_y3 = x3, y3

                if adult_angle is not None:
                    cv2.putText(frame, f'Adult Angle: {adult_angle:.2f}', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                if child_angle is not None:
                    cv2.putText(frame, f'Child Angle: {child_angle:.2f}', (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                results_file.write(f"{frame_number},{adult_pitch},{adult_yaw},{adult_roll},{child_pitch},{child_yaw},{child_roll},{adult_angle},{child_angle},{adult_tdx},{adult_tdy},{child_tdx},{child_tdy},{adult_x3},{adult_y3},{child_x3},{child_y3}\n")

                out.write(frame)
        out.release()
