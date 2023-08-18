import cv2
import argparse
from ultralytics import YOLO
import supervision as sv
import subprocess
import os
from Utilities import roboFlowDataSet, vDetect, sDetect, modelValid, modelTrain


def get_dataset():
    roboFlowDataSet.roboflow_dataset()

def video_detect():
    vDetect.video_Detect()

def stream():
    sDetect.stream()


def valid():
    modelValid.m_valid()


def train():
    modelTrain.m_train()

'''
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='YOLOv8 live')
    parser.add_argument("--webcam-resolution", default=[640, 480], nargs=2, type=int)
    args = parser.parse_args()
    return args

def test():
    #VC=0
    VC=input("VideoCapture :")
    window_name="yolov8 Live Stream"
    cap = cv2.VideoCapture(VC)

    while True:
        ret, frame = cap.read()
        cv2.imshow(window_name, frame)
        if (cv2.waitKey(30) == 27):
            break;

'''
