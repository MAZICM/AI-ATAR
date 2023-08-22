from src.Utilities import roboFlowDataSet, vDetect, sDetect, modelValid, modelTrain
import sys

def get_dataset():
    roboFlowDataSet.roboflow_dataset()
    action = "Do you want to proceed?"
    if get_user_confirmation(action):
        print("User confirmed to proceed.")
    else:
        print("User chose not to proceed.")
        sys.exit()  # Program will exit here.


def video_detect():
    vDetect.video_detect()

    action = "Do you want to proceed?"
    if get_user_confirmation(action):
        print("User confirmed to proceed.")
    else:
        print("User chose not to proceed.")
        sys.exit()  # Program will exit here.


def stream():
    sDetect.stream()
    action = "Do you want to proceed?"
    if get_user_confirmation(action):
        print("User confirmed to proceed.")
    else:
        print("User chose not to proceed.")
        sys.exit()  # Program will exit here.


def valid():
    modelValid.m_valid()
    action = "Do you want to proceed?"
    if get_user_confirmation(action):
        print("User confirmed to proceed.")
    else:
        print("User chose not to proceed.")
        sys.exit()  # Program will exit here.


def train():
    modelTrain.m_train()
    action = "Do you want to proceed?"
    if get_user_confirmation(action):
        print("User confirmed to proceed.")
    else:
        print("User chose not to proceed.")
        sys.exit()  # Program will exit here.


def get_user_confirmation(message):
    while True:
        user_input = input(f"{message} (Y/N): ").strip().lower()
        if user_input == 'y':
            return True
        elif user_input == 'n':
            return False
        else:
            print("Invalid input. Please enter 'Y' for Yes or 'N' for No.")

