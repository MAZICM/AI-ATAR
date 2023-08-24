from src.Utilities import roboFlowDataSet, vDetect, sDetect, modelValid, modelTrain, rTrain
import sys
import logging
from src.Utilities import log

def resume_train():

    rTrain.r_train()
    action = "\n\t  ======> Do you want to proceed?"
    if get_user_confirmation(action):
        print("")
        log.logger.critical("User confirmed to proceed.")
    else:
        print("")
        log.logger.critical("User chose not to proceed.")
        sys.exit()  # Program will exit here.
def get_dataset():
    roboFlowDataSet.roboflow_dataset()
    action = "\n\t  ======> Do you want to proceed?"
    if get_user_confirmation(action):
        print("")
        log.logger.critical("User confirmed to proceed.")
    else:
        print("")
        log.logger.critical("User chose not to proceed.")
        sys.exit()  # Program will exit here.


def video_detect():
    vDetect.video_detect()

    action = "\n\t  ======> Do you want to proceed?"
    if get_user_confirmation(action):
        print("")
        log.logger.critical("User confirmed to proceed.")
    else:
        print("")
        log.logger.critical("User chose not to proceed.")
        sys.exit()  # Program will exit here.


def stream():
    sDetect.stream()
    action = "\n\t  ======> Do you want to proceed?"
    if get_user_confirmation(action):
        log.logger.critical("User confirmed to proceed.")
    else:
        log.logger.critical("User chose not to proceed.")
        sys.exit()  # Program will exit here.


def valid():
    modelValid.m_valid()
    action = "\n\t  ======> Do you want to proceed?"
    if get_user_confirmation(action):
        print("")
        log.logger.critical("User confirmed to proceed.")
    else:
        print("")
        log.logger.critical("User chose not to proceed.")
        sys.exit()  # Program will exit here.


def train():
    modelTrain.m_train()
    action = "\n\t  ======> Do you want to proceed?"
    if get_user_confirmation(action):
        print("")
        log.logger.critical("User confirmed to proceed.")
    else:
        print("")
        log.logger.critical("User chose not to proceed.")
        sys.exit()  # Program will exit here.


def get_user_confirmation(message):
    while True:
        user_input = input(f"{message} (Y/N): ").strip().lower()
        if user_input == 'y':
            return True
        elif user_input == 'n':
            return False
        else:
            print("")
            log.logger.error("Invalid input. Please enter 'Y' for Yes or 'N' for No.")
            print("")

