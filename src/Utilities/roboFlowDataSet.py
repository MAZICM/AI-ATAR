from roboflow import Roboflow
import time
from src.Utilities import log
import os


def roboflow_dataset():
    start_time = time.time()

    # rbflw_api_key = "BtcWZsdvGqmTQ2Sfmgbj"
    # rbflw_workspace = "moho-rahimi-xyr0w"
    # rbflw_project = "wildfire-ryisc"
    # rbflw_download = "yolov8"

    rbflw_api_key = input("Enter your API_key : ")
    rbflw_workspace = input("Enter your workspace : ")
    rbflw_project = input("Enter your project : ")
    rbflw_version = input("Enter your version :")
    rbflw_download = input("Enter your Download :")


    #!pip install roboflow
    #
    # from roboflow import Roboflow
    # rf = Roboflow(api_key="1F10ZNdjV7NFepJ29yoE")
    # project = rf.workspace("vishwaketu-malakar-o9d0b").project("fire-detection-7oyym")
    # dataset = project.version(6).download("yolov8")
    #

    try:
        # Code that might raise an exception
        log.logger.info("\nDOWNLOAD START")
        print("\n")
        os.chdir("src/datasets/")
        start_time = time.time()

        rf = Roboflow(api_key=rbflw_api_key)
        project = rf.workspace(rbflw_workspace).project(rbflw_project)
        dataset = project.version(rbflw_version).download(rbflw_download)

    except Exception as e:
        # Code to handle other exceptions
        end_time = time.time()
        log.logger.error(f"\nAn error occurred: {e}\nExecution time: %.2f seconds", end_time - start_time)
    else:
        # Code to run if no exception occurred
        end_time = time.time()
        log.logger.info("\nNo errors occurred DONE SUCESS\nExecution time: %.2f seconds", end_time - start_time)
    finally:
        # Code that will run regardless of whether an exception occurred
        log.logger.warning("\nDatasetDownload EXIT\n")
        os.chdir("../../")


