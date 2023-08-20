from roboflow import Roboflow
import time
from Utilities import log

def roboflow_dataset():

    try:
        # Code that might raise an exception
        rbflw_api_key = input("Enter your API_key: ")
        rbflw_workspace = input("Enter your workspace: ")
        rbflw_project = input("Enter your project: ")
        rbflw_download = input("Enter your Download:")
        log.logger.info("\nDOWNLOAD START")
        start_time = time.time()
        rf = Roboflow(api_key=rbflw_api_key)
        project = rf.workspace(rbflw_workspace).project(rbflw_project)
        project.version(2).download(rbflw_download)
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

