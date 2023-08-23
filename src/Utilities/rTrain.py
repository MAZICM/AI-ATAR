from ultralytics import YOLO
import time
from src.Utilities import log
import os
from src.Utilities import flexMenu


def r_train():
    start_time = time.time()
    try:

        model_path = os.getcwd() + "/Train/"
        f = os.listdir("Train/")
        model_name = flexMenu.display_options(f)
        model_path = model_path + model_name + "/weights/last.pt"
        start_time = time.time()
        model = YOLO(model_path)
        log.logger.info("\nTraining START")
        print("\n")
        model.train(resume=True)

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
        log.logger.critical("\nTrainning EXIT")
        print("\n")
