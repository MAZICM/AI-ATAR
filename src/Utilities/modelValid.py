import time
from ultralytics import YOLO
from src.Utilities import log
from src.Utilities import flexMenu
import os

def m_valid():
    f = os.listdir("./Train/")
    name = flexMenu.display_options(f)
    print("\n\t Validate Best or Last Weights : ")
    f = os.listdir("./Train/"+name+"/weights/")
    filename = flexMenu.display_options(f)
    mod = "./Train/"+name+"/weights/"+filename
    filename = filename.rsplit(".", 1)[0]
    name2 = name+"_eval_" + filename
    log.logger.info("\nValidation START")
    print("")
    start_time = time.time()
    try:
        model = YOLO(mod)
        results = model.val(project="Vaild", name=name2)

    except Exception as e:
        # Code to handle other exceptions
        end_time = time.time()
        log.logger.error(f"\nAn error occurred: {e}\nExecution time: %.2f seconds", end_time - start_time)
    else:
        # Code to run if no exception occurred
        end_time = time.time()
        log.logger.info("\nNo errors occurred DONE SUCCESS\nExecution time: %.2f seconds", end_time - start_time)
    finally:
        # Code that will run regardless of whether an exception occurred
        log.logger.warning("\nValidation EXIT\n")
