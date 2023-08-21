import time
from ultralytics import YOLO
from Utilities import log


def m_valid():
    mod = input("model :")
    name2 = input("name :")
    log.logger.info("\nValidation START")
    start_time = time.time()
    try:
        model=YOLO(mod)
        results = model.val(name=name2)

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
        log.logger.warning("\nValidation EXIT\n")
