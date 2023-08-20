from ultralytics import YOLO
import time
from Utilities import log


def m_train():

    try:
        # Code that might raise an exception
        model_path = input("Model Path :")
        data_path = input("Data Path :")
        epochs = int(input("Epochs :"))
        imgsz = int(input("imgsz :"))
        device = int(input("device :"))
        workers = int(input("workers :"))
        project = input("project :")
        name = input("name :")
        start_time = time.time()
        model = YOLO(model_path)
        log.logger.info("\nTraining START\n")
        model.train(data=data_path, epochs=epochs, imgsz=imgsz, device=device,
                    workers=workers, project=project, name=name, show_labels=True)
    except Exception as e:
        # Code to handle other exceptions
        print("\n")
        log.logger.error(f"An error occurred: {e}")
    else:
        # Code to run if no exception occurred
        print("\n")
        end_time = time.time()
        log.logger.info("No errors occurred DONE SUCESS\nExecution time: %.2f seconds", end_time - start_time)

    finally:
        # Code that will run regardless of whether an exception occurred
        print("\n")
        log.logger.warning("Trainning EXIT")
        print("\n")

