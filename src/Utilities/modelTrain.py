from ultralytics import YOLO
import time
from src.Utilities import log
import os
from src.Utilities import flexMenu
def m_train():
    start_time = time.time()
    try:
        # Code that might raise an exception
        # model_path = input("Model Path :") #
        f = ['yolov8n.pt','yolov8s.pt','yolov8m.pt','yolov8l.pt','yolov8x.pt']
        model_path = os.getcwd() + "/src/yolov8DefaultModels/"
        # f = os.listdir(model_path)
        model_name = flexMenu.display_options(f)
        model_path = model_path + model_name
        # data_path = input("Data Path :")   #
        data_path = "src/datasets"
        f = os.listdir(data_path)
        # os.chdir("src/datasets/")
        y = os.getcwd()
        x = flexMenu.display_options(f)
        data_path = y+"/"+data_path + "/" + x + "/data.yaml"
        print("data : "+data_path)


        # data_path = data_path + f +""


        epochs = int(input("Epochs :"))
        imgsz = int(input("imgsz :"))
        device = int(input("device :"))
        workers = int(input("workers :"))
        # project = input("project :")
        project = "Train"
        # name = input("name :")
        if model_name == "yolov8n.pt":
            m = "n"
        elif model_name == "yolov8s.pt":
            m = "s"
        elif model_name == "yolov8m.pt":
            m = "m"
        elif model_name == "yolov8l.pt":
            m = "l"
        elif model_name == "yolov8x.pt":
            m = "x"

        name = "train-e"+str(epochs)+"-i"+str(imgsz)+"-w"+str(workers)+"-"+"v8" + m
        start_time = time.time()
        model = YOLO(model_path)
        log.logger.info("\nTraining START")
        print("\n")
        model.train(data=data_path, epochs=epochs, imgsz=imgsz, device=device,
                    workers=workers, project=project, name=name, show_labels=True,
                    lr0=0.1)

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

