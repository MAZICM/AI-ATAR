from ultralytics import YOLO


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
        model = YOLO(model_path)
        model.train(data=data_path, epochs=epochs, imgsz=imgsz, device=device,
                    workers=workers, project=project, name=name, show_labels=True)
    except Exception as e:
        # Code to handle other exceptions
        print(f"An error occurred: {e}")
    else:
        # Code to run if no exception occurred
        print("No errors occurred DONE SUCESS")
    finally:
        # Code that will run regardless of whether an exception occurred
        print("ExitProcess")
    # Code continues here
    print("Press Key To Continue with the rest of the program")

