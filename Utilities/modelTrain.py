from ultralytics import YOLO


def m_train():

    # model_path="/home/kenaro/ForestFireDetection/yolov8m.pt"
    # data_path="/home/kenaro/ForestFireDetection/AI-Yolo/Wildfire-2/data.yaml"
    # epochs=4
    # imgsz=240
    # device=0
    # workers=8
    # project="labelModel"
    # name="train1"

    model_path = input("Model Path :")
    data_path = input("Data Path :")
    epochs = input("Epochs :")
    imgsz = input("imgsz :")
    device = input("device :")
    workers = input("workers :")
    project = input("project :")
    name = input("name :")

    model = YOLO(model_path)

    model.train(data=data_path, epochs=epochs, imgsz=imgsz, device=device,
                workers=workers, project=project, name=name, show_labels=True)
