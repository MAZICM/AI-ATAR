from ultralytics import YOLO


def m_train():

    # model_path="/home/kenaro/ForestFireDetection/yolov8n.pt"
    # data_path="/home/kenaro/ForestFireDetection/AI-Yolo/Wildfire-2/data.yaml"
    # epochs=100
    # imgsz=240
    # device=0
    # workers=8
    # project="Train"
    # name="train1_v8n100-240"

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
