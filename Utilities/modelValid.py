import subprocess


def m_valid():
    # task2="detect"
    # mode2="val"
    # model2="test/train1/weights/best.pt"
    # data2="/home/kenaro/ForestFireDetection/AI-Yolo/Wildfire-2/data.yaml"
    # name2="yolov8_eval"
    # Define the command as a list of arguments
    task2 = input("task :")
    mode2 = input("mode :")
    model2 = input("model :")
    data2 = input("data :")
    name2 = input("name :")

    validation = [
        "yolo",
        "task=", task2,
        "mode=", mode2,
        "model=", model2,
        "name=", name2,
        "data=", data2
    ]
    subprocess.run(validation)
