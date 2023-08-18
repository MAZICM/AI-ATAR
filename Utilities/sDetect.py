
import subprocess


def stream():
    # task1 = "detect"
    # mode1 = "predict"
    # model1 = "test/train1/weights/best.pt"
    # source1 = "0"
    # device1 = "0"
    # name1 = "yolov8_StreamTest"

    task1 = input("task :")
    mode1 = input("mode :")
    model1 = input("model :")
    source1 = input("source :")
    device1 = input("device :")
    name1 = input("name :")
    live_stream = [
        "yolo",
        "task=", task1,
        "mode=", mode1,
        "model=", model1,
        "source=", source1,
        "device=", device1,
        "name=", name1,
        "show=", "True",
    ]
    subprocess.run(live_stream)