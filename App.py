import cv2
import argparse
from ultralytics import YOLO
import supervision as sv
from roboflow import Roboflow
import subprocess
import os

def videoDetect():
    VIDEOS_DIR = os.path.join('.', 'videos')
    video_path = os.path.join(VIDEOS_DIR, 'alpaca1.mp4')
    video_path_out = '{}_out.mp4'.format(video_path)
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    H, W, _ = frame.shape
    out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))
    model_path = os.path.join('.', 'runs', 'detect', 'train1M240FD2V1', 'weights', 'last.pt')
    # Load a model
    model = YOLO(model_path)  # load a custom model
    threshold = 0.5

    while ret:
        results = model(frame)[0]
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > threshold:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        out.write(frame)
        ret, frame = cap.read()
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def Stream():
    #task1 = "detect"
    #mode1 = "predict"
    #model1 = "test/train1/weights/best.pt"
    #source1 = "0"
    #device1 = "0"
    #name1 = "yolov8_StreamTest"
    task1 = input("task :")
    mode1 = input("mode :")
    model1 = input("model :")
    source1 = input("source :")
    device1 = input("device :")
    name1 = input("name :")
    liveStream = [
        "yolo",
        "task=", task1,
        "mode=", mode1,
        "model=", model1,
        "source=", source1,
        "device=", device1,
        "name=", name1,
        "show=", "True",
    ]
    subprocess.run(liveStream)






def valid():
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


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='YOLOv8 live')
    parser.add_argument("--webcam-resolution", default=[640, 480], nargs=2, type=int)
    args = parser.parse_args()
    return args



def test():
    #VC=0
    VC=input("VideoCapture :")
    window_name="yolov8 Live Stream"
    cap = cv2.VideoCapture(VC)

    while True:
        ret, frame = cap.read()
        cv2.imshow(window_name, frame)
        if (cv2.waitKey(30) == 27):
            break;


def train():
    #model_path="/home/kenaro/ForestFireDetection/yolov8m.pt"
    #data_path="/home/kenaro/ForestFireDetection/AI-Yolo/Wildfire-2/data.yaml"
    #epochs=4
    #imgsz=240
    #device=0
    #workers=8
    #project="labelModel"
    #name="train1"

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


def get_dataset():
    #rbflw_api_key = "BtcWZsdvGqmTQ2Sfmgbj"
    #rbflw_workspace = "moho-rahimi-xyr0w"
    #rbflw_project = "wildfire-ryisc"
    #rbflw_download = "yolov8"

    rbflw_api_key = input("Enter your API_key: ")
    rbflw_workspace = input("Enter your workspace: ")
    rbflw_project = input("Enter your project: ")
    rbflw_download= input("Enter your Download:")
    rf = Roboflow(api_key=rbflw_api_key)
    project = rf.workspace(rbflw_workspace).project(rbflw_project)
    dataset = project.version(2).download(rbflw_download)


def display_menu():
    print("\t-----------------------")
    print("\tWelcome to My CLI Menu")
    print("\t-----------------------\n")
    print("\t\t1. Download RoboFlow straining dataset")
    print("\t\t2. Train")
    print("\t\t3. Valid")
    print("\t\t4. Live Test")
    print("\t\t5. test on an existing file")
    print("\t\t6. Quit")
    print("\n\t----------------------------------------------------------")
    print("\tTo exit the CLI menu, choose option '6' or press 'Ctrl+C'.")
    print("\t------------------------------------------------------------\n")

def get_choice():
    choice = input("Enter your choice: ")
    return choice



def get_resp(choice):
    if choice == '1':
        print("Downloading DataSet ...............................")
        get_dataset()
        return 1
    elif choice == '2':
        print("Training started ...............................")
        train()
        return 1
    elif choice == '3':
        print("Validation ..............................")
        valid()
        return 1
    elif choice == '4':
        print("Stream..............")
        Stream()
        return 1
    elif choice == '5':
        print("V..............")
        videoDetect()
        return 1

    elif choice == '6':
        print("Goodbye!")
        return 0

    else:
        print("Invalid choice. Please select a valid option.")


def main():
    x = 1
    while x != 0:
        print("\n")
        display_menu()
        choice = get_choice()
        x = get_resp(choice)


if __name__ == "__main__":
    main()


"""
def live():
    VC=0
    model_path="/home/kenaro/ForestFireDetection/AI-Yolo/labelModel/train1/weights/best.pt"
    window_name=("yolov8 live Stream")
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution
    cap = cv2.VideoCapture(VC)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    model = YOLO(model_path)
    box_annotator = sv.BoxAnnotator(
        thickness=1,
        text_thickness=2,
        text_scale=1
    )
    while True:
        ret, frame = cap.read()
        results = model(frame)[0]
        detections = sv.Detections.from_yolov8(results)

        frame = box_annotator.annotate(scene=frame, detections=detections)
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            break
"""