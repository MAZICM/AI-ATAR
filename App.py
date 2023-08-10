










#........
from ultralytics import YOLO
from roboflow import Roboflow
import cv2
def test():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        cv2.imshow("yolov8", frame)

        if (cv2.waitKey(30) == 27):
            break;
def train():
    model = YOLO("/home/kenaro/ForestFireDetection/yolov8m.pt")

    model.train(data="/home/kenaro/ForestFireDetection/AI-Yolo/Wildfire-2/data.yaml", epochs=3, imgsz=240, device=0, workers=4, project="project1", name="name1")

def get_dataset():

    rf = Roboflow(api_key="BtcWZsdvGqmTQ2Sfmgbj")
    project = rf.workspace("moho-rahimi-xyr0w").project("wildfire-ryisc")
    dataset = project.version(2).download("yolov8")


def display_menu():
    print("Welcome to My CLI Menu")
    print("-----------------------")
    print("1. Download default training dataset")
    print("2. Train")
    print("3. Valid")
    print("4. Live Test")
    print("5. Quit")
    print("-----------------------")

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
        print("You selected Option 3")
        return 1
    elif choice == '4':
        print("test..............")
        test()
        return 1
    elif choice == '5':
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
