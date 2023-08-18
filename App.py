from Utilities import features


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
        features.get_dataset()
        return 1
    elif choice == '2':
        print("Training started ...............................")
        features.train()
        return 1
    elif choice == '3':
        print("Validation ..............................")
        features.valid()
        return 1
    elif choice == '4':
        print("Stream..............")
        features.stream()
        return 1
    elif choice == '5':
        print("V..............")
        features.video_detect()
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