import cv2
from ultralytics import YOLO
import os


def video_detect():

    
    try:
        # Code that might raise an exception
        filename = input("enter file name :")
        model_path = input("enter model path :")
        threshold = input("enter threshold :")
        video_dir = os.path.join('VideoTest')

        video_path = os.path.join(video_dir, filename)
        video_path_out = '{}_out.mp4'.format(video_path)
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        h, w, _ = frame.shape
        out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (w, h))
        # model_path = os.path.join('.', 'test', 'train1', 'weights', 'best.pt')
        # Load a model
        model = YOLO(model_path)  # load a custom model
        # threshold = 0.7

        while ret:
            results = model(frame)[0]
            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result
                if score > float(threshold):
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                    cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            out.write(frame)
            ret, frame = cap.read()
        cap.release()
        out.release()
        cv2.destroyAllWindows()
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
