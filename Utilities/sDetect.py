# import subprocess
from ultralytics import YOLO
import time
from Utilities import log
import cv2
import argparse
import supervision as sv






def stream():
    #mod = input("model :")
    name = input("name :")
    #source = int(input("source :"))
    #model = YOLO(mod)
    log.logger.info("\nSTREAM START")
    start_time = time.time()

    def parse_arguments() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="YOLOv8 live")
        parser.add_argument("--webcam-resolution", default=[1280, 720], nargs=2, type=int)  # 480, 640, 3
        args = parser.parse_args()
        return args
    try:
        args = parse_arguments()
        frame_width, frame_height = args.webcam_resolution

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_height)
        model = YOLO("/home/kenaro/ForestFireDetection/AI-Yolo/Train/train1_v8n100-240/weights/best.pt")
        box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=2, text_scale=1)

        while True:
            ret, frame = cap.read()
            results = model(frame)
            detections = sv.Detections.from_yolov8(results[0])
            print(detections)
            frame = box_annotator.annotate(scene=frame, detections=detections)
            cv2.imshow("yolov8", frame)

            # print(frame.shape)

            if cv2.waitKey(30) == 27:

                break

        # Code that might raise an exception
        '''
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
        model = YOLO("/home/kenaro/ForestFireDetection/AI-Yolo/Train/train1_v8n100-240/weights/best.pt")

       '''

    except Exception as e:
        # Code to handle other exceptions
        print(e)
        end_time = time.time()
        log.logger.error(f"\nAn error occurred: {e}\nExecution time: %.2f seconds", end_time - start_time)
    else:
        # Code to run if no exception occurred
        end_time = time.time()
        log.logger.info("\nNo errors occurred DONE SUCESS\nExecution time: %.2f seconds", end_time - start_time)
    finally:
        # Code that will run regardless of whether an exception occurred
        log.logger.warning("\nSTREAM EXIT")
        print("\n")


