# import subprocess
import os
import datetime
from ultralytics import YOLO
import time
from src.Utilities import log
import cv2
import argparse
import supervision as sv
from src.Utilities import flexMenu


def create_video_writer(video_cap, output_filename):
    # grab the width, height, and fps of the frames in the video stream.
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))

    # initialize the FourCC and a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter(output_filename, fourcc, fps,
                             (frame_width, frame_height))

    return writer


def stream():
    current_time = datetime.datetime.now()
    desired_format = "%Y-%m-%d_%H-%M-%S_"
    formatted_time = current_time.strftime(desired_format)
    name = formatted_time

    log.logger.info("\nSTREAM START")
    start_time = time.time()

    def parse_arguments() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="YOLOv8 live")
        parser.add_argument("--webcam-resolution", default=[1920, 1080], nargs=2, type=int)  # 480, 640, 3
        args = parser.parse_args()
        return args

    try:
        # source = int(input("\n\t  ======> source :"))
        # source = "rtsp://www.hessdalen.org:1935/rtplive/_definst_/hessdalen03.stream"
        model_path = "./Train/"
        x = os.listdir(model_path)
        train = flexMenu.display_options(x)
        model_path = model_path + train + "/weights/"
        x = os.listdir(model_path)
        m = flexMenu.display_options(x)
        model_path = model_path + "/" + m
        m = m.rsplit(".", 1)[0]
        model = YOLO(model_path)
        args = parse_arguments()
        frame_width, frame_height = args.webcam_resolution
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_height)
        writer = create_video_writer(cap, "runs/Streams/" + formatted_time + train + m + ".mp4")

        box_annotator = sv.BoxAnnotator(thickness=2)
        os.chdir("./runs/Streams")

        while True:
            ret, frame = cap.read()
            results = model(frame)
            detections = sv.Detections.from_ultralytics(results[0])
            frame = box_annotator.annotate(scene=frame, detections=detections)
            if not detections.class_id.any() == 0:
                print('ALAAAAAAARRM RINGING FIRE !!!!!')
            cv2.imshow("yolov8", frame)
            writer.write(frame)
            # print(frame.shape)

            if cv2.waitKey(30) == 27:
                break
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        os.chdir("../../")



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
        log.logger.critical("\nSTREAM EXIT")
        print("\n")
