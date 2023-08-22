import cv2
from ultralytics import YOLO
import os
import time
from src.Utilities import log
from src.Utilities import flexMenu

def video_detect():
    f = os.listdir(os.getcwd() + "/runs/Videos/localSamples/")
    filename = flexMenu.display_options(f)
    f = os.listdir(os.getcwd()+"/Train")
    m = flexMenu.display_options(f)
    model_path = os.getcwd() + "/Train/"+m+"/weights/"
    f = os.listdir(model_path)
    mt = flexMenu.display_options(f)
    model_path = model_path + mt
    threshold = input("enter threshold :")


    video_dir = os.path.join(os.getcwd(),'runs','Videos')

    log.logger.info("\nDetection START")
    start_time = time.time()
    try:
        # Code that might raise an exception
        video_path = os.path.join(video_dir+"/localSamples", filename)
        mt = mt.rsplit(".", 1)[0]
        filename = filename.rsplit(".", 1)[0]
        filename = filename+"-"+m+"-"+mt+"-"+threshold
        video_path_out = os.path.join(video_dir+"/Scans", filename)
        video_path_out = '{}_out.mp4'.format(video_path_out)
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
        end_time = time.time()
        log.logger.error(f"\nAn error occurred: {e}\nExecution time: %.2f seconds", end_time - start_time)
    else:
        # Code to run if no exception occurred
        # Code to run if no exception occurred
        end_time = time.time()
        log.logger.info("\nNo errors occurred DONE SUCESS\nExecution time: %.2f seconds", end_time - start_time)
    finally:
        # Code that will run regardless of whether an exception occurred
        log.logger.warning("\nDetection EXIT\n")
