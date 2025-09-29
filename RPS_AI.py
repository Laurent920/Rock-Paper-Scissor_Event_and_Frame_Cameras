import os
import time
from datetime import datetime
import cv2
from pypylon import pylon
from roboflow import Roboflow
from ultralytics import YOLO

# -------------------------------
# 1. Download dataset from Roboflow
# -------------------------------
yolo_version = "11"  # or "8"
size = "m" # n, s, m, l, x

rf = Roboflow(api_key="UzPPWQYROl2lfRwP9YkH")
project = rf.workspace("roboflow-58fyf").project("rock-paper-scissors-sxsw")
version = project.version(14)
dataset = version.download("yolov"+yolo_version)

DATA_YAML = os.path.join(dataset.location, "data.yaml")
print("Dataset YAML:", DATA_YAML)

# -------------------------------
# 2. Train YOLOv8
# -------------------------------
def train_yolo():
    # Start from pretrained YOLO weights
    model = YOLO("yolo"+ yolo_version + size + ".pt")

    # Train on Roboflow dataset
    model.train(
        data=DATA_YAML,
        epochs=50,
        imgsz=640,
        batch=16,
        project="runs/rps",
        name=yolo_version + size
    )

# -------------------------------
# 3. Validate & Predict
# -------------------------------
def evaluate_and_predict():
    model = YOLO("runs/rps/"+ yolo_version + size +"/weights/best.pt")

    # Validation metrics
    metrics = model.val(data=DATA_YAML)
    print("Validation metrics:", metrics)

    # Run inference on a sample image
    sample_image = os.path.join(dataset.location, "valid/images", os.listdir(os.path.join(dataset.location, "valid/images"))[0])
    print("Predicting on:", sample_image)

    results = model.predict(source=sample_image, conf=0.5, save=True)
    results[0].save(filename="prediction.jpg")
    print("Saved prediction to prediction.jpg")


# -------------------------------
# 3. Frame prediction
# -------------------------------

def predict_frame(frame):
    """
    Run YOLOv8 inference on a frame and return the top prediction.
    """
    model = YOLO("runs/rps/exp12/weights/best.pt")  # change path to your trained model

    results = model.predict(source=frame, verbose=False)

    if len(results[0].boxes) == 0:
        return None, 0.0

    best_box = results[0].boxes[results[0].boxes.conf.argmax()]
    class_id = int(best_box.cls[0])
    confidence = float(best_box.conf[0])
    class_name = model.names[class_id]
    return class_name, confidence


# ------------------------
# Camera functions
# ------------------------
def init_camera():
    tl = pylon.TlFactory.GetInstance()
    device = tl.CreateFirstDevice()
    camera = pylon.InstantCamera(device)
    camera.Open()
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    return camera, converter


def display_camera_video(camera, converter, duration_sec=5, model=None):
    """
    Grab frames from camera, display them live with YOLO predictions on every frame.
    Shows bounding boxes and labels.
    """
    start_time = datetime.now()
    frame_counter = 0

    while duration_sec < 0 or (datetime.now() - start_time).total_seconds() < duration_sec:
        if camera.IsGrabbing():
            grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grab_result.GrabSucceeded():
                frame = converter.Convert(grab_result).GetArray()
                frame_counter += 1

                # Run YOLO prediction on this frame
                if model:
                    results = model.predict(frame, verbose=False)
                    
                    # results[0] contains detections for this frame
                    for box in results[0].boxes:
                        # Extract coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        label = model.names[cls]

                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # Label + confidence text
                        text = f"{label} {conf:.2f}"
                        cv2.putText(frame, text, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                    (0, 255, 0), 2, cv2.LINE_AA)

                # Show frame
                cv2.imshow("Live Camera Feed", frame)
                key = cv2.waitKey(1)
                if key == 27:  # ESC to quit
                    break

            grab_result.Release()

    cv2.destroyAllWindows()
    print("Live display finished.")

def display_video_file(video_path, model=None):
    """
    Run YOLO predictions on a video file (.mp4, .avi, etc.).
    Displays bounding boxes and labels on each frame.
    Press ESC to quit early.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        if model:
            results = model.predict(frame, verbose=False)

            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Label + confidence
                text = f"{label} {conf:.2f}"
                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("YOLO Video Prediction", frame)
        key = cv2.waitKey(1)
        if key == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Video playback finished.")

def close_camera(camera):
    if camera.IsGrabbing():
        camera.StopGrabbing()
    camera.Close()

if __name__ == "__main__":
    train_model = True
    train_model = False
    
    live = True

    if train_model:
        train_yolo()
        evaluate_and_predict()
    else:
        model = YOLO("runs/rps/11m/weights/best.pt") 
        if live:
            cam, conv = init_camera()
            try:
                display_camera_video(cam, conv, duration_sec=-1, model=model)
            finally:
                close_camera(cam)
        else:  
            display_video_file("game_slow_motion_recordings/recording_2025-09-26_17-04-20_926_slow_motion.mp4", model=model)
        
