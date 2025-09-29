import os
import time
from datetime import datetime
import cv2
from pypylon import pylon

def init_camera():
    tl = pylon.TlFactory.GetInstance()
    device = tl.CreateFirstDevice()
    camera = pylon.InstantCamera(device)
    camera.Open()

    # Configure grabbing mode, pixel format, etc.
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    # Create converter to get BGR8 images
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    return camera, converter

def record_camera_video(camera, converter, duration_sec=5, output_dir="frame_recordings"):
    os.makedirs(output_dir, exist_ok=True)

    t1 = datetime.now().strftime("20%y-%m-%d_%H-%M-%S_%f")[:-3]
    base_name = f"recording_{t1}"
    raw_path = os.path.join(output_dir, base_name + ".avi")
    img_path = os.path.join(output_dir, base_name + "_first_frame.png")

    width = camera.Width.GetValue()
    height = camera.Height.GetValue()
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(raw_path, fourcc, fps, (width, height))

    print(f"Recording video to {raw_path}")
    print(f"Saving first frame to {img_path}")

    start_time = datetime.now()
    first_frame_saved = False

    while (datetime.now() - start_time).total_seconds() < duration_sec:
        if camera.IsGrabbing():
            grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grab_result.GrabSucceeded():
                frame = converter.Convert(grab_result).GetArray()
                if not first_frame_saved:
                    cv2.imwrite(img_path, frame)
                    first_frame_saved = True
                writer.write(frame)
            grab_result.Release()

    writer.release()
    print("Recording finished.")

    return raw_path, img_path


def close_camera(camera):
    if camera.IsGrabbing():
        camera.StopGrabbing()
    camera.Close()

# Example usage
if __name__ == "__main__":
    cam, conv = init_camera()
    try:
        video_path, image_path = record_camera_video(cam, conv, duration_sec=5, output_dir="frame_recordings")
        print("Video:", video_path)
        print("First frame:", image_path)
    finally:
        close_camera(cam)
