import pygame
import sys
import random
import time
from datetime import datetime
import cv2
import queue
import keyboard 
import threading
import json
import os
from pypylon import pylon

from ultralytics import YOLO
import multiprocessing as mp 
from collections import deque

from event_camera_recorder import record_raw_file
from camera_slow_motion import convert_to_slow_motion
from frame_camera_recorder import init_camera, close_camera

# UI constants
WIDTH, HEIGHT = 1280, 720
FPS = 60
FONT_SIZE = 100
TEXT_COLOR = (255, 255, 255)
BG_COLOR = (0, 0, 0)

# Timing constants (milliseconds)
GO_DISPLAY_MS = 500         # Additional display time after countdown (1s + GO_DISPLAY_MS)
AI_PLAY_DELAY_MS = 200      # AI makes choice this many ms after green play screen is displayed

MIN_RECORD_MS = 1500        # user's original desired minimum (1s)
PLAYBACK_FPS = 30           # fps for playback
ACCUMULATION_TIME = 10000   # accumulation time in microseconds per frame for slow-motion video

# Game states
STATE_WAIT = 'wait'
STATE_COUNTDOWN = 'countdown'
STATE_DONE = 'done'

DEBUG = True
# Initial state
state = STATE_WAIT
recording_number = 0

# Function run inside a separate process
def record_worker_event(result_queue: mp.Queue, duration_ms: int, output_dir: str = "raw_recordings"):
    """
    Worker that runs in a separate process to execute the recorder.
    It calls your existing record_raw_file(...) and pushes the saved path
    into result_queue when done.
    """
    global recording_number
    try:
        if DEBUG: print("Event camera recorder process started.")
        # convert ms->seconds for record_raw_file
        duration_s = duration_ms / 1000.0
        raw_path = record_raw_file(duration_sec=duration_s, output_dir=output_dir, output_file_addition=recording_number, debug=DEBUG)
        result_queue.put(raw_path)
    except Exception as e:
        # Send exception info back so main process can show an error
        result_queue.put({"error": str(e)})

def predict_frame(model, frame):
    """
    Run YOLOv8 inference on a frame and return the top prediction.
    """
    # model = YOLO("runs/rps/exp12/weights/best.pt")  # change path to your trained model

    results = model.predict(source=frame, verbose=False)

    if len(results[0].boxes) == 0:
        return None, 0.0

    best_box = results[0].boxes[results[0].boxes.conf.argmax()]
    class_id = int(best_box.cls[0])
    confidence = float(best_box.conf[0])
    class_name = model.names[class_id]
    return class_name, confidence

def record_worker_frame(pred_stack, output_folder="raw_recordings"):
    """
    Continuous camera worker.
    - Only runs YOLO when `state_flag.value` == 1 (STATE_COUNTDOWN)
    - Otherwise discards frames.
    - Pushes latest prediction to shared pred_stack.
    """
    global state
    global recording_number
    camera, converter = init_camera()

    # Load YOLO model
    model = YOLO("runs/rps/11m/weights/best.pt")  # replace with the correct model
    
    valid_prediction = None
    valid_prediction_tick = 0
    tick = 0 

    recording = False
    writer = None
    predictions = {}
    base_name = None

    width = camera.Width.GetValue()
    height = camera.Height.GetValue()
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    
    print("Camera worker started.")
    try:    
        # print("", datetime.now().strftime("20%y-%m-%d_%H-%M-%S_%f")[:-3])
        while True:
            if camera.IsGrabbing():
                grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                if grab_result.GrabSucceeded():
                    frame = converter.Convert(grab_result).GetArray()
                    timestamp = datetime.now().strftime("20%y-%m-%d_%H-%M-%S_%f")[:-3]
                    
                    if state == STATE_COUNTDOWN and not recording:
                        t_start = datetime.now().strftime("20%y-%m-%d_%H-%M-%S_%f")[:-3]
                        base_name = f"{recording_number}_recording_{t_start}"
                        os.makedirs(output_folder, exist_ok=True)

                        video_path = os.path.join(output_folder, f"{base_name}.mp4")
                        writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

                        recording = True
                        if DEBUG: print("Frame camera started recording at :", t_start)

                    # Only predict when state is STATE_COUNTDOWN
                    if state == STATE_COUNTDOWN and recording:
                        writer.write(frame)

                        label, conf = predict_frame(model, frame)
                        predictions[timestamp] = {"frame_idx": tick, "label": label, "conf": conf}

                        # Store latest prediction in shared stack if 
                        #   - The latest prediction is None
                        #   - The current prediction is valid
                        #   - It's been more than 3 ticks since last valid prediction          
                        if valid_prediction is None or label is not None or tick - valid_prediction_tick > 3:
                            pred_stack.append((label, conf))
                            valid_prediction = label
                            valid_prediction_tick = tick
                        tick += 1
                    elif state != STATE_COUNTDOWN and recording:
                        writer.release()
                        writer = None

                        # Save predictions to JSON
                        json_path = os.path.join(output_folder, f"{base_name}.json")
                        with open(json_path, "w") as f:
                            json.dump(predictions, f, indent=4)

                        if DEBUG: print(f"Stopped recording. Predictions saved: {json_path}")

                        # Prepare for next session
                        valid_prediction = None
                        predictions = {}
                        tick = 0
                        recording = False
                        recording_number += 1
                        pred_stack.clear()
                grab_result.Release()
            else:
                time.sleep(0.01)  # small delay to avoid busy loop
            
    except Exception as e:
        print("CLOSING FRAME CAMERA DUE TO ERROR:", e)
        close_camera(camera)

def keyboard_listener(key_queue: queue.Queue):
    def on_key_event(event):
        if event.event_type == "down":
            # print(event.name)
            key_queue.put(event.name)
    keyboard.hook(on_key_event)
    keyboard.wait()  # keep listener running

def diff_timestamps(t1_str: str, t2_str: str) -> float:
    """
    Compute the difference in milliseconds between two string timestamps.

    Args:
        t1_str (str): Timestamp string formatted as "20%y-%m-%d_%H-%M-%S_%f" (millisecond precision).
        t2_str (str): Same format.

    Returns:
        float: Difference in milliseconds (t2 - t1).
    """
    # Define the parsing format (with microseconds)
    fmt = "20%y-%m-%d_%H-%M-%S_%f"
    
    # Re-append '000' if input is truncated to ms
    if len(t1_str.split("_")[-1]) == 3:
        t1_str += "000"
    if len(t2_str.split("_")[-1]) == 3:
        t2_str += "000"

    t1 = datetime.strptime(t1_str, fmt)
    t2 = datetime.strptime(t2_str, fmt)
    
    diff = (t2 - t1).total_seconds() * 1000  # milliseconds
    if DEBUG: print("diff_timestamps:", diff)
    return diff

def get_next_recording_number(directory="recordings"):
    """
    Scan a directory for .raw files and determine the next recording number.
    Filenames must start with a number followed by '_' (e.g., '12_filename.raw').
    """
    recording_number = 0

    if not os.path.exists(directory):
        return recording_number  # start at 0 if folder doesn't exist

    for fname in os.listdir(directory):
        if fname.endswith(".raw"):
            parts = fname.split("_")
            if len(parts) > 1 and parts[0].isdigit():
                num = int(parts[0])
                if num >= recording_number:
                    recording_number = num + 1

    return recording_number

def main():
    global state
    global recording_number

    x, y = 300, 300
    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (x,y)

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Countdown Rock-Paper-Scissors")
    font = pygame.font.Font(None, FONT_SIZE)
    clock = pygame.time.Clock()

    # Start keyboard listener thread
    key_queue = queue.Queue()
    threading.Thread(target=keyboard_listener, args=(key_queue,), daemon=True).start()

    # Start frame camera recorder thread
    pred_stack = deque(maxlen=1)
    threading.Thread(target=record_worker_frame, args=(pred_stack,), daemon=True).start()
    
    recording_number = get_next_recording_number("raw_recordings")

    # multiprocessing objects (set when recording starts)
    recorder_proc = None
    recorder_q = None

    # bookkeeping
    start_ticks = 0
    choice = None
    raw_file_path = None
    t_press = None
    t_play = None

    while True:
        while not key_queue.empty():
            key = key_queue.get()
            if state == STATE_WAIT and key == "space":
                t_press = datetime.now().strftime("20%y-%m-%d_%H-%M-%S_%f")[:-3]

                # Start countdown and recording at the same time
                state = STATE_COUNTDOWN
                start_ticks = pygame.time.get_ticks()

                recorder_q = mp.Queue()
                recorder_proc = mp.Process(target=record_worker_event,
                                           args=(recorder_q, MIN_RECORD_MS, "raw_recordings"),
                                           daemon=True)
                recorder_proc.start()
                if DEBUG: print(f"keyboard press time: {t_press}")
            elif key == "esc":
                if recorder_proc is not None and recorder_proc.is_alive():
                    recorder_proc.terminate()
                    recorder_proc.join(timeout=1)
                pygame.quit()
                sys.exit()

        # screen.fill(BG_COLOR)

        if state == STATE_WAIT:
            text = font.render("Press Space to Start", True, TEXT_COLOR)
            rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
            screen.blit(text, rect)

        elif state == STATE_COUNTDOWN:
            elapsed = pygame.time.get_ticks() - start_ticks
            bg = BG_COLOR

            if elapsed < 1000:
                display = "4"
            elif elapsed < 2000:
                display = "3"
            elif elapsed < 3000:
                display = "2"
            elif elapsed < 4000:
                display = "1"
            elif elapsed < 5000 + GO_DISPLAY_MS + AI_PLAY_DELAY_MS:
                # Green background
                bg = (0, 200, 0)  
                screen.fill(bg)

                # Draw "GO!" near the top
                go_text = font.render("GO!", True, TEXT_COLOR)
                go_rect = go_text.get_rect(center=(WIDTH // 2, HEIGHT // 3))
                screen.blit(go_text, go_rect)

                if choice is None and elapsed >= 4000 + AI_PLAY_DELAY_MS:
                    if len(pred_stack) > 0:
                        pred, conf = pred_stack[-1]
                        if pred is not None:
                            print(f"AI predicted {pred} with confidence {conf:.2f}")
                            if pred == "Rock":
                                choice = "Paper"
                            elif pred == "Paper":
                                choice = "Scissors"
                            else:
                                choice = "Rock"
                        else:
                            print("No prediction from stack ===> random choice")
                            choice = random.choice(["Rock", "Paper", "Scissors"])
                    else:
                        print("Empty prediction stack ===> random choice")
                        choice = random.choice(["Rock", "Paper", "Scissors"])
                    
                    display = choice
                    t_play = datetime.now().strftime("20%y-%m-%d_%H-%M-%S_%f")[:-3]
            else:
                state = STATE_DONE

            if elapsed < 4000:
                screen.fill(BG_COLOR)
                text = font.render(display, True, TEXT_COLOR)
                rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
                screen.blit(text, rect)

            # Draw choice below center if available
            if choice:
                choice_text = font.render(choice, True, TEXT_COLOR)
                choice_rect = choice_text.get_rect(center=(WIDTH // 2, 2 * HEIGHT // 3))
                screen.blit(choice_text, choice_rect)

        elif state == STATE_DONE:
            # Try to get the result from the recorder process
            if recorder_q is not None and raw_file_path is None:
                try:
                    res = recorder_q.get(timeout=0.1)
                    if isinstance(res, dict) and "error" in res:
                        err_msg = res["error"]
                        print("Recorder error:", err_msg)
                        err_text = font.render("Recorder error", True, (255, 0, 0))
                        screen.fill(BG_COLOR)
                        screen.blit(err_text, err_text.get_rect(center=(WIDTH//2, HEIGHT//2)))
                        pygame.display.flip()
                        time.sleep(3)
                        pygame.quit()
                        sys.exit(1)
                    else:
                        raw_file_path = res
                        if DEBUG: print("choice played at time:", t_play)
                        diff_timestamps(t_press, t_play)
                except queue.Empty:
                    pass  # Not ready yet

            # If recording finished, proceed
            if raw_file_path is not None:
                if recorder_proc is not None:
                    recorder_proc.join(timeout=1)
                if DEBUG: print("Raw file produced at:", raw_file_path)

                # screen.fill(BG_COLOR)
                # proc_text = font.render("Processing...", True, TEXT_COLOR)
                # screen.blit(proc_text, proc_text.get_rect(center=(WIDTH//2, HEIGHT//2)))
                # pygame.display.flip()

                slow_motion_path = convert_to_slow_motion(raw_file_path, frame_rate=PLAYBACK_FPS, accumulation_time=ACCUMULATION_TIME, debug=DEBUG)
                if DEBUG: print("Slow-motion path:", slow_motion_path)

                cap = cv2.VideoCapture(slow_motion_path)
                if not cap.isOpened():
                    print("Error opening slow-motion video:", slow_motion_path)
                    pygame.quit()
                    sys.exit(1)

                # Play back at the same frame rate as used in convert_to_slow_motion
                playback_delay = 1.0 / PLAYBACK_FPS

                while cap.isOpened():
                    start_time = time.time()
                    ret, frame = cap.read() 
                    if not ret:
                        break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (WIDTH, HEIGHT))
                    surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                    screen.blit(surface, (0, 0))
                    pygame.display.flip()
                    for ev in pygame.event.get():
                        if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                            cap.release()
                            pygame.quit()
                            sys.exit()
                    elapsed = time.time() - start_time
                    if playback_delay > elapsed:
                        time.sleep(playback_delay - elapsed)
                cap.release()

                state = STATE_WAIT
                recorder_proc = None
                recorder_q = None
                raw_file_path = None
                choice = None
                recording_number += 1

        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()
    # Run with sudo /home/guangzhi-tang/Env/prophesee-py3venv/bin/python countdown_game.py