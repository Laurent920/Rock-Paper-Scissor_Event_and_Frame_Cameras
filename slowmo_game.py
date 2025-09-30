import pygame
import sys
import time
import os
import json
from datetime import datetime, timedelta
import random
import threading
import cv2
from camera_slow_motion import convert_to_slow_motion
import queue
import keyboard 

LANG = "ENG"

RAW_DIR = "raw_recordings"
SLOWMO_DIR = "game_slow_motion_recordings"
CHECK_INTERVAL = 5  # seconds
WIDTH, HEIGHT = 1280, 820
FONT_SIZE = 50
TEXT_COLOR = (255, 255, 255)
BG_COLOR = (0, 0, 0)
PLAYBACK_FPS = 15         # fps for slow-motion playback
ACCUMULATION_TIME = 5000   # accumulation time in microseconds per frame for slow-motion video (basic = 33 333 = 33.333ms)

KEEP_ENDING = 700    # keep last 700ms of the recording for slow-motion video (in milliseconds)

# Game states
STATE_WAIT = 'wait'
STATE_PLAYING = 'playing'
STATE_FINISHED = 'finished'

# Keep track of already processed raw file
processed_files = set()

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

ensure_dir(RAW_DIR)
ensure_dir(SLOWMO_DIR)

# Queues for communication
key_queue = queue.Queue()
new_slowmo_queue = queue.Queue()   # holds new slow-mo files


# ---------------- Background slow-motion processor ---------------- #
def process_new_raw_files():
    while True:
        try:
            raw_files = set(f for f in os.listdir(RAW_DIR) if f.endswith(".raw"))  # Only .raw files
            new_files = raw_files - processed_files
            for file in new_files:
                raw_path = os.path.join(RAW_DIR, file)
                output_file = os.path.splitext(file)[0] + "_slow_motion.mp4"
                output_path = os.path.join(SLOWMO_DIR, output_file)
                # Skip if slow-motion file already exists
                if os.path.exists(output_path):
                    print(f"Skipping {file}, slow-motion file already exists.")
                    processed_files.add(file)
                    continue

                print(f"Processing {file} -> {output_path}")
                convert_to_slow_motion(raw_path, output_dir=SLOWMO_DIR,
                                       frame_rate=PLAYBACK_FPS, 
                                       accumulation_time=ACCUMULATION_TIME,
                                       keep_ending=KEEP_ENDING*1000)
                processed_files.add(file)

                # Notify main thread that a new file is ready
                new_slowmo_queue.put(output_path)
        except Exception as e:
            print("Error in processing raw files:", e)
        time.sleep(CHECK_INTERVAL)


# ---------------- Global keyboard listener ---------------- #
def keyboard_listener():
    def on_key_event(event):
        if event.event_type == "down":
            key_queue.put(event.name)
    keyboard.hook(on_key_event)
    keyboard.wait()  # keep listener running


# ---------------- Retrieving AI latest prediction ---------------- #
def find_matching_json(mp4_path, json_dir):
    """
    Given an MP4 file path like:
      game_slow_motion_recordings/0_recording_2025-09-29_11-15-21_472_slow_motion.mp4
    find the matching JSON file in json_dir:
      raw_recordings/0_recording_2025-09-29_11-15-18_506.json
    Matching is based on the first number before `_recording_`.
    """
    filename = os.path.basename(mp4_path)
    rec_number = filename.split("_")[0]  # "0"

    # Look for a JSON file that starts with same rec_number
    for f in os.listdir(json_dir):
        if f.startswith(f"{rec_number}_recording_") and f.endswith(".json"):
            return os.path.join(json_dir, f)

    return None


def load_predictions(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def find_closest_prediction(predictions, target_ts, offset_sec=0.0):
    """
    Given predictions = {timestamp: {"frame_idx":..., "label":..., "conf":...}}
    and a target timestamp string (format "20%y-%m-%d_%H-%M-%S_%f"),
    find the closest earlier/equal timestamp after applying offset_sec.
    """
    # Convert target timestamp + offset into datetime
    fmt = "20%y-%m-%d_%H-%M-%S_%f"
    if len(target_ts.split("_")[-1]) == 3:  # ensure microseconds format
        target_ts += "000"

    target_dt = datetime.strptime(target_ts, fmt) + timedelta(seconds=offset_sec)

    # Convert keys to datetimes for comparison
    pred_times = []
    for ts in predictions.keys():
        ts_norm = ts if len(ts.split("_")[-1]) == 6 else ts + "000"
        pred_times.append(datetime.strptime(ts_norm, fmt))

    # Find closest <= target
    closest_dt = None
    latest_valid_dt = None
    for ts_dt in sorted(pred_times):
        pred_entry = predictions[ts_dt.strftime(fmt)[:-3]]
        if pred_entry["label"] is not None:
            latest_valid_dt = ts_dt 

        if ts_dt <= target_dt:
            closest_dt = ts_dt

    if not closest_dt:
        return None, None  # nothing found before target

    # Retrieve prediction
    closest_key = closest_dt.strftime(fmt)[:-3]  # back to ms
    pred = predictions.get(closest_key)

    # If pred is None, walk backwards
    idx = pred_times.index(closest_dt)
    while pred and pred["label"] is None and idx > 0:
        idx -= 1
        prev_dt = pred_times[idx]
        prev_key = prev_dt.strftime(fmt)[:-3]
        pred = predictions.get(prev_key)
    print(pred_times.index(closest_dt), idx)

    # Retrieve latest valid prediction
    if latest_valid_dt:
        latest_valid_key = latest_valid_dt.strftime(fmt)[:-3]
        latest_valid_pred = predictions[latest_valid_key]
    else:
        latest_valid_pred = None

    return pred, latest_valid_pred

# ---------------- Pygame main loop ---------------- #
def main():
    text_display = None
    with open("instruction_language.json", "r", encoding="utf-8") as f:
        text_display = json.load(f)

    x, y = 1600, 300
    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (x,y)
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(text_display[LANG]["slowmo game"]["window caption"])
    font = pygame.font.Font(None, FONT_SIZE)
    clock = pygame.time.Clock()

    # Start background threads
    threading.Thread(target=process_new_raw_files, daemon=True).start()
    threading.Thread(target=keyboard_listener, daemon=True).start()

    state = STATE_WAIT  # wait, playing, finished
    video_path = None
    cap = None
    video_start_time = None
    prediction_text = ""
    prediction_time = None
    duration = None
    prediction_frame = None
    AI_pred = None
    AI_latest_pred = None
    AI_prediction_made = False

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        while not key_queue.empty():
            key = key_queue.get()
            if key == "esc":
                pygame.quit()
                sys.exit()
            elif state == STATE_WAIT and (key == "r" or key == "p" or key == "s"):
                # Prefer newest slowmo file if available
                if not new_slowmo_queue.empty():
                    video_path = new_slowmo_queue.get()
                else:
                    # fallback: random file
                    slowmo_files = os.listdir(SLOWMO_DIR)
                    if not slowmo_files:
                        print("No slow-motion videos available!")
                        continue
                    video_file = random.choice(slowmo_files)
                    video_path = os.path.join(SLOWMO_DIR, video_file)

                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)         # playback fps
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                video_start_time = time.time()
                prediction_text = ""
                prediction_time = None
                prediction_frame = None
                AI_prediction_made = False
                AI_pred = None
                AI_latest_pred = None
                state = STATE_PLAYING

            elif state == STATE_PLAYING and prediction_text == "":
                if key in ("r", "1"):
                    prediction_text = text_display[LANG]["slowmo game"]["r"]
                    prediction_time = time.time() - video_start_time
                elif key in ("p", "2"):
                    prediction_text = text_display[LANG]["slowmo game"]["p"]
                    prediction_time = time.time() - video_start_time
                elif key in ("s", "3"):
                    prediction_text = text_display[LANG]["slowmo game"]["s"]
                    prediction_time = time.time() - video_start_time
            elif state == STATE_FINISHED:
                state = STATE_WAIT

        screen.fill(BG_COLOR)

        if state == STATE_WAIT:
            text = font.render(text_display[LANG]["slowmo game"]["home"], True, TEXT_COLOR)
            screen.blit(text, text.get_rect(center=(WIDTH//2, HEIGHT//2)))
        elif state == STATE_PLAYING or state == STATE_FINISHED:
            if cap is not None:
                ret, frame = cap.read()
                if not ret:
                    # Video finished
                    cap.release()
                    cap = None
                    state = STATE_FINISHED
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (WIDTH, HEIGHT-120))  # leave space for prediction text
                    surface = pygame.surfarray.make_surface(frame.swapaxes(0,1))
                    screen.blit(surface, (0,0))
                    if not AI_prediction_made:
                        prediction_frame = frame

            if state == STATE_FINISHED and prediction_frame is not None:
                surface = pygame.surfarray.make_surface(prediction_frame.swapaxes(0,1))
                screen.blit(surface, (0,0))

                text = font.render(text_display[LANG]["slowmo game"]["continue"], True, TEXT_COLOR)
                screen.blit(text, text.get_rect(center=(WIDTH//2, 30)))

            # Display prediction text
            if prediction_text:
                real_speed_prediction_time = prediction_time / duration

                if not AI_prediction_made:
                    json_dir = "raw_recordings"
                    json_path = find_matching_json(video_path, json_dir)
                    print("Matching JSON:", json_path)

                    predictions = load_predictions(json_path)

                    filename = os.path.basename(video_path)
                    parts = filename.split("_")
                    if len(parts) >= 5: # Combine date + time + ms
                        event_video_start_time = "_".join(parts[2:5])
                        
                        offset = real_speed_prediction_time
                        AI_pred, AI_latest_pred = find_closest_prediction(predictions, event_video_start_time, offset)
                        print(AI_pred, AI_latest_pred)
                        AI_prediction_made = True

                text = text_display[LANG]["slowmo game"]["pred time"].format(
                            prediction_text=prediction_text,
                            prediction_time=prediction_time,
                            real_speed_prediction_time=real_speed_prediction_time
                        )
                pred_surface = font.render(f"{prediction_text} in {prediction_time:.2f}s (real time: {real_speed_prediction_time:.2f}s)", True, TEXT_COLOR)
                screen.blit(pred_surface, pred_surface.get_rect(center=(WIDTH//2, HEIGHT-90)))
                text = text_display[LANG]["slowmo game"]["AI predict"].format(
                            label=AI_pred["label"],
                            conf=AI_pred["conf"] * 100
                        )
                # pred_surface = font.render(text, True, TEXT_COLOR)
                # screen.blit(pred_surface, pred_surface.get_rect(center=(WIDTH//2, HEIGHT-55)))
                text = text_display[LANG]["slowmo game"]["AI latest predict"].format(
                            label=AI_latest_pred["label"],
                            conf=AI_latest_pred["conf"] * 100
                        )
                pred_surface = font.render(text, True, TEXT_COLOR)
                screen.blit(pred_surface, pred_surface.get_rect(center=(WIDTH//2, HEIGHT-20)))

        pygame.display.flip()
        clock.tick(PLAYBACK_FPS)

if __name__ == "__main__":
    main()
    # Run with sudo /home/guangzhi-tang/Env/prophesee-py3venv/bin/python slowmo_game.py