import os
import math
from collections import deque
import numpy as np
import cv2

from metavision_core.event_io.raw_reader import initiate_device
from metavision_core.event_io import EventsIterator, LiveReplayEventsIterator
from metavision_sdk_core import BaseFrameGenerationAlgorithm
from metavision_sdk_ui import MTWindow, BaseWindow, EventLoop, UIKeyEvent


def convert_to_slow_motion(
    input_path: str,
    accumulation_time: int = 33300,
    frame_rate: int = 30,
    replay_speed: float = 1.0,
    output_dir: str = "replay_recordings",
    replay_with_metavision_window: bool = False,
    keep_ending: int = None,  # µs of events to keep from the end; None = full video
    debug=False
) -> str:
    """
    Convert a RAW/HDF5 recording into a slow-motion MP4 video.

    Args:
        input_path (str): Path to input RAW or HDF5 file.
        accumulation_time (int): Time slice in µs for frame generation (default: 33300 ≈ 33.3 ms).
        frame_rate (int): Output video FPS (default: 30).
        replay_speed (float): Replay factor (default: 1.0).
        output_dir (str): Directory to store the resulting video.
        replay_with_metavision_window (bool): If True, show in Metavision window instead of saving.
        keep_ending (int, optional): If set, only keep the last `keep_ending` µs of events.

    Returns:
        str: Path to the generated slow-motion video (if saved) or "" if shown in window mode.
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_video_path = os.path.join(output_dir, f"{base_name}_slow_motion.mp4")

    # Init events
    device = initiate_device(input_path)
    events_it = EventsIterator.from_device(device, delta_t=accumulation_time)
    slicer = LiveReplayEventsIterator(events_it, replay_factor=replay_speed)

    height, width = events_it.get_size()
    frame = np.zeros((height, width, 3), np.uint8)

    # Decide how many frames to keep
    frames_to_keep = None
    if keep_ending is not None and keep_ending > 0:
        frames_to_keep = max(1, math.ceil(keep_ending / float(accumulation_time)))

    # If we need to trim, buffer frames; otherwise, write/show immediately
    if frames_to_keep is not None:
        frame_deque = deque(maxlen=frames_to_keep)
        for event in slicer:
            EventLoop.poll_and_dispatch()
            BaseFrameGenerationAlgorithm.generate_frame(event, frame)
            frame_deque.append(frame.copy())

        # ---- WINDOW MODE ----
        if replay_with_metavision_window:
            with MTWindow(title="Metavision Slow Motion Viewer", width=width, height=height,
                          mode=BaseWindow.RenderMode.BGR) as window:
                def keyboard_cb(key, scancode, action, mods):
                    if key in (UIKeyEvent.KEY_ESCAPE, UIKeyEvent.KEY_Q):
                        window.set_close_flag()
                window.set_keyboard_callback(keyboard_cb)

                for f in frame_deque:
                    window.show_async(f)
            return ""

        # ---- FILE MODE ----
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))
        if not writer.isOpened():
            raise RuntimeError("Could not open video writer.")
        for f in frame_deque:
            writer.write(f)
        writer.release()
        if debug: print(f"Video saved successfully (trimmed to last {frames_to_keep} frames).")
        return output_video_path

    else:
        # ---- FULL REPLAY (no trimming) ----
        if replay_with_metavision_window:
            with MTWindow(title="Metavision Slow Motion Viewer", width=width, height=height,
                          mode=BaseWindow.RenderMode.BGR) as window:
                def keyboard_cb(key, scancode, action, mods):
                    if key in (UIKeyEvent.KEY_ESCAPE, UIKeyEvent.KEY_Q):
                        window.set_close_flag()
                window.set_keyboard_callback(keyboard_cb)

                for event in slicer:
                    EventLoop.poll_and_dispatch()
                    BaseFrameGenerationAlgorithm.generate_frame(event, frame)
                    window.show_async(frame)
            return ""

        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))
            if not writer.isOpened():
                raise RuntimeError("Could not open video writer.")
            for event in slicer:
                EventLoop.poll_and_dispatch()
                BaseFrameGenerationAlgorithm.generate_frame(event, frame)
                writer.write(frame)
            writer.release()
            if debug: print("Video saved successfully (full replay).")
            return output_video_path



def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert RAW/HDF5 recording to slow-motion video.")
    parser.add_argument("-i", "--input-event-file", required=True,
                        help="Path to input event file (RAW or HDF5).")
    parser.add_argument("-o", "--output-dir", default="replay_recordings",
                        help="Directory to store the slow-motion video.")
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS.")
    parser.add_argument("--accumulation", type=int, default=33300,
                        help="Accumulation time in microseconds per frame.")
    parser.add_argument("--replay-speed", type=float, default=1.0,
                        help="Replay factor (e.g. 0.5 = 2x faster, 2.0 = 2x slower).")
    parser.add_argument("--show", action="store_true",
                        help="Show video in Metavision window instead of saving.")
    args = parser.parse_args()

    convert_to_slow_motion(
        input_path=args.input_event_file,
        accumulation_time=args.accumulation,
        frame_rate=args.fps,
        replay_speed=args.replay_speed,
        output_dir=args.output_dir,
        replay_with_metavision_window=args.show
    )


if __name__ == "__main__":
    main()
