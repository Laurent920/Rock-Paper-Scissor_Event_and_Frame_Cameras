# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Sample code that demonstrates how to use Metavision SDK to record events from a live camera in a RAW file
"""

from metavision_core.event_io.raw_reader import initiate_device
from metavision_core.event_io import EventsIterator
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIAction, UIKeyEvent
import argparse
import time
from datetime import datetime
import os


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Metavision RAW file Recorder sample.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-o', '--output-dir', default="raw_recordings", help="Directory where to create RAW file with recorded event data")
    args = parser.parse_args()
    return args


def main():
    """ Main """
    args = parse_args()

    # HAL Device on live camera
    device = initiate_device("")

    # Start the recording
    if device.get_i_events_stream():
        log_path = "recording_" + time.strftime("20%y-%m-%d_%H-%M-%S", time.localtime()) + ".raw"
        if args.output_dir != "":
            log_path = os.path.join(args.output_dir, log_path)
        print(f'Recording to {log_path}')
        device.get_i_events_stream().log_raw_data(log_path)

    # Events iterator on Device
    mv_iterator = EventsIterator.from_device(device=device)
    height, width = mv_iterator.get_size()  # Camera Geometry

    # Window - Graphical User Interface
    with MTWindow(title="Metavision Events Viewer", width=width, height=height,
                  mode=BaseWindow.RenderMode.BGR) as window:
        def keyboard_cb(key, scancode, action, mods):
            if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                window.set_close_flag()

        window.set_keyboard_callback(keyboard_cb)

        # Event Frame Generator
        event_frame_gen = PeriodicFrameGenerationAlgorithm(sensor_width=width, sensor_height=height, fps=25,
                                                           palette=ColorPalette.Dark)

        def on_cd_frame_cb(ts, cd_frame):
            window.show_async(cd_frame)

        event_frame_gen.set_output_callback(on_cd_frame_cb)

        # Process events
        for evs in mv_iterator:
            # Dispatch system events to the window
            EventLoop.poll_and_dispatch()
            event_frame_gen.process_events(evs)

            if window.should_close():
                # Stop the recording
                device.get_i_events_stream().stop_log_raw_data()
                break


def record_raw_file(duration_sec: int = 4, output_dir: str = "raw_recordings", output_file_addition="", debug=False) -> str:
    """
    Record events from the first available live camera into a RAW file.

    This starts logging (log_raw_data) and then consumes events from
    EventsIterator.from_device(...) for `duration_sec` seconds so that
    events are actually streamed and written to disk.

    Returns:
        Path to the recorded RAW file.
    """
    os.makedirs(output_dir, exist_ok=True)
    t0 = datetime.now().strftime("20%y-%m-%d_%H-%M-%S_%f")[:-3]

    # Open HAL device on live camera (empty string => first available)
    device = initiate_device("")
    if device is None:
        raise RuntimeError("Could not open a live device with initiate_device('').")

    ev_stream = device.get_i_events_stream()
    if ev_stream is None:
        raise RuntimeError("Device does not expose get_i_events_stream(). Cannot record.")

    # Build output path (same format as the SDK sample)
    t1 = datetime.now().strftime("20%y-%m-%d_%H-%M-%S_%f")[:-3]

    timestamp = t1
    raw_path = os.path.join(output_dir, f"{output_file_addition}_recording_{timestamp}.raw")
    
    if debug: print(f"Starting recording to: {raw_path}")
    # Start logging raw data
    ev_stream.log_raw_data(raw_path)
    t2 = datetime.now().strftime("20%y-%m-%d_%H-%M-%S_%f")[:-3]

    # Create an EventsIterator that will start the device streaming.
    # Iterate over it and stop after duration_sec seconds.
    mv_iterator = EventsIterator.from_device(device=device)
    t3 = datetime.now().strftime("20%y-%m-%d_%H-%M-%S_%f")[:-3]

    start_t = time.time()
    try:
        for evs in mv_iterator:
            # evs is a buffer of events. We don't need to process it here—
            # just iterating is enough to make the device produce events.
            if time.time() - start_t >= duration_sec:
                break
            # Small yield to avoid a tight busy loop if necessary
            # (the iterator usually blocks/waits, so this may not be required).
            # time.sleep(0.001)
    finally:
        # Always stop logging
        ev_stream.stop_log_raw_data()
        if debug: print("Stopped recording.")
    t4 = datetime.now().strftime("20%y-%m-%d_%H-%M-%S_%f")[:-3]

    if debug: print(f"t0: {t0}, t1: {t1}, t2: {t2}, t3: {t3}, t4: {t4}")
    # Quick sanity check on file size: if it's tiny warn the user
    try:
        size = os.path.getsize(raw_path)
        if debug: print(f"Recorded file size: {size} bytes")
        if size < 1024:
            print("Warning: recorded file is very small — check camera connection/streaming.")
    except OSError:
        print("Warning: could not stat the recorded file.")

    return raw_path

if __name__ == "__main__":
    # main()
    record_raw_file()
