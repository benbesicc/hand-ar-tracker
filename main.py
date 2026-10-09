"""main entry point for the hand ar tracker application

initializes the webcam capture hand tracker gesture recognizer
renderer and fps counter runs the main processing loop that
captures frames detects hands recognizes gestures and renders
the augmented reality overlay in real time
"""

import sys
import time
import math
from typing import Dict, Any, Optional, Tuple

import cv2
import yaml
import numpy as np

from tracker import HandTracker
from renderer import HandRenderer
from gesture import recognize_gesture
from utils.fps_counter import FPSCounter
from utils.device_utils import select_device, get_device_label
from ascii_processor import AsciiProcessor  #  new import

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """load and validate the configuration from a yaml file

    reads the configyaml file and merges with default values
    for any missing fields to ensure all required settings exist

    args
        configpath path to the yaml configuration file

    returns
        a dictionary containing all configuration settings with
        defaults applied for any missing values
    """

    defaults = _get_default_config()
    try:
        with open(config_path, "r", encoding="utf-8") as config_file:
            user_config = yaml.safe_load(config_file)
        if user_config is None:
            return defaults
        return _merge_configs(defaults, user_config)
    except FileNotFoundError:
        print(f"[WARNING] Config file '{config_path}' not found, using defaults.")
        return defaults

def _get_default_config() -> Dict[str, Any]:
    """return the default configuration dictionary

    provides sensible defaults for all configuration parameters
    so the application can run without a config file

    returns
        dictionary with default values for all config sections
    """

    return {
        "camera": {
            "index": 0,
            "width": 1280,
            "height": 720,
            "fps": 60,
            "rotate_180": True,
        },
        "tracking": {
            "max_hands": 2,
            "min_detection_confidence": 0.8,
            "min_tracking_confidence": 0.7,
            "model_complexity": 1,
        },
        "renderer": {
            "skeleton_color": [0, 255, 0],
            "keypoint_color": [0, 0, 255],
            "keypoint_radius": 6,
            "skeleton_thickness": 2,
            "show_fps": True,
            "show_coordinates": False,
            "show_gesture_label": False,
            "show_hand_label": False,
            "show_device_label": True,
            "show_ascii_overlay": True,
        },
        "gestures": {
            "enabled": False,
            "pinch_threshold": 0.05,
            "fist_threshold": 0.85,
        },
        "device": "auto",
    }

def _merge_configs(
    defaults: Dict[str, Any],
    overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """recursively merge user config overrides into default config

    for nested dictionaries merges at each level for other types
    the override value replaces the default

    args
        defaults default configuration dictionary
        overrides user-provided configuration overrides

    returns
        merged configuration dictionary with user values taking
        precedence over defaults
    """

    merged = defaults.copy()
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged

def initialize_camera(config: Dict[str, Any]) -> cv2.VideoCapture:
    """initialize the webcam video capture device

    opens the camera specified in the config and sets the requested
    resolution and frame rate

    args
        config configuration dictionary containing camera settings
            under the camera key

    returns
        an opened cv2videocapture object ready for frame capture

    raises
        systemexit if the camera cannot be opened
    """

    cam_config = config["camera"]
    cap = cv2.VideoCapture(cam_config["index"])
    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Check camera index in config.yaml.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_config["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_config["height"])
    cap.set(cv2.CAP_PROP_FPS, cam_config["fps"])
    print(f"[INFO] Camera opened: index={cam_config['index']}, "
          f"resolution={cam_config['width']}x{cam_config['height']}")
    return cap

def initialize_tracker(config: Dict[str, Any]) -> HandTracker:
    """create and configure the hand tracker instance

    args
        config configuration dictionary containing tracking settings
            under the tracking key

    returns
        a configured handtracker instance ready for frame processing
    """

    track_config = config["tracking"]
    tracker = HandTracker(
        max_hands=track_config["max_hands"],
        min_detection_confidence=track_config["min_detection_confidence"],
        min_tracking_confidence=track_config["min_tracking_confidence"],
        model_complexity=track_config.get("model_complexity", 1),
    )
    print("[INFO] Hand tracker initialized.")
    return tracker

def initialize_renderer(config: Dict[str, Any]) -> HandRenderer:
    """create and configure the hand renderer instance

    args
        config configuration dictionary containing renderer settings
            under the renderer key

    returns
        a configured handrenderer instance ready for frame rendering
    """

    rend_config = config["renderer"]
    renderer = HandRenderer(
        skeleton_color=tuple(rend_config["skeleton_color"]),
        keypoint_color=tuple(rend_config["keypoint_color"]),
        keypoint_radius=rend_config.get("keypoint_radius", 8),
        skeleton_thickness=rend_config["skeleton_thickness"],
        show_fps=rend_config["show_fps"],
        show_coordinates=rend_config.get("show_coordinates", False),
        show_gesture_label=rend_config.get("show_gesture_label", False),
        show_hand_label=rend_config.get("show_hand_label", False),
        show_device_label=rend_config["show_device_label"],
    )
    print("[INFO] Renderer initialized.")
    return renderer

def process_hands(
    frame: np.ndarray,
    tracker: HandTracker,
    renderer: HandRenderer,
    config: Dict[str, Any],
    show_gui: bool = True,
    clap_active: bool = False,
    time_val: float = 0.0,
    reverse_mode: bool = False,
    locked_coords: Optional[Tuple[list, list]] = None,
) -> Tuple[np.ndarray, bool, bool, Optional[Tuple[list, list]]]:
    """process a single frame for hand detection and rendering

    returns
        tuple of (renderedframe clapactive snapdetected currentcoords)
    """

    if not hasattr(process_hands, "hands_close"):
        process_hands.hands_close = False
    if not hasattr(process_hands, "prev_snap_dists"):
        process_hands.prev_snap_dists = {}

    results = tracker.process_frame(frame)
    all_landmarks = tracker.extract_landmarks(results)
    hand_labels = tracker.extract_handedness(results)

    #  smooth landmarks
    smoothed_all = [
        tracker.get_smoothed_landmarks(i, lm)
        for i, lm in enumerate(all_landmarks)
    ]

    #   snap detection thumb tip (4)  middle finger tip (12) rapid separation 
    snap_detected = False
    new_snap_dists: dict = {}
    for i, smoothed in enumerate(smoothed_all):
        thumb = smoothed[4]
        mid   = smoothed[12]
        d = math.sqrt(
            (thumb[0] - mid[0])**2 + (thumb[1] - mid[1])**2 + (thumb[2] - mid[2])**2
        )
        prev_d = process_hands.prev_snap_dists.get(i, d)
        #  snap  was close ( 005) now far ( 013)  rapid separation
        if prev_d < 0.05 and d > 0.13:
            snap_detected = True
            print(f"[INFO] Snap detected on hand {i}!")
        new_snap_dists[i] = d
    process_hands.prev_snap_dists = new_snap_dists

    current_coords = None

    #   handle  2 hands 
    if len(all_landmarks) < 2:
        clap_active = False
        process_hands.hands_close = False
        if show_gui:
            if len(all_landmarks) == 0:
                frame = renderer.draw_no_hands_message(frame)
            else:
                gesture_config = config["gestures"]
                for i, smoothed in enumerate(smoothed_all):
                    label = hand_labels[i] if i < len(hand_labels) else ""
                    gesture = _get_gesture(smoothed, gesture_config)
                    frame = renderer.draw_hand(frame, smoothed, label, gesture)
    else:
        #   2 hands detected 
        clap_active = True
        sorted_indices = sorted(range(len(smoothed_all)), key=lambda idx: smoothed_all[idx][0][0])
        current_coords = (smoothed_all[sorted_indices[0]], smoothed_all[sorted_indices[1]])
        
        #  draw skeleton overlay
        if show_gui:
            gesture_config = config["gestures"]
            for i, smoothed in enumerate(smoothed_all):
                label = hand_labels[i] if i < len(hand_labels) else ""
                gesture = _get_gesture(smoothed, gesture_config)
                frame = renderer.draw_hand(frame, smoothed, label, gesture)

    #   render ribbon 
    coords_to_use = locked_coords if locked_coords else current_coords
    if coords_to_use:
        coords_l, coords_r = coords_to_use
        if reverse_mode:
            #  reverse mode ribbon area shows real live video (just draw glowing border)
            frame = renderer.draw_real_ribbon_border(frame, coords_l, coords_r)
        else:
            person_mask = tracker.segment_frame(frame)
            frame = renderer.draw_holographic_ribbon(frame, coords_l, coords_r, time_val, person_mask)

    return frame, clap_active, snap_detected, current_coords

def _get_gesture(
    landmarks: list,
    gesture_config: Dict[str, Any],
) -> str:
    """determine the gesture for a set of hand landmarks

    args
        landmarks list of 21 (x y z) coordinate tuples
        gestureconfig gesture configuration with thresholds

    returns
        gesture name string or empty string if gestures are disabled
    """

    if not gesture_config.get("enabled", True):
        return ""
    return recognize_gesture(
        landmarks,
        pinch_threshold=gesture_config.get("pinch_threshold", 0.05),
        fist_threshold=gesture_config.get("fist_threshold", 0.85),
    )

def run_main_loop(
    cap: cv2.VideoCapture,
    tracker: HandTracker,
    renderer: HandRenderer,
    fps_counter: FPSCounter,
    device_label: str,
    config: Dict[str, Any],
) -> None:
    """execute the main webcam processing loop

    continuously captures frames processes them for hand detection
    and gesture recognition renders overlays and displays results
    exits on q key press or escape key

    args
        cap opened video capture device
        tracker configured hand tracker instance
        renderer configured hand renderer instance
        fpscounter fps counter for performance monitoring
        devicelabel string label for the active compute device
        config full configuration dictionary
    """

    print("[INFO] Starting main loop. Press 'q' or ESC to quit.")
    ascii_processor = AsciiProcessor()

    window_name = "Hand AR Tracker"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    btn_bbox     = [140, 10, 260, 35]
    show_gui     = True
    show_skeleton = True
    clap_active  = False
    reverse_mode = False   #  snap toggles this
    rotate_180   = False
    ribbon_locked = False
    locked_coords = None

    #   state 

    def on_mouse(event, x, y, flags, param):
        nonlocal rotate_180
        if not show_gui:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            if btn_bbox[0] <= x <= btn_bbox[2] and btn_bbox[1] <= y <= btn_bbox[3]:
                rotate_180 = not rotate_180
                print(f"[INFO] Camera rotation toggled. Active: {rotate_180}")

    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        fps_counter.tick()  #   must be first thing each frame

        success, frame = cap.read()
        if not success:
            print("[WARNING] Failed to capture frame, retrying...")
            continue

        frame = cv2.flip(frame, 1)
        if rotate_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        #   process hands (tracking  ribbon rendering) 
        orig_frame = frame.copy()   #  keep real-camera copy for reverse-mode ascii
        frame, clap_active, _, current_coords = process_hands(
            frame, tracker, renderer, config,
            show_gui=show_skeleton, clap_active=clap_active, time_val=time.time(),
            reverse_mode=reverse_mode,
            locked_coords=locked_coords,
        )

        #  reverse mode  no ribbon - full-frame ascii on black background
        if reverse_mode and not clap_active and not locked_coords:
            frame = ascii_processor.ascii_full_frame(orig_frame)

        #  gui rendering
        if show_gui:
            frame = renderer.draw_fps(frame, fps_counter.get_fps_string())
            #  render flip button and update its exact bounding box
            frame, bbox = renderer.draw_flip_button(frame, rotate_180)
            btn_bbox[0], btn_bbox[1] = bbox[0]
            btn_bbox[2], btn_bbox[3] = bbox[1]
            frame = renderer.draw_device_label(frame, device_label)

        cv2.imshow(window_name, frame)

        #  capture keyboard input (waitkeyex captures extended keys)
        key = cv2.waitKeyEx(1)
        if key != -1:
            ascii_key = key & 0xFF
            #  q  quit
            if ascii_key == ord("q"):
                break
            #  esc  toggle hand skeleton visibility
            if ascii_key == 27:
                show_skeleton = not show_skeleton
                print(f"[INFO] Skeleton visibility: {'ON' if show_skeleton else 'OFF'}")
            #  space bar  invert effect (reverse mode)
            if ascii_key == 32:
                reverse_mode = not reverse_mode
                print(f"[INFO] Space bar pressed! Reverse mode: {'ON (ribbon=real, no-ribbon=ASCII)' if reverse_mode else 'OFF'}")
            #  backspace  lock ribbon
            if ascii_key == 8:
                ribbon_locked = not ribbon_locked
                if ribbon_locked:
                    locked_coords = current_coords
                    print("[INFO] Ribbon locked.")
                else:
                    locked_coords = None
                    print("[INFO] Ribbon unlocked.")
            #  delete  toggle all gui (hud fps flip button device label)
            if key in [3014656, 46, 127, 65535, 0x2E0000]:
                show_gui = not show_gui
                print(f"[INFO] HUD visibility: {'ON' if show_gui else 'OFF'}")

def cleanup(
    cap: cv2.VideoCapture,
    tracker: HandTracker,
) -> None:
    """release all resources and close windows

    properly releases the camera capture device mediapipe tracker
    resources and all opencv windows

    args
        cap video capture device to release
        tracker hand tracker instance to close
    """

    cap.release()
    tracker.release()
    cv2.destroyAllWindows()
    print("[INFO] Resources released. Application closed.")

def main() -> None:
    """application entry point

    loads configuration initializes all components runs the
    main processing loop and performs cleanup on exit
    """

    print("Hand AR Tracker - Starting...")
    config = load_config()
    device = select_device(str(config.get("device", "auto")))
    device_label = get_device_label(device)
    cap = initialize_camera(config)
    tracker = initialize_tracker(config)
    renderer = initialize_renderer(config)
    fps_counter = FPSCounter(window_size=30)
    try:
        run_main_loop(
            cap, tracker, renderer,
            fps_counter, device_label, config,
        )
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        cleanup(cap, tracker)

if __name__ == "__main__":
    main()
