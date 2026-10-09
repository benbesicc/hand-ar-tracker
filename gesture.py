"""rule-based gesture recognition from hand landmark geometry

detects gestures by analyzing finger extension states and relative
distances between specific hand landmarks supports pinch fist
open hand peace sign and thumbs up gestures
"""

from typing import List, Tuple, Optional, Dict

from utils.math_utils import calculate_distance, calculate_angle

#  mediapipe hand landmark indices for reference
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

#  finger tip and pip index pairs for extension checking
FINGER_TIP_INDICES = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_PIP_INDICES = [THUMB_IP, INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]
FINGER_MCP_INDICES = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]

def recognize_gesture(
    landmarks: List[Tuple[float, float, float]],
    pinch_threshold: float = 0.05,
    fist_threshold: float = 0.85,
) -> str:
    """identify the current hand gesture from landmark positions

    analyzes the 21 hand landmarks to determine which gesture is
    being performed checks gestures in priority order pinch first
    then fist thumbs up peace sign and finally open hand

    args
        landmarks list of 21 (x y z) tuples representing hand
            landmark positions in normalized coordinates
        pinchthreshold maximum distance between thumb tip and index
            tip to register as a pinch gesture
        fistthreshold minimum ratio of curled fingers to total
            fingers required to register as a fist

    returns
        a string label for the detected gesture or unknown if
        no known gesture pattern is matched
    """

    if len(landmarks) != 21:
        return "Unknown"
    finger_states = get_finger_states(landmarks)
    if _is_pinch(landmarks, pinch_threshold):
        return "Pinch"
    if _is_fist(finger_states, fist_threshold):
        return "Fist"
    if _is_thumbs_up(finger_states, landmarks):
        return "Thumbs Up"
    if _is_peace_sign(finger_states):
        return "Peace"
    if _is_open_hand(finger_states):
        return "Open Hand"
    return "Unknown"

def get_finger_states(
    landmarks: List[Tuple[float, float, float]],
) -> List[bool]:
    """determine the extendedcurled state of each finger

    for the thumb uses a horizontal distance comparison relative
    to the palm for other fingers compares tip position to pip
    joint position along the y-axis

    args
        landmarks list of 21 (x y z) tuples for hand landmarks

    returns
        a list of 5 booleans corresponding to thumb index middle
        ring pinky true means the finger is extended
    """

    states: List[bool] = []
    thumb_extended = _is_thumb_extended(landmarks)
    states.append(thumb_extended)
    for i in range(1, 5):
        tip = landmarks[FINGER_TIP_INDICES[i]]
        pip_joint = landmarks[FINGER_PIP_INDICES[i]]
        extended = tip[1] < pip_joint[1]
        states.append(extended)
    return states

def _is_thumb_extended(
    landmarks: List[Tuple[float, float, float]],
) -> bool:
    """check if the thumb is extended outward from the palm

    uses the horizontal distance between the thumb tip and the
    thumb cmc joint compared to the index finger mcp position

    args
        landmarks list of 21 (x y z) tuples for hand landmarks

    returns
        true if the thumb appears to be extended outward
    """

    thumb_tip = landmarks[THUMB_TIP]
    thumb_mcp = landmarks[THUMB_MCP]
    index_mcp = landmarks[INDEX_MCP]
    thumb_tip_dist = abs(thumb_tip[0] - index_mcp[0])
    thumb_mcp_dist = abs(thumb_mcp[0] - index_mcp[0])
    return thumb_tip_dist > thumb_mcp_dist

def _is_pinch(
    landmarks: List[Tuple[float, float, float]],
    threshold: float,
) -> bool:
    """detect a pinch gesture between thumb and index finger

    measures the 3d distance between the thumb tip and index finger
    tip landmarks and compares against the threshold

    args
        landmarks list of 21 (x y z) tuples for hand landmarks
        threshold maximum distance to consider as a pinch

    returns
        true if the thumb and index tips are closer than threshold
    """

    thumb_tip = landmarks[THUMB_TIP]
    index_tip = landmarks[INDEX_TIP]
    distance = calculate_distance(thumb_tip, index_tip)
    return distance < threshold

def _is_fist(
    finger_states: List[bool],
    threshold: float,
) -> bool:
    """detect a fist gesture where most fingers are curled

    counts the number of curled (non-extended) fingers and compares
    the ratio to the threshold

    args
        fingerstates list of 5 booleans for finger extension states
        threshold minimum ratio of curled fingers (00 to 10)

    returns
        true if enough fingers are curled to constitute a fist
    """

    curled_count = sum(1 for state in finger_states if not state)
    ratio = curled_count / len(finger_states)
    return ratio >= threshold

def _is_thumbs_up(
    finger_states: List[bool],
    landmarks: List[Tuple[float, float, float]],
) -> bool:
    """detect a thumbs up gesture

    requires the thumb to be extended and pointing upward while
    all other fingers are curled into the palm

    args
        fingerstates list of 5 booleans for finger extension states
        landmarks list of 21 (x y z) tuples for hand landmarks

    returns
        true if the hand is showing a thumbs up gesture
    """

    thumb_extended = finger_states[0]
    others_curled = all(not state for state in finger_states[1:])
    if not (thumb_extended and others_curled):
        return False
    thumb_tip = landmarks[THUMB_TIP]
    thumb_mcp = landmarks[THUMB_MCP]
    thumb_points_up = thumb_tip[1] < thumb_mcp[1]
    return thumb_points_up

def _is_peace_sign(finger_states: List[bool]) -> bool:
    """detect a peace sign (v sign) gesture

    requires the index and middle fingers to be extended while
    the ring and pinky fingers are curled thumb state is ignored

    args
        fingerstates list of 5 booleans for finger extension states

    returns
        true if the hand is showing a peace sign
    """

    index_extended = finger_states[1]
    middle_extended = finger_states[2]
    ring_curled = not finger_states[3]
    pinky_curled = not finger_states[4]
    return (
        index_extended
        and middle_extended
        and ring_curled
        and pinky_curled
    )

def _is_open_hand(finger_states: List[bool]) -> bool:
    """detect an open hand gesture with all fingers extended

    requires all five fingers including the thumb to be extended

    args
        fingerstates list of 5 booleans for finger extension states

    returns
        true if all fingers are extended in an open hand pose
    """

    return all(finger_states)

def get_extended_finger_count(
    finger_states: List[bool],
) -> int:
    """count the number of extended fingers

    args
        fingerstates list of 5 booleans for finger extension states

    returns
        integer count of extended fingers (0 to 5)
    """

    return sum(1 for state in finger_states if state)

def get_gesture_confidence(
    landmarks: List[Tuple[float, float, float]],
    gesture_name: str,
    pinch_threshold: float = 0.05,
) -> float:
    """calculate a confidence score for a specific gesture

    provides a rough confidence metric based on how clearly the
    landmark geometry matches the expected gesture pattern

    args
        landmarks list of 21 (x y z) tuples for hand landmarks
        gesturename name of the gesture to evaluate
        pinchthreshold distance threshold for pinch detection

    returns
        a confidence score between 00 and 10 where 10 indicates
        a very clear match for the specified gesture
    """

    if len(landmarks) != 21:
        return 0.0
    finger_states = get_finger_states(landmarks)
    if gesture_name == "Pinch":
        return _pinch_confidence(landmarks, pinch_threshold)
    if gesture_name == "Fist":
        return _fist_confidence(finger_states)
    if gesture_name == "Open Hand":
        return _open_hand_confidence(finger_states)
    return 0.0

def _pinch_confidence(
    landmarks: List[Tuple[float, float, float]],
    threshold: float,
) -> float:
    """calculate confidence score for a pinch gesture

    args
        landmarks list of 21 (x y z) tuples for hand landmarks
        threshold distance threshold for pinch detection

    returns
        confidence score between 00 and 10
    """

    distance = calculate_distance(
        landmarks[THUMB_TIP],
        landmarks[INDEX_TIP],
    )
    if distance >= threshold:
        return 0.0
    return 1.0 - (distance / threshold)

def _fist_confidence(finger_states: List[bool]) -> float:
    """calculate confidence score for a fist gesture

    args
        fingerstates list of 5 booleans for finger extension states

    returns
        confidence score between 00 and 10
    """

    curled = sum(1 for s in finger_states if not s)
    return curled / len(finger_states)

def _open_hand_confidence(finger_states: List[bool]) -> float:
    """calculate confidence score for an open hand gesture

    args
        fingerstates list of 5 booleans for finger extension states

    returns
        confidence score between 00 and 10
    """

    extended = sum(1 for s in finger_states if s)
    return extended / len(finger_states)
