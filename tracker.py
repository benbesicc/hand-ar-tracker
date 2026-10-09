"""mediapipe hand tracking wrapper module

encapsulates the mediapipe hands solution to provide a clean interface
for hand detection landmark extraction and multi-hand tracking
with configurable confidence thresholds
"""

from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np

_MP_AVAILABLE = False
mp = None
mp_hands = None
mp_selfie = None

def _load_mediapipe_hands() -> bool:
    """import mediapipe hands from any supported module path

    returns
        true when mediapipe hands is available false otherwise
    """

    global mp, mp_hands, mp_selfie, _MP_AVAILABLE
    try:
        import mediapipe as mp_module
        mp = mp_module
    except ImportError:
        return False

    for loader in (
        lambda: (mp.solutions.hands, mp.solutions.selfie_segmentation),
        lambda: (
            __import__("mediapipe.python.solutions.hands", fromlist=["Hands"]),
            __import__("mediapipe.python.solutions.selfie_segmentation", fromlist=["SelfieSegmentation"])
        ),
    ):
        try:
            mp_hands, mp_selfie = loader()
            _MP_AVAILABLE = True
            return True
        except (AttributeError, ImportError, ModuleNotFoundError):
            continue
    return False

_load_mediapipe_hands()

from utils.math_utils import smooth_landmarks

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

LANDMARK_NAMES = [
    "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]

class HandTracker:
    """wrapper around mediapipe hands for hand detection and tracking"""

    def __init__(
        self,
        max_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.6,
        model_complexity: int = 1,
    ) -> None:
        """initialize the hand tracker with detection parameters"""
        if not _MP_AVAILABLE or mp_hands is None:
            raise RuntimeError(
                "MediaPipe Hands is not available. "
                "Run install.bat to install mediapipe==0.10.14."
            )

        self.max_hands: int = max_hands
        self.min_detection_confidence: float = min_detection_confidence
        self.min_tracking_confidence: float = min_tracking_confidence
        self.model_complexity: int = model_complexity

        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_hands,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        #  initialize selfiesegmentation model
        self.selfie = mp_selfie.SelfieSegmentation(model_selection=0) #  0 for generalfast model
        #  initialize facemesh solution for facial landmarks
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.previous_face_landmarks: List[Tuple[float, float]] = []
        #  store previous hand landmarks for smoothing (keyed by hand index)
        self.previous_landmarks: Dict[int, List[Tuple[float, float, float]]] = {}

        print(
            f"[INFO] MediaPipe Hands & SelfieSegmentation initialised "
            f"(model_complexity={model_complexity})."
        )

    def process_frame(self, frame: np.ndarray) -> Optional[Any]:
        """process a single video frame for hand detection"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.hands.process(rgb_frame)
        rgb_frame.flags.writeable = True
        return results

    def segment_frame(self, frame: np.ndarray) -> np.ndarray:
        """segment foreground (person) from background using mediapipe selfiesegmentation

        pipeline
          1 run mediapipe selfiesegmentation  raw float confidence map
          2 gaussian blur the confidence map to smooth sub-pixel noise
          3 temporal ema blend with the previous mask to eliminate flicker
          4 threshold at 055 confidence (slightly strict to cut noisy edges)
          5 morphological close with a large kernel  fills body holes
             (armpits gaps between arms clothing gaps etc)
          6 morphological open with a small kernel  removes isolated noise blobs
          7 final feather  dilate to recover any edge pixels cut by the threshold

        returns
            uint8 mask same spatial size as frame 255  foreground 0  background
        """

        h, w = frame.shape[:2]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.selfie.process(rgb_frame)
        rgb_frame.flags.writeable = True

        if results.segmentation_mask is None:
            self._prev_seg_mask = np.zeros((h, w), dtype=np.float32)
            return np.zeros((h, w), dtype=np.uint8)

        #   1 gaussian blur on raw float confidence map 
        #  softens pixel-level noise in the segmentation output before we threshold
        raw: np.ndarray = results.segmentation_mask.astype(np.float32)
        raw = cv2.GaussianBlur(raw, (11, 11), 0)

        #   2 temporal ema smoothing 
        #  blend 70  current  30  previous to damp per-frame flicker
        if not hasattr(self, "_prev_seg_mask") or self._prev_seg_mask.shape != raw.shape:
            self._prev_seg_mask = raw
        blended: np.ndarray = 0.70 * raw + 0.30 * self._prev_seg_mask
        self._prev_seg_mask = blended

        #   3 threshold 
        #  055  255  140 - slightly stricter than the old 128 so noisy edge
        #  pixels with low confidence are rejected rather than included
        thresh_val = int(0.55 * 255)
        mask_u8 = (blended * 255).clip(0, 255).astype(np.uint8)
        _, mask = cv2.threshold(mask_u8, thresh_val, 255, cv2.THRESH_BINARY)

        #   4 fill holes (large closing) 
        #  a 2121 ellipse kernel fills armpits gaps between arms and torso
        #  and other body cavities that the model rates as background
        fill_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, fill_kernel)

        #   5 remove isolated noise blobs (small opening) 
        #  a 77 open removes pepper-noise specks that survived the threshold
        noise_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, noise_kernel)

        #   6 recover lost edge pixels (slight dilation) 
        #  the stricter threshold can eat into the persons outline slightly
        #  a 3-pixel dilation compensates without reintroducing background noise
        edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, edge_kernel, iterations=1)

        return mask

    def extract_landmarks(
        self,
        results: Any,
    ) -> List[List[Tuple[float, float, float]]]:
        """extract landmark coordinates from mediapipe results"""
        if results is None or results.multi_hand_landmarks is None:
            self.previous_landmarks.clear()
            return []
        all_hands: List[List[Tuple[float, float, float]]] = []
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = self._parse_single_hand(hand_landmarks)
            all_hands.append(landmarks)
        return all_hands

    def _parse_single_hand(
        self,
        hand_landmarks: Any,
    ) -> List[Tuple[float, float, float]]:
        """parse landmarks for a single detected hand"""
        landmarks: List[Tuple[float, float, float]] = []
        for landmark in hand_landmarks.landmark:
            landmarks.append((landmark.x, landmark.y, landmark.z))
        return landmarks

    def extract_handedness(self, results: Any) -> List[str]:
        """extract hand labels (leftright) from mediapipe results"""
        if results is None or results.multi_handedness is None:
            return []
        labels: List[str] = []
        for handedness in results.multi_handedness:
            label = handedness.classification[0].label
            labels.append(label)
        return labels

    def get_smoothed_landmarks(
        self,
        hand_index: int,
        current_landmarks: List[Tuple[float, float, float]],
        smoothing_factor: float = 0.7,
    ) -> List[Tuple[float, float, float]]:
        """apply temporal smoothing to hand landmarks"""
        previous = self.previous_landmarks.get(hand_index)
        smoothed = smooth_landmarks(
            current_landmarks,
            previous,
            smoothing_factor,
        )
        self.previous_landmarks[hand_index] = smoothed
        return smoothed

    def get_landmark_count(self) -> int:
        """get the number of landmarks per hand"""
        return 21

    def get_connection_list(self) -> List[Tuple[int, int]]:
        """get the list of connections defining the hand skeleton"""
        return HAND_CONNECTIONS.copy()

    def release(self) -> None:
        """release mediapipe resources"""
        self.hands.close()
        self.selfie.close()
        self.face_mesh.close()
        self.previous_landmarks.clear()

    def extract_face_landmarks(self, frame: np.ndarray) -> List[Tuple[float, float]]:
        """extract facial landmarks using mediapipe facemesh

        returns a list of (x y) normalized coordinates for the detected face
        if no face is detected returns an empty list
        """

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.face_mesh.process(rgb)
        rgb.flags.writeable = True
        if results.multi_face_landmarks:
            #  use first face only
            face_landmarks = results.multi_face_landmarks[0]
            return [(lm.x, lm.y) for lm in face_landmarks.landmark]
        return []
