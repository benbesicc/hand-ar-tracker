"""opencv rendering module for hand tracking visualization

draws skeletal mesh connections and keypoint dots onto video frames
using opencv drawing primitives matching the clean skeleton-only
style (red circles  green lines) shown in the reference photo
"""

from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np

from utils.math_utils import landmark_to_pixel
from tracker import HAND_CONNECTIONS, LANDMARK_NAMES
from ascii_processor import AsciiProcessor as _AsciiProcessor

#  shared ascii processor for ribbon effect
_ribbon_ascii = _AsciiProcessor()

class HandRenderer:
    """renders hand tracking visualization overlays on video frames

    draws the skeletal mesh and keypoint markers onto opencv frames
    based on detected hand landmarks and configuration settings
    the default style matches the reference photo bright red filled
    circles on each landmark connected by bright green lines

    attributes
        skeletoncolor bgr color tuple for skeleton lines
        keypointcolor bgr color tuple for keypoint dots
        keypointradius pixel radius for keypoint circles
        skeletonthickness pixel thickness for skeleton lines
        showfps whether to display the fps counter overlay
        showgesturelabel whether to display detected gesture names
        showhandlabel whether to display leftright hand labels
        showdevicelabel whether to display the active device
    """

    def __init__(
        self,
        skeleton_color: Tuple[int, int, int] = (0, 255, 0),
        keypoint_color: Tuple[int, int, int] = (0, 0, 255),
        keypoint_radius: int = 8,
        skeleton_thickness: int = 2,
        show_fps: bool = True,
        show_coordinates: bool = False,
        show_gesture_label: bool = False,
        show_hand_label: bool = False,
        show_device_label: bool = True,
    ) -> None:
        """initialize the renderer with display configuration

        args
            skeletoncolor bgr color for skeleton connection lines
            keypointcolor bgr color for landmark keypoint dots
            keypointradius radius in pixels for keypoint circles
            skeletonthickness line thickness in pixels for skeleton
            showfps enable fps counter display in top-left corner
            showcoordinates enable landmark coordinate display
            showgesturelabel enable gesture name display near hand
            showhandlabel enable leftright label display
            showdevicelabel enable device label in top-right corner
        """

        self.skeleton_color = skeleton_color
        self.keypoint_color = keypoint_color
        self.keypoint_radius = keypoint_radius
        self.skeleton_thickness = skeleton_thickness
        self.show_fps = show_fps
        self.show_coordinates = show_coordinates
        self.show_gesture_label = show_gesture_label
        self.show_hand_label = show_hand_label
        self.show_device_label = show_device_label
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._font_scale = 0.5
        self._font_thickness = 1
        self._text_color = (255, 255, 255)
        self._bg_color = (0, 0, 0)

        #  physics state for jiggly ribbon
        self._last_hand_y = None
        self._last_time = None
        self._ribbon_pos = 0.0
        self._ribbon_vel = 0.0

    def draw_hand(
        self,
        frame: np.ndarray,
        landmarks: List[Tuple[float, float, float]],
        hand_label: str = "",
        gesture_label: str = "",
    ) -> np.ndarray:
        """draw a complete hand visualization on the frame

        renders the skeleton lines first (under the dots) then the
        filled keypoint circles on top for a clean layered look
        matching the reference photo

        args
            frame input video frame as bgr numpy array
            landmarks list of 21 (x y z) normalized coordinates
            handlabel leftright hand label string
            gesturelabel detected gesture name string

        returns
            the frame with hand visualization drawn on it
        """

        height, width = frame.shape[:2]
        pixel_coords = self._compute_pixel_coords(landmarks, width, height)

        #  draw skeleton lines first so dots render on top
        frame = self._draw_skeleton(frame, pixel_coords)
        #  draw filled keypoint circles on top of lines
        frame = self._draw_keypoints(frame, pixel_coords)

        if self.show_hand_label and hand_label:
            frame = self._draw_hand_label(frame, pixel_coords, hand_label)
        if self.show_gesture_label and gesture_label:
            frame = self._draw_gesture_label(frame, pixel_coords, gesture_label)
        if self.show_coordinates:
            frame = self._draw_coordinates(frame, landmarks, pixel_coords)
        return frame

    def _compute_pixel_coords(
        self,
        landmarks: List[Tuple[float, float, float]],
        width: int,
        height: int,
    ) -> List[Tuple[int, int]]:
        """convert normalized landmarks to pixel coordinates

        args
            landmarks list of (x y z) normalized coordinates
            width frame width in pixels
            height frame height in pixels

        returns
            list of (pixelx pixely) integer coordinate tuples
        """

        coords: List[Tuple[int, int]] = []
        for lm in landmarks:
            px, py = landmark_to_pixel(lm[0], lm[1], width, height)
            coords.append((px, py))
        return coords

    def _draw_skeleton(
        self,
        frame: np.ndarray,
        pixel_coords: List[Tuple[int, int]],
    ) -> np.ndarray:
        """draw the skeletal mesh connecting hand landmarks

        draws anti-aliased lines between connected landmarks as defined
        by the handconnections topology

        args
            frame input video frame as bgr numpy array
            pixelcoords list of pixel coordinate tuples for each landmark

        returns
            frame with skeleton lines drawn
        """

        for start_idx, end_idx in HAND_CONNECTIONS:
            if start_idx < len(pixel_coords) and end_idx < len(pixel_coords):
                start_point = pixel_coords[start_idx]
                end_point = pixel_coords[end_idx]
                cv2.line(
                    frame,
                    start_point,
                    end_point,
                    self.skeleton_color,
                    self.skeleton_thickness,
                    cv2.LINE_AA,
                )
        return frame

    def _draw_keypoints(
        self,
        frame: np.ndarray,
        pixel_coords: List[Tuple[int, int]],
    ) -> np.ndarray:
        """draw circular markers on each hand landmark

        draws filled circles at each of the 21 hand keypoint positions
        using the configured color and radius with a thin white border
        to improve visibility against any background

        args
            frame input video frame as bgr numpy array
            pixelcoords list of pixel coordinate tuples for each landmark

        returns
            frame with keypoint dots drawn
        """

        for point in pixel_coords:
            #  main filled dot
            cv2.circle(
                frame,
                point,
                self.keypoint_radius,
                self.keypoint_color,
                cv2.FILLED,
                cv2.LINE_AA,
            )
        return frame

    def _draw_hand_label(
        self,
        frame: np.ndarray,
        pixel_coords: List[Tuple[int, int]],
        label: str,
    ) -> np.ndarray:
        """draw the leftright hand label near the wrist

        args
            frame input video frame as bgr numpy array
            pixelcoords list of pixel coordinate tuples
            label hand label string (left or right)

        returns
            frame with hand label drawn
        """

        if len(pixel_coords) == 0:
            return frame
        wrist = pixel_coords[0]
        text_position = (wrist[0] - 20, wrist[1] + 30)
        frame = self._draw_text_with_background(
            frame, label, text_position, scale=0.7
        )
        return frame

    def _draw_gesture_label(
        self,
        frame: np.ndarray,
        pixel_coords: List[Tuple[int, int]],
        gesture: str,
    ) -> np.ndarray:
        """draw the detected gesture name above the hand

        args
            frame input video frame as bgr numpy array
            pixelcoords list of pixel coordinate tuples
            gesture detected gesture name string

        returns
            frame with gesture label drawn
        """

        if len(pixel_coords) < 10:
            return frame
        anchor = pixel_coords[9]
        text_position = (anchor[0] - 30, anchor[1] - 40)
        frame = self._draw_text_with_background(
            frame, gesture, text_position, scale=0.8
        )
        return frame

    def _draw_coordinates(
        self,
        frame: np.ndarray,
        landmarks: List[Tuple[float, float, float]],
        pixel_coords: List[Tuple[int, int]],
    ) -> np.ndarray:
        """draw coordinate labels next to key landmarks

        args
            frame input video frame as bgr numpy array
            landmarks list of (x y z) normalized coordinates
            pixelcoords list of pixel coordinate tuples

        returns
            frame with coordinate labels drawn
        """

        tip_indices = [4, 8, 12, 16, 20]
        for idx in tip_indices:
            if idx < len(landmarks) and idx < len(pixel_coords):
                lm = landmarks[idx]
                px, py = pixel_coords[idx]
                coord_text = f"({lm[0]:.2f},{lm[1]:.2f})"
                text_pos = (px + 10, py + 5)
                cv2.putText(
                    frame,
                    coord_text,
                    text_pos,
                    self._font,
                    0.35,
                    self._text_color,
                    1,
                    cv2.LINE_AA,
                )
        return frame

    def draw_fps(
        self,
        frame: np.ndarray,
        fps_text: str,
    ) -> np.ndarray:
        """draw the fps counter in the top-left corner

        args
            frame input video frame as bgr numpy array
            fpstext formatted fps string to display

        returns
            frame with fps overlay drawn
        """

        if not self.show_fps:
            return frame
        position = (10, 30)
        frame = self._draw_text_with_background(
            frame, fps_text, position, scale=0.7
        )
        return frame

    def draw_flip_button(
        self,
        frame: np.ndarray,
        rotate_180: bool,
    ) -> Tuple[np.ndarray, Tuple[Tuple[int, int], Tuple[int, int]]]:
        """draw a clickable button next to the fps counter to flip the camera

        args
            frame input video frame as bgr numpy array
            rotate180 current state of the 180-degree rotation

        returns
            a tuple of (frame (btntopleft btnbottomright)) where the
            coordinates define the bounding box of the interactive button
        """

        btn_text = "Flip 180: ON" if rotate_180 else "Flip 180: OFF"
        scale = 0.6
        thickness = 1
        text_size = cv2.getTextSize(btn_text, self._font, scale, thickness)[0]

        pad_x = 10
        pad_y = 6
        x_min = 140
        y_min = 10
        x_max = x_min + text_size[0] + 2 * pad_x
        y_max = y_min + text_size[1] + 2 * pad_y

        #  bgr red borderaccent if active grey if inactive
        overlay = frame.copy()
        bg_color = (0, 0, 180) if rotate_180 else (60, 60, 60)
        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), bg_color, cv2.FILLED)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        border_color = (0, 0, 255) if rotate_180 else (180, 180, 180)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), border_color, 1, cv2.LINE_AA)

        text_x = x_min + pad_x
        text_y = y_max - pad_y - 2
        cv2.putText(
            frame,
            btn_text,
            (text_x, text_y),
            self._font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        return frame, ((x_min, y_min), (x_max, y_max))

    def draw_device_label(
        self,
        frame: np.ndarray,
        device_text: str,
    ) -> np.ndarray:
        """draw the active device label in the top-right corner

        args
            frame input video frame as bgr numpy array
            devicetext device label string to display

        returns
            frame with device label drawn
        """

        if not self.show_device_label:
            return frame
        text_size = cv2.getTextSize(
            device_text, self._font, 0.6, 1
        )[0]
        width = frame.shape[1]
        position = (width - text_size[0] - 15, 30)
        frame = self._draw_text_with_background(
            frame, device_text, position, scale=0.6
        )
        return frame

    def draw_no_hands_message(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:
        """draw a message when no hands are detected

        args
            frame input video frame as bgr numpy array

        returns
            frame with the informational message drawn
        """

        height, width = frame.shape[:2]
        message = "No hands detected – show hands to camera"
        text_size = cv2.getTextSize(
            message, self._font, 0.7, 1
        )[0]
        x = (width - text_size[0]) // 2
        y = height - 30
        frame = self._draw_text_with_background(
            frame, message, (x, y), scale=0.7
        )
        return frame

    def _draw_text_with_background(
        self,
        frame: np.ndarray,
        text: str,
        position: Tuple[int, int],
        scale: float = 0.5,
    ) -> np.ndarray:
        """draw text with a semi-transparent dark background

        args
            frame input video frame as bgr numpy array
            text text string to render
            position (x y) position for the text baseline origin
            scale font scale multiplier for text size

        returns
            frame with background-highlighted text drawn
        """

        text_size = cv2.getTextSize(
            text, self._font, scale, self._font_thickness
        )[0]
        pad = 5
        x, y = position
        bg_start = (x - pad, y - text_size[1] - pad)
        bg_end = (x + text_size[0] + pad, y + pad)
        overlay = frame.copy()
        cv2.rectangle(overlay, bg_start, bg_end, self._bg_color, cv2.FILLED)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(
            frame,
            text,
            position,
            self._font,
            scale,
            self._text_color,
            self._font_thickness,
            cv2.LINE_AA,
        )
        return frame

    def draw_real_ribbon_border(
        self,
        frame: np.ndarray,
        landmarks_left: List[Tuple[float, float, float]],
        landmarks_right: List[Tuple[float, float, float]],
    ) -> np.ndarray:
        """reverse mode inside ribbon  real video outside  ascii art

        converts the full frame to ascii then restores the original
        pixels inside the ribbon polygon so the ribbon area is clear
        no border is drawn

        args
            frame input video frame as bgr numpy array
            landmarksleft left-hand landmark list
            landmarksright right-hand landmark list

        returns
            composite frame ascii outside ribbon real video inside
        """

        h_frame, w_frame = frame.shape[:2]
        coords_left  = self._compute_pixel_coords(landmarks_left,  w_frame, h_frame)
        coords_right = self._compute_pixel_coords(landmarks_right, w_frame, h_frame)

        if len(coords_left) < 21 or len(coords_right) < 21:
            #  fall back to full ascii if ribbon coords arent ready
            return _ribbon_ascii.ascii_full_frame(frame)

        left_top  = np.array(coords_left[8],  dtype=np.int32)
        left_bot  = np.array(coords_left[4],  dtype=np.int32)
        right_top = np.array(coords_right[8], dtype=np.int32)
        right_bot = np.array(coords_right[4], dtype=np.int32)

        poly = np.array([left_top, right_top, right_bot, left_bot], dtype=np.int32)

        #  build ribbon mask (255  inside ribbon)
        mask = np.zeros((h_frame, w_frame), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)

        #  convert entire frame to ascii art
        ascii_frame = _ribbon_ascii.ascii_full_frame(frame)

        #  paste real video back inside the ribbon area
        #  npwhere broadcasts (hw1) mask over 3 channels
        mask3 = mask[:, :, np.newaxis]
        result = np.where(mask3 > 0, frame, ascii_frame).astype(np.uint8)

        return result

    def draw_holographic_ribbon(
        self,
        frame: np.ndarray,
        landmarks_left: List[Tuple[float, float, float]],
        landmarks_right: List[Tuple[float, float, float]],
        time_val: float,
        full_person_mask: np.ndarray,
    ) -> np.ndarray:
        """apply the scanline effect on peopleforeground using mediapipes segmentation mask and space backdrop on walls matching 11"""
        h_frame, w_frame = frame.shape[:2]

        coords_left = self._compute_pixel_coords(landmarks_left, w_frame, h_frame)
        coords_right = self._compute_pixel_coords(landmarks_right, w_frame, h_frame)

        if len(coords_left) < 21 or len(coords_right) < 21:
            return frame

        left_top  = np.array(coords_left[8],  dtype=np.float32)
        left_bot  = np.array(coords_left[4],  dtype=np.float32)
        right_top = np.array(coords_right[8], dtype=np.float32)
        right_bot = np.array(coords_right[4], dtype=np.float32)

        poly = np.array([left_top, right_top, right_bot, left_bot], dtype=np.int32)
        x0 = max(0, int(poly[:, 0].min()))
        x1 = min(w_frame, int(poly[:, 0].max()) + 1)
        y0 = max(0, int(poly[:, 1].min()))
        y1 = min(h_frame, int(poly[:, 1].max()) + 1)
        if x1 <= x0 or y1 <= y0:
            return frame

        bw, bh = x1 - x0, y1 - y0

        #  create localized masks for ribbon pane
        poly_local = poly - np.array([x0, y0], dtype=np.int32)
        mask = np.zeros((bh, bw), dtype=np.uint8)
        cv2.fillPoly(mask, [poly_local], 255)

        #   1 create space backdrop (navy background  starfield) 
        pane = np.zeros((bh, bw, 3), dtype=np.uint8)
        pane[:] = (32, 28, 24) #  dark slatenavy space backing (matching 1st photo)

        #  add a much sparser space background starfield (matching the photo)
        np.random.seed(42)
        noise = np.random.rand(bh, bw)
        stars_mask = (noise > 0.9985)
        pane[stars_mask] = (240, 240, 255)

        #   2 crop the high-quality mediapipe segmentation mask 
        person_mask_global = full_person_mask[y0:y1, x0:x1]

        #  ensure its thresholded properly and limited inside the ribbon pane
        _, person_mask = cv2.threshold(person_mask_global, 128, 255, cv2.THRESH_BINARY)
        person_mask = cv2.bitwise_and(person_mask, mask)

        #   3 render blue glowing scanline particle effect on the person silhouette 
        y_indices, x_indices = np.where(person_mask > 0)
        if len(x_indices) > 0:
            #  multi-frequency scanline pattern
            scanline = np.abs(np.sin(y_indices * 0.45 - time_val * 12.0))

            #  seed-less random noise generator per frame to make points dynamically buzzvibrate
            vibe_x = (np.random.randn(len(x_indices)) * 2.0).astype(np.int32)
            vibe_y = (np.random.randn(len(y_indices)) * 0.8).astype(np.int32)

            #  wavy horizontal distortion mapping (11 with photo)
            wave_displacement = np.sin(y_indices * 0.15 + time_val * 8.0) * 4.0

            #  apply displacement  vibration noise
            warped_x = np.clip(x_indices + wave_displacement.astype(np.int32) + vibe_x, 0, bw - 1)
            warped_y = np.clip(y_indices + vibe_y, 0, bh - 1)

            #  restrict points so we only draw 40 of the pixels as individual point-cloud particles
            rand_filter = np.random.rand(len(x_indices))
            draw_mask = (rand_filter > 0.60) & (scanline > 0.3)

            wx = warped_x[draw_mask]
            wy = warped_y[draw_mask]

            #  map bluecyan intensities
            intensity = scanline[draw_mask] * 0.8 + 0.2
            blue_val = (255 * intensity).astype(np.uint8)
            cyan_val = (210 * intensity).astype(np.uint8)

            pane[wy, wx, 0] = np.maximum(pane[wy, wx, 0], blue_val)
            pane[wy, wx, 1] = np.maximum(pane[wy, wx, 1], cyan_val)
            pane[wy, wx, 2] = np.maximum(pane[wy, wx, 2], 0)

        #   4 convert the pane to ascii before blending 
        pane = _ribbon_ascii.ascii_full_frame(pane, color=(255, 255, 255))

        #  mask the ascii pane so only the persons silhouette contains text and the rest of the pane is plain black (0 0 0)
        pane = cv2.bitwise_and(pane, pane, mask=person_mask)

        #   5 copy the ascii pane directly over the roi to keep the background plain black 
        roi = frame[y0:y1, x0:x1]
        np.copyto(roi, pane, where=mask[:, :, None] > 0)

        #   5 glowing white outer ribbon border 
        contours_ribbon, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(roi, contours_ribbon, -1, (255, 255, 255), 2, cv2.LINE_AA) #  white boundary
        cv2.drawContours(roi, contours_ribbon, -1, (255, 255, 255), 1, cv2.LINE_AA) #  outer white hairline

        return frame
