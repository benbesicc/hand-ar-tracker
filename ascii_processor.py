"""ascii art face overlay - fast vectorised renderer

approach
--------
 pre-renders all character tiles as numpy arrays at startup
 uses clahe contrast equalisation so the full range of characters
  is always used (no flat dark-dots look)
 all characters in the gradient are visible (no spaceempty cells)
 assembles the full ascii canvas via numpy tiling - no python loops

colors pure ffffff characters on 000000 background
"""

from typing import List, Tuple

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

#  base character pool covering letters numbers and requested special symbols
_CHAR_POOL = " .:-=+*!%#@$&?abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
_CW = 6    #  character cell width  (pixels)
_CH = 10   #  character cell height (pixels)

#  clahe for local contrast enhancement before mapping
_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

#  pre-render and sort every character by density at startup
def _build_char_tiles() -> Tuple[np.ndarray, int]:
    """pre-renders character pool and returns (sortedtiles numchars)"""
    chars = list(_CHAR_POOL)
    if not _PIL_OK:
        #  fallback empty tiles if pil is missing
        return np.zeros((len(chars), _CH, _CW), dtype=np.uint8), len(chars)

    font = None
    for name in ("cour.ttf", "consola.ttf", "lucon.ttf", "DejaVuSansMono.ttf"):
        try:
            font = ImageFont.truetype(name, 9)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()

    rendered = []
    for ch in chars:
        img = Image.new("L", (_CW, _CH), 0)
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), ch, fill=255, font=font)
        tile = np.array(img)
        #  density is sum of pixel brightness values
        density = int(np.sum(tile))
        rendered.append((density, tile))

    #  sort characters by brightness density (darkest first brightest last)
    rendered.sort(key=lambda x: x[0])

    sorted_tiles = np.array([item[1] for item in rendered], dtype=np.uint8)
    return sorted_tiles, len(rendered)

_CHAR_TILES, _N = _build_char_tiles()  #  built once at import

#  core vectorised renderer
def _fast_ascii(grey: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """convert a greyscale image to an ascii greyscale canvas

    applies clahe first so the full character range is always used
    regardless of ambient lighting

    returns (outh outw) uint8 - white chars on black
    """

    cols = out_w // _CW
    rows = out_h // _CH
    if cols < 1 or rows < 1:
        return np.zeros((out_h, out_w), dtype=np.uint8)

    #  enhance local contrast so dark faces still show varied chars
    eq = _CLAHE.apply(grey)

    #  downsample to char grid
    small = cv2.resize(eq, (cols, rows), interpolation=cv2.INTER_AREA)

    #  map 0-255 - 0-(n-1) char index
    idx = (small.astype(np.float32) / 255.0 * (_N - 1)).astype(np.int32)
    idx = np.clip(idx, 0, _N - 1)

    #  assemble via advanced indexing  reshape (no python loops)
    #  chartilesidx - (rows cols ch cw)
    canvas_tiles = _CHAR_TILES[idx]

    canvas_h = rows * _CH
    canvas_w = cols * _CW
    #  (rows cols ch cw) - (rows ch cols cw) - (rowsch colscw)
    canvas = canvas_tiles.transpose(0, 2, 1, 3).reshape(canvas_h, canvas_w)

    if canvas_h < out_h or canvas_w < out_w:
        padded = np.zeros((out_h, out_w), dtype=np.uint8)
        padded[:canvas_h, :canvas_w] = canvas
        canvas = padded

    return canvas

def _grey_to_bgr_white(grey: np.ndarray) -> np.ndarray:
    """greyscale ascii canvas - bgr with pure white chars (ffffff on 000000)"""
    return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)

#  public class
class AsciiProcessor:
    """real-time ascii art overlay - ffffff on 000000"""

    def render_face(
        self,
        frame: np.ndarray,
        face_landmarks: List[Tuple[float, float]],
    ) -> np.ndarray:
        """overwrite the face region with live ascii art

        characters update every frame based on actual face brightness
        clahe ensures the full character range is used even in low light

        parameters
        ----------
        frame           bgr video frame (modified in-place returned)
        facelandmarks  (xnorm ynorm) pairs from mediapipe facemesh
        """

        if not face_landmarks:
            return frame

        h, w = frame.shape[:2]

        #  face convex hull  bounding rect
        pts = np.array(
            [(int(x * w), int(y * h)) for x, y in face_landmarks],
            dtype=np.int32,
        )
        hull = cv2.convexHull(pts)
        rx, ry, rw, rh = cv2.boundingRect(hull)
        rx, ry = max(0, rx), max(0, ry)
        rw = min(rw, w - rx)
        rh = min(rh, h - ry)
        if rw < _CW * 2 or rh < _CH * 2:
            return frame

        #  hull mask in roi space
        face_mask = np.zeros((rh, rw), dtype=np.uint8)
        cv2.fillConvexPoly(face_mask, hull - np.array([rx, ry]), 255)

        #  sample roi
        roi = frame[ry:ry + rh, rx:rx + rw]
        grey = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        #  build ascii canvas
        ascii_grey = _fast_ascii(grey, rw, rh)
        ascii_bgr = _grey_to_bgr_white(ascii_grey)

        #  paste inside hull only
        mask3 = cv2.merge([face_mask, face_mask, face_mask])
        np.copyto(roi, ascii_bgr, where=mask3 > 0)

        return frame

    def ascii_full_frame(
        self,
        frame: np.ndarray,
        color: Tuple[int, int, int] = (255, 255, 255),
    ) -> np.ndarray:
        """convert a full bgr frame to ascii art on black background

        used to render the holographic ribbon in ascii style
        returns a new bgr array of the same shape
        """

        h, w = frame.shape[:2]
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ascii_grey = _fast_ascii(grey, w, h)
        return _grey_to_bgr_white(ascii_grey)
