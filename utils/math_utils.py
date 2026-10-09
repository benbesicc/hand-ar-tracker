"""mathematical utility functions for hand landmark processing

provides distance calculations angle measurements vector operations
and coordinate transformation helpers used throughout the tracking pipeline
"""

import math
from typing import Tuple, List, Optional

import numpy as np

def calculate_distance(
    point_a: Tuple[float, float, float],
    point_b: Tuple[float, float, float],
) -> float:
    """calculate the euclidean distance between two 3d points

    args
        pointa first point as (x y z) tuple with normalized coordinates
        pointb second point as (x y z) tuple with normalized coordinates

    returns
        the euclidean distance between the two points as a float
    """

    dx = point_a[0] - point_b[0]
    dy = point_a[1] - point_b[1]
    dz = point_a[2] - point_b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)

def calculate_distance_2d(
    point_a: Tuple[float, float],
    point_b: Tuple[float, float],
) -> float:
    """calculate the euclidean distance between two 2d points

    args
        pointa first point as (x y) tuple
        pointb second point as (x y) tuple

    returns
        the euclidean distance between the two points as a float
    """

    dx = point_a[0] - point_b[0]
    dy = point_a[1] - point_b[1]
    return math.sqrt(dx * dx + dy * dy)

def calculate_angle(
    point_a: Tuple[float, float, float],
    point_b: Tuple[float, float, float],
    point_c: Tuple[float, float, float],
) -> float:
    """calculate the angle at pointb formed by points a b and c

    uses the dot product formula to compute the angle in degrees
    at the vertex pointb between rays ba and bc

    args
        pointa first endpoint as (x y z) tuple
        pointb vertex point as (x y z) tuple
        pointc second endpoint as (x y z) tuple

    returns
        the angle in degrees at pointb clamped between 0 and 180
    """

    vector_ba = (
        point_a[0] - point_b[0],
        point_a[1] - point_b[1],
        point_a[2] - point_b[2],
    )
    vector_bc = (
        point_c[0] - point_b[0],
        point_c[1] - point_b[1],
        point_c[2] - point_b[2],
    )
    dot_product = sum(a * b for a, b in zip(vector_ba, vector_bc))
    magnitude_ba = math.sqrt(sum(v * v for v in vector_ba))
    magnitude_bc = math.sqrt(sum(v * v for v in vector_bc))
    if magnitude_ba == 0.0 or magnitude_bc == 0.0:
        return 0.0
    cosine = dot_product / (magnitude_ba * magnitude_bc)
    cosine = max(-1.0, min(1.0, cosine))
    angle_radians = math.acos(cosine)
    return math.degrees(angle_radians)

def normalize_vector(
    vector: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """normalize a 3d vector to unit length

    args
        vector the input vector as (x y z) tuple

    returns
        a unit vector in the same direction returns (0 0 0) if
        the input vector has zero magnitude
    """

    magnitude = math.sqrt(sum(v * v for v in vector))
    if magnitude == 0.0:
        return (0.0, 0.0, 0.0)
    return (
        vector[0] / magnitude,
        vector[1] / magnitude,
        vector[2] / magnitude,
    )

def landmark_to_pixel(
    landmark_x: float,
    landmark_y: float,
    frame_width: int,
    frame_height: int,
) -> Tuple[int, int]:
    """convert normalized landmark coordinates to pixel coordinates

    mediapipe returns landmarks in normalized 0 1 coordinate space
    this function maps them to actual pixel positions in the frame

    args
        landmarkx normalized x coordinate from mediapipe (00 to 10)
        landmarky normalized y coordinate from mediapipe (00 to 10)
        framewidth width of the video frame in pixels
        frameheight height of the video frame in pixels

    returns
        a tuple of (pixelx pixely) as integers clamped to frame bounds
    """

    pixel_x = int(min(max(landmark_x * frame_width, 0), frame_width - 1))
    pixel_y = int(min(max(landmark_y * frame_height, 0), frame_height - 1))
    return (pixel_x, pixel_y)

def smooth_landmarks(
    current: List[Tuple[float, float, float]],
    previous: Optional[List[Tuple[float, float, float]]],
    smoothing_factor: float = 0.5,
) -> List[Tuple[float, float, float]]:
    """apply exponential moving average smoothing to landmark positions

    reduces jitter in hand landmark positions by blending the current
    frame landmarks with the previous frame using a weighted average

    args
        current list of current frame landmark positions as (x y z) tuples
        previous list of previous frame landmark positions or none if
            this is the first frame
        smoothingfactor weight for the current frame (00 to 10)
            higher values favor current frame data lower values
            produce smoother but more delayed tracking

    returns
        smoothed landmark positions as a list of (x y z) tuples
    """

    if previous is None or len(previous) != len(current):
        return current
    smoothed = []
    for curr, prev in zip(current, previous):
        sx = smoothing_factor * curr[0] + (1.0 - smoothing_factor) * prev[0]
        sy = smoothing_factor * curr[1] + (1.0 - smoothing_factor) * prev[1]
        sz = smoothing_factor * curr[2] + (1.0 - smoothing_factor) * prev[2]
        smoothed.append((sx, sy, sz))
    return smoothed

def vector_between(
    point_a: Tuple[float, float, float],
    point_b: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """compute the vector from pointa to pointb

    args
        pointa starting point as (x y z) tuple
        pointb ending point as (x y z) tuple

    returns
        the direction vector from pointa to pointb as (x y z) tuple
    """

    return (
        point_b[0] - point_a[0],
        point_b[1] - point_a[1],
        point_b[2] - point_a[2],
    )

def vector_magnitude(vector: Tuple[float, float, float]) -> float:
    """calculate the magnitude (length) of a 3d vector

    args
        vector input vector as (x y z) tuple

    returns
        the magnitude of the vector as a float
    """

    return math.sqrt(sum(v * v for v in vector))

def dot_product(
    vector_a: Tuple[float, float, float],
    vector_b: Tuple[float, float, float],
) -> float:
    """calculate the dot product of two 3d vectors

    args
        vectora first vector as (x y z) tuple
        vectorb second vector as (x y z) tuple

    returns
        the scalar dot product of the two vectors
    """

    return sum(a * b for a, b in zip(vector_a, vector_b))

def cross_product(
    vector_a: Tuple[float, float, float],
    vector_b: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """calculate the cross product of two 3d vectors

    args
        vectora first vector as (x y z) tuple
        vectorb second vector as (x y z) tuple

    returns
        the cross product vector as (x y z) tuple
    """

    return (
        vector_a[1] * vector_b[2] - vector_a[2] * vector_b[1],
        vector_a[2] * vector_b[0] - vector_a[0] * vector_b[2],
        vector_a[0] * vector_b[1] - vector_a[1] * vector_b[0],
    )

def midpoint(
    point_a: Tuple[float, float, float],
    point_b: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """calculate the midpoint between two 3d points

    args
        pointa first point as (x y z) tuple
        pointb second point as (x y z) tuple

    returns
        the midpoint as (x y z) tuple
    """

    return (
        (point_a[0] + point_b[0]) / 2.0,
        (point_a[1] + point_b[1]) / 2.0,
        (point_a[2] + point_b[2]) / 2.0,
    )

def clamp(value: float, min_val: float, max_val: float) -> float:
    """clamp a value to a specified range

    args
        value the input value to clamp
        minval the minimum allowed value
        maxval the maximum allowed value

    returns
        the clamped value guaranteed to be within minval maxval
    """

    return max(min_val, min(max_val, value))
