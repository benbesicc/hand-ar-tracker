"""utility modules for the hand ar tracker system

this package contains helper modules for fps calculation
mathematical operations and compute device management
"""

from utils.fps_counter import FPSCounter
from utils.math_utils import (
    calculate_distance,
    calculate_angle,
    normalize_vector,
    landmark_to_pixel,
    smooth_landmarks,
)
from utils.device_utils import (
    list_available_devices,
    select_device,
    get_device_label,
)

__all__ = [
    "FPSCounter",
    "calculate_distance",
    "calculate_angle",
    "normalize_vector",
    "landmark_to_pixel",
    "smooth_landmarks",
    "list_available_devices",
    "select_device",
    "get_device_label",
]
