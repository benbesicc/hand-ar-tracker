import cv2
import numpy as np

def detect_hand(frame: np.ndarray) -> list:
    """detect hand landmarks using a simple opencv heuristic

    this fallback is used when mediapipe is unavailable the implementation
    1 convert the bgr frame to hsv color space
    2 apply a skincolor mask using a broad hsv range
    3 find contours and select the largest one assuming it is the hand
    4 compute the convex hull of the contour
    5 sample up to 21 points from the hull (or evenly from the bounding box) and
       return them as normalized (x y 0) coordinates where x and y are in
       0 1 relative to the frame widthheight
    if no suitable contour is found an empty list is returned
    """

    #  step 1 convert to hsv
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #  step 2 skin color mask (this range works reasonably for many skin tones)
    lower = np.array([0, 30, 60], dtype=np.uint8)
    upper = np.array([20, 150, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    #  apply some morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    #  step 3 find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    #  choose the largest contour assuming it is the hand
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 1000:
        #  too small to be a hand
        return []
    #  step 4 compute convex hull
    hull = cv2.convexHull(largest, returnPoints=True)
    hull_points = hull.squeeze().tolist()
    if not isinstance(hull_points[0], list):
        #  when hull has a single point ensure its a list of points
        hull_points = [hull_points]
    #  step 5 sample up to 21 points
    num_points = min(21, len(hull_points))
    #  evenly sample indices
    indices = np.linspace(0, len(hull_points) - 1, num=num_points, dtype=int)
    sampled = [hull_points[i] for i in indices]
    height, width = frame.shape[:2]
    #  normalize coordinates and add dummy z0
    normalized = [(x / width, y / height, 0.0) for x, y in sampled]
    #  if fewer than 21 points pad with the last point
    while len(normalized) < 21:
        normalized.append(normalized[-1])
    return [normalized]
