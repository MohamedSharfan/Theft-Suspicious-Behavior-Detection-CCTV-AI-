import numpy as np

def detect_clothing_color(frame, bbox):
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]

    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w, int(x2)))
    y2 = max(0, min(h, int(y2)))

    if x2 <= x1 or y2 <= y1:
        return "unknown"

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return "unknown"

    avg = crop.mean(axis=(0,1))

    b, g, r = avg

    if r > 120 and g < 80 and b < 80:
        return "red"

    if b > 120 and g < 80:
        return "blue"

    if r < 80 and g < 80 and b < 80:
        return "black"

    if r > 180 and g > 180 and b > 180:
        return "white"

    return "dark-coloured"