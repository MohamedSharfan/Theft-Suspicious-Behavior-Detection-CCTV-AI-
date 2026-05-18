import numpy as np

def detect_clothing_color(frame, bbox):
    x1, y1, x2, y2 = bbox

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