import os
from datetime import datetime

import cv2


def save_clip(frame, track_id):
    os.makedirs("clips", exist_ok=True)

    filename = f"clips/{track_id}_{datetime.now().timestamp()}.jpg"

    cv2.imwrite(filename, frame)

    return filename
