#this file is for translating videos into numbers(to intelligence)


import numpy as np
from datetime import datetime

class FeatureExtractor:
    def __init__(self):
        self.history = {}

    def update(self, track_id, bbox, frame_shape):

        TIMEOUT = 2

        x, y, w, h = bbox
        H, W = frame_shape[:2]

        center = np.array([(x + w/2)/W, (y + h/2)/H])

        to_delete = []

        for tid, data in self.history.items():
            if (datetime.now() - data["last_seen"]).total_seconds() > TIMEOUT:
                to_delete.append(tid)
            
        for tid in to_delete:
            del self.history[tid]


        if track_id not in self.history:
            self.history[track_id] = {
                "positions" : [],
                "speeds" : [],
                "angles": [],
                "start_time": datetime.now(),
                "last_seen": datetime.now()
            }

        person = self.history[track_id]
        person["last_seen"] = datetime.now()

        MAX_HISTORY = 30

        person["positions"].append(center)
        if len(person["positions"]) > MAX_HISTORY:
            person["positions"].pop(0)


        alpha = 0.7
        if len(person["positions"]) > 1:
            prev = person["positions"][-2]
            curr = person["positions"][-1]

            smoothed = alpha * prev + (1 - alpha) * curr
            person["positions"][-1] = smoothed

        speed = 0
        acceleration = 0

        #speed
        fps = 30
        if len(person["positions"]) >= 2:
            prev = person["positions"][-2]
            curr = person["positions"][-1]
            movement = np.linalg.norm(curr - prev)

            if movement < 0.002:
                speed = 0
                return None
            else:
                speed = movement * fps
            
            person["speeds"].append(speed)
            if len(person["speeds"]) > MAX_HISTORY:
                person["speeds"].pop(0)

        #direction
        if len(person["positions"]) >= 3:
            v1 = person["positions"][-2] - person["positions"][-3]
            v2 = person["positions"][-1] - person["positions"][-2]

            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)

            if norm_v1 > 0 and norm_v2 > 0:
                cos_theta = np.dot(v1,v2) / (norm_v1 * norm_v2)
                angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))

                angle = angle / np.pi

                person["angles"].append(angle)
                if len(person["angles"]) > MAX_HISTORY:
                    person["angles"].pop(0)
        
        #acceleration
        if len(person["speeds"]) >= 3:
            s1 = person["speeds"][-1]
            s2 = person["speeds"][-2]
            # s3 = person["speeds"][-3]

            # acceleration = (s1 - s2 + s2 - s3) / 2
            acceleration = s1 - s2
            
        time_in_zone = (datetime.now() - person["start_time"]).total_seconds()
        time_in_zone = min(time_in_zone, 60)

        angle = person["angles"][-1] if person["angles"] else 0
        
        if speed < 0.01:
            return None

        return{
            "speed" : speed,
            "angle" : angle,
            "acceleration" : acceleration,
            "time_in_zone" : time_in_zone
        }
