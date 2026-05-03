#this file is for translating videos into numbers(to intelligence)


import numpy as np
from datetime import datetime

class FeatureExtractor:
    def __init__(self):
        self.history = {}

    def update(self, track_id, bbox):
        x, y, w, h = bbox
        center = np.array([x + w/2, y + h/2])

        if track_id not in self.history:
            self.history[track_id] = {
                "positions" : [],
                "speeds" : [],
                "angles": [],
                "start_time": datetime.now()
            }

        person = self.history[track_id]
        person["positions"].append(center)

        speed = 0
        direction = 0
        acceleration = 0

        #speed
        if len(person["positions"]) >= 2:
            prev = person["positions"][-2]
            curr = person["positions"][-1]
            speed = np.linalg.norm(curr - prev)
            person["speeds"].append(speed)

        #direction
        if len(person["positions"]) >= 3:
            v1 = person["positions"][-2] - person["positions"][-3]
            v2 = person["positions"][-1] - person["positions"][-2]

            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)

            if norm_v1 > 0 and norm_v2 > 0:
                cos_theta = np.dot(v1,v2) / (norm_v1 * norm_v2)
                angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))

                person["angles"].append(angle)
        
        #acceleration
        if len(person["speeds"]) >= 2:
            acceleration = person["speeds"][-1] - person["speeds"][-2]
            
        time_in_zone = (datetime.now() - person["start_time"]).total_seconds()

        avg_angle = np.mean(person["angles"]) if person["angles"] else 0


        return{
            "speed" : speed,
            "angle" : avg_angle,
            "acceleration" : acceleration,
            "time_in_zone" : time_in_zone
        }
