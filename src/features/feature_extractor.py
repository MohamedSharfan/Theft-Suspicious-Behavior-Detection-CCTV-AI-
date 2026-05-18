#this file is for translating videos into numbers(to intelligence)


import numpy as np
from datetime import datetime
import cv2


def is_frame_blurry(frame, threshold=80):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold
class FeatureExtractor:
    def rolling_mean(self, values, window=5):
        if len(values) == 0:
            return 0
        
        values = values[-window:]
        return float(np.mean(values))

    def __init__(self):
        self.history = {}
        self.window_size = 30
        self.window_buffer = {}


    def update(self, track_id, bbox, frame_shape, pose_landmarks=None):

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
                "acceleration":[],
                "hand_distances":[],
                "stop_count": 0,
                "is_stopped": False,
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

            # if movement < 0.002:
            #     speed = 0
            #     person["stop_count"] += 1

            STOP_THRESHOLD = 0.002

            if movement < STOP_THRESHOLD:
                speed = 0

                if not person["is_stopped"]:
                    person["stop_count"] += 1
                    person["is_stopped"] = True
            else:
                raw_speed = movement * fps
                speed = raw_speed
                person["is_stopped"] = False

            person["speeds"].append(speed)

            # def smooth(values, alpha=0.7):
            #     if len(values) < 2:
            #         return values[-1]
            #     return alpha * values[-1] + (1- alpha) * values[-2]

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

            person["acceleration"].append(acceleration)

            if len(person["acceleration"]) > MAX_HISTORY:
                person["acceleration"].pop(0)
            

        hand_distance = 0
        crouching = False

        if pose_landmarks:
            try:
                LEFT_WRIST = 15
                RIGHT_WRIST = 16
                LEFT_SHOULDER = 11
                RIGHT_SHOULDER = 12

                lw = np.array(pose_landmarks[LEFT_WRIST])
                rw = np.array(pose_landmarks[RIGHT_WRIST])

                ls = np.array(pose_landmarks[LEFT_SHOULDER])
                rs = np.array(pose_landmarks[RIGHT_SHOULDER])

                wrist_center = (lw + rw)/2
                shoulder_center = (ls + rs) /2

                hand_distance = np.linalg.norm(wrist_center - shoulder_center)

                person["hand_distances"].append(hand_distance)

                if len(person["hand_distances"]) > MAX_HISTORY:
                    person["hand_distances"].pop(0)

                shoulder_y = shoulder_center[1]
                wrist_y = wrist_center[1]

                crouching = wrist_y > shoulder_y + 0.25

            except:
                pass


        time_in_zone = (datetime.now() - person["start_time"]).total_seconds()
        time_in_zone = min(time_in_zone, 60)

        angle = person["angles"][-1] if person["angles"] else 0

        avg_speed = self.rolling_mean(person["speeds"])
        avg_angle = self.rolling_mean(person["angles"])
        avg_acceleration = self.rolling_mean(person["acceleration"])
        avg_hand_distance = self.rolling_mean(person["hand_distances"])

        
        MIN_VALID_SPEED = 0.02
        if speed < MIN_VALID_SPEED and acceleration == 0 and time_in_zone < 2:
            return None

        feature_vector = [
            speed,
            angle,
            acceleration,
            time_in_zone,
            hand_distance,
            person["stop_count"],
            int(crouching),
        ]

        if track_id not in self.window_buffer:
            self.window_buffer[track_id] = []

        self.window_buffer[track_id].append(feature_vector)

        if len(self.window_buffer[track_id]) > self.window_size:
            self.window_buffer[track_id].pop(0)


        # return {
        #     "speed": avg_speed,
        #     "angle": avg_angle,
        #     "acceleration": avg_acceleration,
        #     "time_in_zone": time_in_zone,
        #     "hand_distance": avg_hand_distance,
        #     "stop_count": person["stop_count"],
        #     "crouching": crouching,
        #     "hand_speed": 0,
        #     "body_expansion": 0
        # }
        return None
    
    def get_window_features(self, track_id):
        if track_id not in self.window_buffer:
            return None
        
        window = self.window_buffer[track_id]

        if len(window) < self.window_size:
            return None
        
        arr = np.array(window)

        speed_mean = np.mean(arr[:, 0])
        speed_std = np.std(arr[:, 0])

        ang_mean = np.mean(arr[:, 1])
        ang_std = np.std(arr[:, 1])

        acc_mean = np.mean(arr[:, 2])
        acc_std =  np.std(arr[:, 2])

        time_mean = np.mean(arr[:, 3])

        hand_mean = np.mean(arr[:, 4])
        hand_std = np.std(arr[:, 4])

        stop_mean = np.mean(arr[:, 5])
        stop_std = np.std(arr[:, 5])

        crouch_ratio = np.mean(arr[:, 6])



        return{
            "speed_mean": speed_mean,
            "speed_std":speed_std,

            "angle_mean": ang_mean,
            "angle_std": ang_std,

            "acc_mean": acc_mean,
            "acc_std": acc_std,

            "time_mean": time_mean,

            "hand_mean": hand_mean,
            "hand_std": hand_std,

            "stop_mean": stop_mean,
            "stop_std": stop_std,
            
            "crouch_ratio": crouch_ratio,

            "hand_speed": 0,
            "body_expansion": 0
        }
