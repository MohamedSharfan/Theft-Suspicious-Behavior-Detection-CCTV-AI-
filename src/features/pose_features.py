import numpy as np

class PoseFeatureExtractor:
    def __init__(self):
        self.history = {}

    def extract(self, track_id, landmarks):
        if landmarks is None:
            return None
        
        if track_id not in self.history:
            self.history[track_id] = {
                "left_wrist":[],
                "right_wrist":[],
                "shoulder_width":[]
            }

        person = self.history[track_id]

        # mediapipe landmark indexes default num,bers
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_WRIST = 15
        RIGHT_WRIST = 16

        lw = np.array(landmarks[LEFT_WRIST])
        rw = np.array(landmarks[RIGHT_WRIST])
        ls = np.array(landmarks[LEFT_SHOULDER])
        rs = np.array(landmarks[RIGHT_SHOULDER])


        #shoulder movement
        shoulder_width = np.linalg.norm(ls - rs)

        person["shoulder_width"].append(shoulder_width)

        if len(person["shoulder_width"]) > 10:
            person["shoulder_width"].pop(0)

        body_expension = np.mean(person["shoulder_width"])



        #calculating the hand movement
        hand_speed = 0

        person["left_wrist"].append(lw)
        person["right_wrist"].append(rw)

        if len(person["left_wrist"]) > 10:
            person["left_wrist"].pop(0)

        if len(person["right_wrist"]) > 10:
            person["right_wrist"].pop(0)

        if len(person["left_wrist"]) >=2:

            lw_prev = person["left_wrist"][-2]
            lw_curr = person["left_wrist"][-1]

            rw_prev = person["right_wrist"][-2]
            rw_curr = person["right_wrist"][-1]

            lw_speed = np.linalg.norm(lw_curr - lw_prev)
            rw_speed = np.linalg.norm(rw_curr - rw_prev)

            wrist_speed = (lw_speed + rw_speed) / 2

            return{
                "wrist_speed": float(wrist_speed),
                "body_expension": float(body_expension)
            }  



        



