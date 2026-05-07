import mediapipe as mp 

class PoseEstimator:
    def __init__(self):
        self.mp_pose = mp.solution.pose

        self.pose = self.mp_pose.Pose(
            static_image_mode = False,
            min_detection_confidence = 0.5,
            min_tracking_confidence = 0.5
        )

    def estimate(self, person_crop):
        rgb = person_crop[:, :, ::-1]

        results = self.pose.process(rgb)

        if not results.pose_landmarks:
            return None
        
        landmarks = []

        for lm in results.pose_landmarks.landmark:
            landmarks.append((lm.x, lm.y))

        return landmarks