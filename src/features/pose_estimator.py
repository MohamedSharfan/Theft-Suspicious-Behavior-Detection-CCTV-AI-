import mediapipe as mp
import cv2

class PoseEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose

        self.pose = self.mp_pose.Pose(
            static_image_mode = False,
            min_detection_confidence = 0.5,
            min_tracking_confidence = 0.5
        )

    def estimate(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # rgb = person_crop[:, :, ::-1]

        results = self.pose.process(rgb)

        if not results.pose_landmarks:
            return None
        
        landmarks = []

        for lm in results.pose_landmarks.landmark:
            landmarks.append((lm.x, lm.y))

        return landmarks