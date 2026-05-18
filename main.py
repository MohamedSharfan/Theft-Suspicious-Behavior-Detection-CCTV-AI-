import cv2
from ultralytics import YOLO
from src.features.feature_extractor import FeatureExtractor
from src.ml.predict import AnomalyDetector


model = YOLO("yolov8n.pt")
extractor = FeatureExtractor()
detector = AnomalyDetector()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox = (x1, y1, x2-x1, y2-y1)

                features = extractor.update(0, bbox)
                anomaly = detector.predict(features)

                if anomaly:
                    print("🤔Suspicious Activity")
    
    cv2.imshow("Live", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()