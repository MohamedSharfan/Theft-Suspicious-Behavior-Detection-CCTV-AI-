import os
import time
from collections import deque
from datetime import datetime
from threading import Lock, Thread

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from ultralytics import YOLO

from src.features.feature_extractor import FeatureExtractor, is_frame_blurry
from src.features.pose_estimator import PoseEstimator
from src.features.pose_features import PoseFeatureExtractor
from src.ml.predict import AnomalyDetector
from src.scene.appearance import detect_clothing_color

SCORE_THRESHOLD = -0.1
RISK_INC = 1.5
RISK_DEC = 0.2
RISK_MIN = 0.0
RISK_MAX = 10.0
RISK_LOW = 3.0
RISK_MID = 6.0
ALERT_RISK = 5.0
HAND_SPEED_THRESH = 0.05
HAND_DIST_THRESH = 0.12
GESTURE_BONUS = 1.0

MAX_EVENTS = int(os.getenv("CCTV_MAX_EVENTS", "40"))
EVENT_INTERVAL = float(os.getenv("CCTV_EVENT_INTERVAL", "3.0"))
CAMERA_NAME = os.getenv("CCTV_CAMERA", "CAM-01")
LOCATION_NAME = os.getenv("CCTV_LOCATION", "shop aisle")

IMPORTANT_OBJECTS = {
    24,
    26,
    28,
    39,
    41,
    43,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    62,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    70,
    71,
    72,
    73,
    74
}

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

_event_lock = Lock()
_events = deque(maxlen=MAX_EVENTS)
_latest_event = None
_last_emit = {}
_last_status = {}
_frame_lock = Lock()
_latest_frame = None
_detections_lock = Lock()
_latest_detections = []
_metrics_lock = Lock()
_latest_metrics = {
    "subjects": 0,
    "suspicious": 0,
    "fps": 0.0,
    "camera_count": 1
}


class ChatRequest(BaseModel):
    question: str


def _now_stamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _risk_label(score: float) -> str:
    if score >= 0.8:
        return "High"
    if score >= 0.6:
        return "Medium"
    return "Low"


def _build_reason(window_features: dict) -> str:
    reasons = []
    if window_features.get("time_mean", 0) > 15:
        reasons.append("loitering")
    if window_features.get("angle_mean", 0) > 0.4:
        reasons.append("frequent direction changes")
    if window_features.get("stop_mean", 0) > 10:
        reasons.append("frequent stopping")
    if window_features.get("crouch_ratio", 0) > 0.5:
        reasons.append("crouching")
    if window_features.get("hand_mean", 1.0) < 0.15:
        reasons.append("concealment gesture")

    if reasons:
        return " + ".join(reasons)
    return "no clear suspicious behaviors"


def _build_alert(risk_score: float) -> str:
    if risk_score >= 0.8:
        return "HIGH RISK BEHAVIOR DETECTED"
    if risk_score >= 0.6:
        return "MEDIUM RISK BEHAVIOR DETECTED"
    return "PATTERN STABLE"


def _score_to_risk(model_score: float | None, risk: float) -> float:
    if model_score is not None:
        scaled = 1 / (1 + np.exp(model_score))
        return float(min(max(scaled, 0.0), 1.0))
    return float(min(max(risk / (ALERT_RISK * 2.0), 0.0), 1.0))


def _safe_explain(event: dict) -> str:
    try:
        from src.genai.explainer import explain_event

        return explain_event(event)
    except Exception as exc:
        return f"GenAI unavailable: {exc}"


def _parse_source(value: str | None):
    if value is None:
        return 0
    if value.isdigit():
        return int(value)
    return value


def _push_event(event: dict) -> None:
    global _latest_event
    with _event_lock:
        _events.appendleft(event)
        _latest_event = event


def _get_latest_event() -> dict | None:
    with _event_lock:
        return _latest_event


def _get_events() -> list[dict]:
    with _event_lock:
        return list(_events)


def _set_frame(frame: np.ndarray) -> None:
    global _latest_frame
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        return
    with _frame_lock:
        _latest_frame = encoded.tobytes()


def _get_frame() -> bytes | None:
    with _frame_lock:
        return _latest_frame


def _set_detections(detections: list[dict]) -> None:
    with _detections_lock:
        _latest_detections.clear()
        _latest_detections.extend(detections)


def _get_detections() -> list[dict]:
    with _detections_lock:
        return list(_latest_detections)


def _set_metrics(metrics: dict) -> None:
    with _metrics_lock:
        _latest_metrics.update(metrics)


def _get_metrics() -> dict:
    with _metrics_lock:
        return dict(_latest_metrics)


def _should_emit(track_id: int, status: str) -> bool:
    now = time.time()
    last_time = _last_emit.get(track_id, 0)
    last_status = _last_status.get(track_id)
    if status != last_status:
        return True
    return now - last_time >= EVENT_INTERVAL


def _mark_emitted(track_id: int, status: str) -> None:
    _last_emit[track_id] = time.time()
    _last_status[track_id] = status


def _build_behavior_chain(window_features: dict) -> list[str]:
    chain = ["Entered aisle"]
    if window_features.get("stop_mean", 0) > 10:
        chain.append("Stopped repeatedly")
    if window_features.get("angle_mean", 0) > 0.4:
        chain.append("Looked around frequently")
    if window_features.get("hand_mean", 1.0) < 0.15:
        chain.append("Concealment gesture")
    if window_features.get("crouch_ratio", 0) > 0.5:
        chain.append("Crouched near shelf")
    if window_features.get("time_mean", 0) > 15:
        chain.append("Remained in zone")
    return chain


def _build_recommendation(risk_score: float) -> str:
    if risk_score >= 0.8:
        return "Dispatch security to aisle and monitor exit routes."
    if risk_score >= 0.6:
        return "Monitor subject continuously and review prior frames."
    return "Continue observation and log activity for trend analysis."


def _process_stream() -> None:
    model_path = os.getenv("CCTV_MODEL_PATH", "yolov8n.pt")
    source = _parse_source(os.getenv("CCTV_SOURCE"))

    model = YOLO(model_path)
    model.to("cpu")

    extractor = FeatureExtractor()
    detector = AnomalyDetector()
    pose_estimator = PoseEstimator()
    pose_extractor = PoseFeatureExtractor()
    tracker = DeepSort(max_age=60, n_init=5, nn_budget=200)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        _push_event(
            {
                "id": int(time.time() * 1000),
                "timestamp": _now_stamp(),
                "person_id": 0,
                "risk_score": 0.0,
                "status": "normal",
                "reason": "camera source unavailable",
                "alert": "FEED OFFLINE",
                "camera": CAMERA_NAME,
                "explanation": "Live video feed could not be opened. Verify CCTV source configuration."
            }
        )
        return

    suspicion_history: dict[int, float] = {}
    crowd_history: list[int] = []
    frame_count = 0
    last_frame_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame = cv2.resize(frame, (640, 360))
        display_frame = frame.copy()
        if frame_count % 5 != 0:
            continue

        if is_frame_blurry(frame):
            continue

        results = model(frame)
        detections = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                if conf < 0.5:
                    continue
                if class_id == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))
                elif class_id in IMPORTANT_OBJECTS:
                    continue

        tracks = tracker.update_tracks(detections, frame=frame)
        valid_tracks = [track for track in tracks if track.is_confirmed()]

        crowd_count = len(valid_tracks)
        if crowd_count > 8:
            crowd_density = "high"
        elif crowd_count > 4:
            crowd_density = "medium"
        else:
            crowd_density = "low"

        frame_area = frame.shape[0] * frame.shape[1]
        crowd_density_ratio = crowd_count / (frame_area + 1e-6)

        centers = []
        for track in valid_tracks:
            x1, y1, x2, y2 = track.to_ltrb()
            shirt_color = detect_clothing_color(
                frame,
                (x1, y1, x2, y2)
            )
            centers.append(((x1 + x2) / 2, (y1 + y2) / 2))

        if len(centers) > 1:
            dists = []
            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    dists.append(np.linalg.norm(np.array(centers[i]) - np.array(centers[j])))
            avg_person_distance = float(np.mean(dists))
        else:
            avg_person_distance = 999.0

        crowd_history.append(crowd_count)
        if len(crowd_history) > 30:
            crowd_history.pop(0)

        crowd_mean_30 = float(np.mean(crowd_history))
        crowd_std_30 = float(np.std(crowd_history))

        now = time.time()
        dt = max(now - last_frame_time, 1e-6)
        fps = 1.0 / dt
        last_frame_time = now

        detections_payload = []
        suspicious_count = 0

        for track in valid_tracks:
            track_id = int(track.track_id)
            x1, y1, x2, y2 = map(int, track.to_ltrb())

            padding = 20
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(frame.shape[1], x2 + padding)
            y2 = min(frame.shape[0], y2 + padding)

            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue

            crop_h, crop_w = person_crop.shape[:2]
            if crop_h < 50 or crop_w < 50:
                continue

            pose_landmarks = None
            try:
                pose_landmarks = pose_estimator.estimate(person_crop)
            except Exception:
                pose_landmarks = None

            pose_points = []
            if pose_landmarks:
                for lm in pose_landmarks:
                    if len(lm) < 2:
                        continue
                    px = int(x1 + float(lm[0]) * (x2 - x1))
                    py = int(y1 + float(lm[1]) * (y2 - y1))
                    if 0 <= px < frame.shape[1] and 0 <= py < frame.shape[0]:
                        pose_points.append({"x": px, "y": py})

            extractor.update(track_id, (x1, y1, x2, y2), frame.shape, pose_landmarks)
            window_features = extractor.get_window_features(track_id)
            if window_features is None:
                risk = suspicion_history.get(track_id, 0.0)
                if risk < RISK_LOW:
                    level = "LOW"
                elif risk < RISK_MID:
                    level = "MEDIUM"
                else:
                    level = "HIGH"
                risk_score = _score_to_risk(None, risk)
                status = "suspicious" if risk >= ALERT_RISK else "normal"
                if status == "suspicious":
                    suspicious_count += 1

                color = (0, 255, 0) if risk < RISK_LOW else (0, 255, 255) if risk < RISK_MID else (0, 0, 255)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    display_frame,
                    f"ID:{track_id} {level} ({risk:.1f})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )
                if status == "suspicious":
                    cv2.putText(
                        display_frame,
                        "Suspicious",
                        (x1, y2 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        color,
                        1
                    )
                for point in pose_points:
                    cv2.circle(display_frame, (point["x"], point["y"]), 3, (255, 0, 0), -1)

                detections_payload.append(
                    {
                        "id": track_id,
                        "x": x1,
                        "y": y1,
                        "w": max(0, x2 - x1),
                        "h": max(0, y2 - y1),
                        "risk": float(risk_score),
                        "risk_raw": float(risk),
                        "risk_level": level,
                        "status": status,
                        "pose": pose_points
                    }
                )
                continue

            pose_features = pose_extractor.extract(track_id, pose_landmarks)
            if pose_features:
                window_features.update(pose_features)

            window_features["crowd_count"] = crowd_count
            window_features["crowd_density_ratio"] = crowd_density_ratio
            window_features["avg_person_distance"] = avg_person_distance
            window_features["crowd_mean_30"] = crowd_mean_30
            window_features["crowd_std_30"] = crowd_std_30

            is_suspicious = None
            model_score = None
            try:
                X = np.array([[
                    window_features["speed_mean"],
                    window_features["speed_std"],
                    window_features["angle_mean"],
                    window_features["angle_std"],
                    window_features["acc_mean"],
                    window_features["acc_std"],
                    window_features["time_mean"],
                    window_features["hand_mean"],
                    window_features["hand_std"],
                    window_features["stop_mean"],
                    window_features["stop_std"],
                    window_features["crouch_ratio"],
                    window_features["hand_speed"],
                    window_features["body_expansion"],
                    window_features["crowd_count"],
                    window_features["crowd_density_ratio"],
                    window_features["avg_person_distance"],
                    window_features["crowd_mean_30"],
                    window_features["crowd_std_30"]
                ]])
                X_scaled = detector.scaler.transform(X)
                model_score = float(detector.model.decision_function(X_scaled)[0])
                is_suspicious = model_score < SCORE_THRESHOLD
            except Exception:
                is_suspicious = None

            if is_suspicious is None:
                is_suspicious = detector.predict(window_features) == 1

            risk = suspicion_history.get(track_id, 0.0)
            if is_suspicious:
                risk += RISK_INC
            else:
                risk -= RISK_DEC

            gesture_sus = (
                window_features.get("hand_speed", 0) > HAND_SPEED_THRESH
                and window_features.get("hand_mean", 0) > HAND_DIST_THRESH
            )
            if gesture_sus:
                risk += GESTURE_BONUS

            risk = max(RISK_MIN, min(risk, RISK_MAX))
            suspicion_history[track_id] = risk

            if risk < RISK_LOW:
                level = "LOW"
            elif risk < RISK_MID:
                level = "MEDIUM"
            else:
                level = "HIGH"

            status = "suspicious" if risk >= ALERT_RISK else "normal"
            if status == "suspicious":
                suspicious_count += 1

            color = (0, 255, 0) if risk < RISK_LOW else (0, 255, 255) if risk < RISK_MID else (0, 0, 255)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                display_frame,
                f"ID:{track_id} {level} ({risk:.1f})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
            if status == "suspicious":
                cv2.putText(
                    display_frame,
                    "Suspicious",
                    (x1, y2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1
                )
            for point in pose_points:
                cv2.circle(display_frame, (point["x"], point["y"]), 3, (255, 0, 0), -1)

            risk_score = _score_to_risk(model_score, risk)
            detections_payload.append(
                {
                    "id": track_id,
                    "x": x1,
                    "y": y1,
                    "w": max(0, x2 - x1),
                    "h": max(0, y2 - y1),
                    "risk": float(risk_score),
                    "risk_raw": float(risk),
                    "risk_level": level,
                    "status": status,
                    "pose": pose_points
                }
            )

            if status != "suspicious":
                continue
            if not _should_emit(track_id, status):
                continue

            reason = _build_reason(window_features)
            # alert = _build_alert(risk_score)
            alert = (
                    f"⚠️ Suspicious behavior detected: "
                    f"Person wearing {shirt_color} clothing "
                    f"showing unusual movement, repeated activity, "
                    f"or concealment-like behavior."
                )
            behavior_chain = _build_behavior_chain(window_features)
            recommendation = _build_recommendation(risk_score)

            explanation = _safe_explain(
                {
                    "person_id": track_id,
                    "loitering": window_features.get("time_mean", 0) > 15,
                    "direction_change": window_features.get("angle_mean", 0) > 0.4,
                    "frequent_stopping": window_features.get("stop_mean", 0) > 10,
                    "crouching": window_features.get("crouch_ratio", 0) > 0.5,
                    "hand_distance": window_features.get("hand_mean", 0),
                    "risk_score": round(risk, 2),
                    "crowd_density": crowd_density,
                    "location": LOCATION_NAME
                }
            )

            event = {
                "id": int(time.time() * 1000),
                "timestamp": _now_stamp(),
                "person_id": track_id,
                "risk_score": float(risk_score),
                "status": status,
                "reason": reason,
                "alert": alert,
                "camera": CAMERA_NAME,
                "explanation": explanation,
                "behavior_chain": behavior_chain,
                "recommendation": recommendation
            }

            _push_event(event)
            _mark_emitted(track_id, status)
            _set_frame(display_frame)

        cv2.putText(
            display_frame,
            "GREEN: LOW | YELLOW: MED | RED: HIGH",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        _set_detections(detections_payload)
        _set_metrics(
            {
                "subjects": len(valid_tracks),
                "suspicious": suspicious_count,
                "fps": round(fps, 1),
                "camera_count": 1
            }
        )

        _set_frame(display_frame)

    cap.release()


@app.on_event("startup")
def _startup() -> None:
    worker = Thread(target=_process_stream, daemon=True)
    worker.start()


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/events")
def get_events() -> dict:
    latest = _get_latest_event()
    events = _get_events()
    detections = _get_detections()
    metrics = _get_metrics()
    return {
        "latest_event": latest,
        "events": events,
        "detections": detections,
        "metrics": metrics
    }


@app.get("/api/stream")
def stream() -> StreamingResponse:
    def generate():
        while True:
            frame = _get_frame()
            if frame is None:
                time.sleep(0.03)
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
            time.sleep(0.03)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache",
            "Pragma": "no-cache"
        }
    )


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict:
    latest = _get_latest_event()
    if not latest:
        return {"answer": "No active events yet. The system is still warming up."}

    question = request.question.strip().lower()
    if not question:
        return {"answer": "Please ask a question about the current feed."}

    if "why" in question or "flagged" in question:
        answer = (
            f"Person {latest['person_id']} was flagged because the model detected {latest['reason']}. "
            f"Confidence is {round(latest['risk_score'] * 100)}%, with status: {latest['status'].upper()}."
        )
    elif "last" in question or "event" in question:
        answer = (
            f"Last event: {latest['alert']} at {latest['timestamp']}. "
            f"Explanation: {latest['explanation']}"
        )
    elif "risk" in question or "level" in question:
        answer = (
            f"Current threat level is {_risk_label(latest['risk_score']).upper()}. "
            f"Highest active subject is Person {latest['person_id']} with {round(latest['risk_score'] * 100)}% risk."
        )
    else:
        answer = (
            f"Latest feed intelligence: Person {latest['person_id']}, {round(latest['risk_score'] * 100)}% risk, "
            f"reason: {latest['reason']}."
        )
    threat_level = _risk_label(latest["risk_score"]).upper()
    recommendation = latest.get("recommendation") or _build_recommendation(latest["risk_score"])

    return {
        "answer": answer,
        "threat_level": threat_level,
        "recommendation": recommendation
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=False)
