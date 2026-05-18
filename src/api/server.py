import os
import time
from collections import deque
from datetime import datetime
from threading import Lock, Thread
from contextlib import asynccontextmanager

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
from ultralytics import YOLO

from src.features.feature_extractor import FeatureExtractor, is_frame_blurry
from src.features.pose_estimator import PoseEstimator
from src.features.pose_features import PoseFeatureExtractor
from src.ml.predict import AnomalyDetector
from src.scene.appearance import detect_clothing_color
from src.alerts.telegram_alert import send_telegram_alert

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
FRAME_SKIP = max(1, int(os.getenv("CCTV_FRAME_SKIP", "2")))
FRAME_API_IMGSZ = int(os.getenv("CCTV_FRAME_API_IMGSZ", "416"))
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("CLOUD", "").lower() != "true":
        worker = Thread(target=_process_stream, daemon=True)
        worker.start()
    yield


app = FastAPI(lifespan=lifespan)
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
_report_lock = Lock()
_latest_report = None
_detections_lock = Lock()
_latest_detections = []
_metrics_lock = Lock()
_latest_metrics = {
    "subjects": 0,
    "suspicious": 0,
    "fps": 0.0,
    "camera_count": 1
}
_frame_model = None
_frame_model_lock = Lock()
_frame_pipeline = None
_frame_pipeline_lock = Lock()


class ChatRequest(BaseModel):
    question: str


class _FrameTrack:
    def __init__(self, track_id: int, bbox: list[int], confidence: float):
        x, y, w, h = bbox
        self.track_id = track_id
        self.det_conf = confidence
        self.time_since_update = 0
        self._ltrb = (x, y, x + w, y + h)

    def is_confirmed(self) -> bool:
        return True

    def to_ltrb(self):
        return self._ltrb


def _now_stamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _get_frame_model():
    global _frame_model
    with _frame_model_lock:
        if _frame_model is None:
            model_path = os.getenv("CCTV_MODEL_PATH", "yolov8n.pt")
            _frame_model = YOLO(model_path)
            _frame_model.to("cpu")
        return _frame_model


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


def _risk_level_from_raw(risk: float) -> str:
    if risk < RISK_LOW:
        return "LOW"
    if risk < RISK_MID:
        return "MEDIUM"
    return "HIGH"


def _get_frame_pipeline() -> dict:
    global _frame_pipeline
    with _frame_pipeline_lock:
        if _frame_pipeline is None:
            try:
                detector = AnomalyDetector()
            except Exception as exc:
                print(f"[WARN] Anomaly model unavailable for frame API: {exc}")
                detector = None

            try:
                pose_estimator = PoseEstimator()
            except Exception as exc:
                print(f"[WARN] Pose estimator unavailable for frame API: {exc}")
                pose_estimator = None

            try:
                tracker = DeepSort(max_age=45, n_init=2, nn_budget=200)
            except Exception as exc:
                print(f"[WARN] DeepSORT unavailable for frame API: {exc}")
                tracker = None

            _frame_pipeline = {
                "tracker": tracker,
                "extractor": FeatureExtractor(),
                "detector": detector,
                "pose_estimator": pose_estimator,
                "pose_extractor": PoseFeatureExtractor(),
                "suspicion_history": {},
                "crowd_history": [],
                "last_frame_time": time.time()
            }
        return _frame_pipeline


def _heuristic_suspicion(window_features: dict) -> bool:
    signals = 0
    if window_features.get("time_mean", 0) > 15:
        signals += 1
    if window_features.get("angle_mean", 0) > 0.4:
        signals += 1
    if window_features.get("stop_mean", 0) > 10:
        signals += 1
    if window_features.get("crouch_ratio", 0) > 0.5:
        signals += 1
    if window_features.get("hand_speed", 0) > HAND_SPEED_THRESH:
        signals += 1
    return signals >= 2


def _safe_explain(event: dict) -> dict:
    try:
        from src.genai.explainer import explain_event

        result = explain_event(event)
        if isinstance(result, dict):
            return result
        return {
            "suspicious": False,
            "threat_level": "LOW",
            "explanation": "AI response was not JSON",
            "alert_message": "No actionable threat detected"
        }
    except Exception as exc:
        return {
            "suspicious": False,
            "threat_level": "LOW",
            "explanation": f"GenAI unavailable: {exc}",
            "alert_message": "No actionable threat detected"
        }


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


def _set_report(report: str) -> None:
    global _latest_report
    with _report_lock:
        _latest_report = report


def _get_report() -> str | None:
    with _report_lock:
        return _latest_report


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
    chain = ["Entered"]
    if window_features.get("stop_mean", 0) > 10:
        chain.append("Stopped repeatedly")
    if window_features.get("angle_mean", 0) > 0.4:
        chain.append("Looked around frequently")
    if window_features.get("hand_mean", 1.0) < 0.15:
        chain.append("Concealment gesture")
    if window_features.get("crouch_ratio", 0) > 0.5:
        chain.append("Crouched")
    if window_features.get("time_mean", 0) > 15:
        chain.append("Remained in zone")
    return chain


def _build_recommendation(risk_score: float) -> str:
    if risk_score >= 0.8:
        return "Dispatch security to aisle and monitor exit routes."
    if risk_score >= 0.6:
        return "Monitor subject continuously and review prior frames."
    return "Continue observation and log activity for trend analysis."


def _needs_suspicious_override(text: str | None) -> bool:
    if not text:
        return True
    lowered = text.lower()
    red_flags = [
        "no suspicious",
        "normal behavior",
        "no actionable threat",
        "low risk",
        "no threat"
    ]
    return any(flag in lowered for flag in red_flags)


def _process_stream() -> None:
    if os.getenv("CLOUD", "").lower() == "true":
        return

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
    last_overlays: list[dict] = []
    frame_count = 0
    last_frame_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame = cv2.resize(frame, (640, 360))
        display_frame = frame.copy()

        if last_overlays:
            for overlay in last_overlays:
                color = overlay["color"]
                x1, y1, x2, y2 = overlay["bbox"]
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    display_frame,
                    overlay["label"],
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )
                if overlay.get("suspicious"):
                    cv2.putText(
                        display_frame,
                        "Suspicious",
                        (x1, y2 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        color,
                        1
                    )
                for point in overlay.get("pose", []):
                    cv2.circle(display_frame, (point["x"], point["y"]), 3, (255, 0, 0), -1)

        cv2.putText(
            display_frame,
            "GREEN: LOW | YELLOW: MED | RED: HIGH",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        _set_frame(display_frame)
        if frame_count % FRAME_SKIP != 0:
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
        current_overlays: list[dict] = []

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

            extractor.update(track_id, (x1, y1, x2 - x1, y2 - y1), frame.shape, pose_landmarks)
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
                current_overlays.append(
                    {
                        "bbox": (x1, y1, x2, y2),
                        "color": color,
                        "label": f"ID:{track_id} {level} ({risk:.1f})",
                        "suspicious": status == "suspicious",
                        "pose": pose_points
                    }
                )

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

            is_suspicious = risk >= ALERT_RISK
            status = "suspicious" if is_suspicious else "normal"
            print(f"[DEBUG] Track:{track_id} Risk:{risk:.2f} Status:{status}")
            if status == "suspicious":
                suspicious_count += 1

            color = (0, 255, 0) if risk < RISK_LOW else (0, 255, 255) if risk < RISK_MID else (0, 0, 255)
            current_overlays.append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "color": color,
                    "label": f"ID:{track_id} {level} ({risk:.1f})",
                    "suspicious": status == "suspicious",
                    "pose": pose_points
                }
            )

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

            if not is_suspicious:
                continue

            reason = _build_reason(window_features)
            if reason == "no clear suspicious behaviors":
                reason = "Multiple suspicious activity patterns detected"
            alert = _build_alert(risk_score)
            behavior_chain = _build_behavior_chain(window_features)
            recommendation = _build_recommendation(risk_score)

            ai_result = _safe_explain(
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
            explanation = ai_result.get("explanation", "")
            if _needs_suspicious_override(explanation):
                explanation = (
                    f"Behavioral anomalies observed: {reason}. "
                    f"Crowd density {crowd_density}. "
                    f"Risk score {round(risk_score * 100, 1)}%."
                )

            alert_message = ai_result.get("alert_message", "")
            if _needs_suspicious_override(alert_message):
                alert_message = (
                    f"Suspicious activity detected. Verify subject {track_id} on {CAMERA_NAME}."
                )
            alert = alert_message

            if not _should_emit(track_id, status):
                continue

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

            from src.clips.clip_saver import save_clip
            from src.reports.incident_report import generate_report

            clip_path = save_clip(display_frame, track_id)
            event["clip"] = clip_path

            report = generate_report(event)
            print(report)
            _set_report(report)

            _push_event(event)
            from src.memory.event_store import store_event

            store_event(event)
            send_telegram_alert(alert)
            _mark_emitted(track_id, status)
            _set_frame(display_frame)

        _set_detections(detections_payload)
        _set_metrics(
            {
                "subjects": len(valid_tracks),
                "suspicious": suspicious_count,
                "fps": round(fps, 1),
                "camera_count": 1
            }
        )
        last_overlays = current_overlays

    cap.release()


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.post("/api/frame")
async def process_frame(file: UploadFile = File(...)) -> dict:
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty frame upload.")

    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image frame.")

    frame = cv2.resize(frame, (640, 360))
    model = _get_frame_model()
    pipeline = _get_frame_pipeline()
    tracker = pipeline["tracker"]
    extractor = pipeline["extractor"]
    detector = pipeline["detector"]
    pose_estimator = pipeline["pose_estimator"]
    pose_extractor = pipeline["pose_extractor"]
    suspicion_history = pipeline["suspicion_history"]
    crowd_history = pipeline["crowd_history"]

    results = model.predict(frame, imgsz=FRAME_API_IMGSZ, conf=0.4, verbose=False)

    raw_detections = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            if class_id != 0 or confidence < 0.4:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            raw_detections.append(([x1, y1, max(0, x2 - x1), max(0, y2 - y1)], confidence, "person"))

    if tracker is not None:
        tracks = tracker.update_tracks(raw_detections, frame=frame)
    else:
        tracks = [
            _FrameTrack(index + 1, bbox, confidence)
            for index, (bbox, confidence, _) in enumerate(raw_detections)
        ]
    valid_tracks = [
        track
        for track in tracks
        if track.time_since_update <= 1 and (track.is_confirmed() or raw_detections)
    ]

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

    crowd_mean_30 = float(np.mean(crowd_history)) if crowd_history else 0.0
    crowd_std_30 = float(np.std(crowd_history)) if crowd_history else 0.0

    now = time.time()
    dt = max(now - pipeline["last_frame_time"], 1e-6)
    fps = 1.0 / dt
    pipeline["last_frame_time"] = now

    detections = []
    suspicious_count = 0

    for track in valid_tracks:
        track_id = int(track.track_id)
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            continue

        person_crop = frame[y1:y2, x1:x2]
        pose_landmarks = None
        if person_crop.size and pose_estimator is not None:
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

        extractor.update(track_id, (x1, y1, x2 - x1, y2 - y1), frame.shape, pose_landmarks)
        window_features = extractor.get_window_features(track_id)
        risk = suspicion_history.get(track_id, 0.0)
        model_score = None

        if window_features is not None:
            pose_features = pose_extractor.extract(track_id, pose_landmarks)
            if pose_features:
                window_features.update(pose_features)

            window_features["crowd_count"] = crowd_count
            window_features["crowd_density_ratio"] = crowd_density_ratio
            window_features["avg_person_distance"] = avg_person_distance
            window_features["crowd_mean_30"] = crowd_mean_30
            window_features["crowd_std_30"] = crowd_std_30

            is_model_suspicious = None
            if detector is not None:
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
                    is_model_suspicious = model_score < SCORE_THRESHOLD
                except Exception:
                    try:
                        is_model_suspicious = detector.predict(window_features) == 1
                    except Exception:
                        is_model_suspicious = _heuristic_suspicion(window_features)
            else:
                is_model_suspicious = _heuristic_suspicion(window_features)

            if is_model_suspicious:
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

        level = _risk_level_from_raw(risk)
        risk_score = _score_to_risk(model_score, risk)
        status = "suspicious" if risk >= ALERT_RISK else "normal"
        if status == "suspicious":
            suspicious_count += 1

        track_confidence = getattr(track, "det_conf", 1.0)
        if callable(track_confidence):
            try:
                track_confidence = track_confidence()
            except Exception:
                track_confidence = 1.0
        if track_confidence is None:
            track_confidence = 1.0

        detections.append(
            {
                "id": track_id,
                "x": x1,
                "y": y1,
                "w": max(0, x2 - x1),
                "h": max(0, y2 - y1),
                "confidence": float(track_confidence),
                "risk": float(risk_score),
                "risk_raw": float(risk),
                "risk_level": level,
                "status": status,
                "pose": pose_points
            }
        )

        if status != "suspicious" or window_features is None or not _should_emit(track_id, status):
            continue

        reason = _build_reason(window_features)
        if reason == "no clear suspicious behaviors":
            reason = "Multiple suspicious activity patterns detected"
        behavior_chain = _build_behavior_chain(window_features)
        recommendation = _build_recommendation(risk_score)
        ai_result = _safe_explain(
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
        explanation = ai_result.get("explanation", "")
        if _needs_suspicious_override(explanation):
            explanation = (
                f"Behavioral anomalies observed: {reason}. "
                f"Crowd density {crowd_density}. "
                f"Risk score {round(risk_score * 100, 1)}%."
            )

        alert_message = ai_result.get("alert_message", "")
        if _needs_suspicious_override(alert_message):
            alert_message = f"Suspicious activity detected. Verify subject {track_id} on {CAMERA_NAME}."

        event = {
            "id": int(time.time() * 1000),
            "timestamp": _now_stamp(),
            "person_id": track_id,
            "risk_score": float(risk_score),
            "status": status,
            "reason": reason,
            "alert": alert_message,
            "camera": CAMERA_NAME,
            "explanation": explanation,
            "behavior_chain": behavior_chain,
            "recommendation": recommendation
        }

        from src.reports.incident_report import generate_report
        from src.memory.event_store import store_event

        _set_report(generate_report(event))
        _push_event(event)
        store_event(event)
        send_telegram_alert(alert_message)
        _mark_emitted(track_id, status)

    _set_metrics(
        {
            "subjects": len(detections),
            "suspicious": suspicious_count,
            "fps": round(fps, 1),
            "camera_count": 1
        }
    )
    _set_detections(detections)

    return {"detections": detections}


@app.get("/api/events")
def get_events() -> dict:
    latest = _get_latest_event()
    if latest and "explanation" not in latest:
        latest["explanation"] = "Analysis pending..."
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


@app.get("/api/report/latest")
def latest_report() -> Response:
    report = _get_report()
    if not report:
        return Response(status_code=404, content="No report available.")

    from io import BytesIO
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 72
    for line in report.splitlines():
        pdf.drawString(72, y, line)
        y -= 14
        if y < 72:
            pdf.showPage()
            y = height - 72
    pdf.save()

    buffer.seek(0)
    return Response(
        content=buffer.read(),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=incident_report.pdf"}
    )


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict:
    from src.genai.chat_agent import answer_question

    answer = answer_question(request.question)

    latest = _get_latest_event()
    if latest and "explanation" not in latest:
        latest["explanation"] = "Analysis pending..."

    return {
        "answer": answer
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=False)
