from src.memory.event_store import get_recent_events
from src.reports.incident_report import generate_report


def answer_question(question):
    events = get_recent_events()

    if not events:
        return "No suspicious activity found."

    q = question.lower()
    latest_event = events[-1]

    if "show last suspicious event" in q or "last suspicious event" in q or "incident report" in q:
        return generate_report(latest_event)

    if "current risk level" in q or ("risk" in q and "level" in q):
        score = float(latest_event.get("risk_score", 0))
        if score >= 0.8:
            level = "HIGH"
        elif score >= 0.6:
            level = "MEDIUM"
        else:
            level = "LOW"
        return (
            f"Current threat level is {level} ({round(score * 100, 1)}%). "
            f"Latest subject: Person {latest_event.get('person_id', 'N/A')} on {latest_event.get('camera', 'CAM')}."
        )

    if "latest alert" in q or "alert message" in q:
        return latest_event.get("alert", "No alert message available.")

    if "which camera" in q or "active camera" in q:
        return f"Active camera: {latest_event.get('camera', 'CAM-01')}"

    if "last 10 minutes" in q or "last 10 min" in q:
        summaries = []

        for e in events:
            summaries.append(
                f"{e['timestamp']} - {e['reason']}"
            )

        return "\n".join(summaries)

    return "No matching activity found."
