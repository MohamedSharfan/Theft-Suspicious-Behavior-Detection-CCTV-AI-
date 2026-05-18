def build_prompt(event):

    behaviors = []

    if event.get("loitering"):
        behaviors.append("extended loitering")
    
    if event.get("direction_change"):
        behaviors.append("frequent directional changes")

    if event.get("frequent_stopping"):
        behaviors.append("repeated stopping behavior")

    if event.get("crouching"):
        behaviors.append("the person is crouching")
    
    if event.get("hand_distance", 1.0) < 0.15:
        behaviors.append("possible concealment hand motion")

    if behaviors:
        behavior_text = "multiple suspicious movement patterns detected"
    else:
        behavior_text = "normal behavior observed"

    return f"""
    CCTV INCIDENT EVENT

    ENVIRONMENT:
    Location Type: general indoor area

    PERSON DETAILS:
    Person ID: {event.get("person_id", "unknown")}

    DETECTED BEHAVIORS:
    {behavior_text}

    RISK METRICS:
    Risk Score: {event.get("risk_score", "unknown")}
    Hand Distance: {event.get("hand_distance", "unknown")}

    Crowd Density: {event.get("crowd_density", "unknown")}

    CONTEXT:
    This event comes from an AI CCTV surveillance system monitoring suspicious retail behavior.

    TASK:
    Analyze whether this behavior appears suspicious within the given environment.

    IMPORTANT:
    - Do not exaggerate
    - Avoid false accusations
    - Explain reasoning carefully
    - Consider retail shoplifting patterns
    - Standing alone is NOT suspicious
    - Loitering alone is NOT enough for theft suspicion
    - Avoid specific place references (aisle, shelf, register)

    REQUIRED RESPONSE FORMAT (STRICT JSON):

    {{
        "suspicious": true,
        "threat_level": "HIGH",
        "explanation": "short behavioral explanation in 1-2 lines",
        "alert_message": "short operator alert (max 20 words)"
    }}

    IMPORTANT:
    - suspicious must be true or false only
    - Return ONLY JSON
    - No markdown
    - No extra text
    """