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
        behavior_text = ", ".join(behaviors)
    else:
        behavior_text = "no clear suspicious behaviors detected"

    return f"""
    CCTV INCIDENT EVENT

    ENVIRONMENT:
    Location Type: {event.get("location", "unknown")}

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
    - Add a few relevant stickers/emojis in the response to improve readability

    REQUIRED RESPONSE FORMAT:

    Suspicious: YES or NO

    Threat Level: LOW / MEDIUM / HIGH

    Explanation:
    Explain why the behavior may or may not be suspicious.

    Alert Message:
    Write a professional CCTV operator alert under 25 words.
    """