def build_prompt(event):
    return f"""
    You are a security AI assistant.

    Analyze this structured behavior event:

    Person ID: {event["person_id"]}
    Speed: {event["speed"]}
    Acceleration: {event["acceleration"]}
    Direction change: {event["direction_change"]}
    Loitering: {event["loitering"]}
    Location: {event["location"]}

    TASK:
    1. Is this suspicious? (YES/NO)
    2. Why?
    3. Give suspicion score (0-1)
    4. Write a short CCTV alert message (max 20 words)
    """