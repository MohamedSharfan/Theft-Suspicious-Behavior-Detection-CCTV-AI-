def build_prompt(event):
    return f"""
    You are a security AI assistant.

    Analyze this structured behavior event:

    Person ID: {event["person_id"]}
    Direction change: {event["direction_change"]}
    Hand Distance: {event["hand_distance"]}
    Frequently stopping: {event["frequent_stopping"]}
    crouching: {event["crouching"]}
    Loitering: {event["loitering"]}
    Risk score: {event["risk_score"]}
    Location: {event["location"]}

    TASK:
    1. Is this suspicious? (YES/NO)
    2. Why?
    3. Give suspicion score (0-1)
    4. Write a short CCTV alert message (max 20 words)
    """