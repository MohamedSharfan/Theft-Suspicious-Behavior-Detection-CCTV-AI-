from src.genai.explainer import explain_event

event = {
    "person_id": 1,
    "speed": 0.12,
    "acceleration": -0.02,
    "direction_change": True,
    "loitering": True,
    "location": "shop aisle"
}

response = explain_event(event)

print(response)