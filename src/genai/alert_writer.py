import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(
    api_key=os.getenv("AI_API_KEY")
)


def generate_human_alert(event):
    prompt = f"""
    You are an AI CCTV surveillance assistant.

    Write a realistic human-like CCTV operator alert.

    IMPORTANT:
    - Sound professional
    - Sound natural
    - Be concise
    - Avoid robotic wording
    - Avoid bullet points
    - Avoid JSON
    - Avoid saying 'Suspicious: YES'
    - Write like a real security monitoring system

    EVENT:
    Location: {event.get("location")}
    Risk Score: {event.get("risk_score")}
    Loitering: {event.get("loitering")}
    Frequent stopping: {event.get("frequent_stopping")}
    Direction changes: {event.get("direction_change")}
    Concealment motion: {event.get("hand_distance") < 0.15}
    Clothing: {event.get("clothing")}

    OUTPUT EXAMPLE:
    ⚠️ Possible shoplifting behavior detected near aisle 2. Individual repeatedly scanned surroundings and made concealment-like hand movements.

    ONLY RETURN THE ALERT MESSAGE.
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.7
    )

    return response.choices[0].message.content.strip()
