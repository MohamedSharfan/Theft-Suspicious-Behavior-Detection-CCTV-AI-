from groq import Groq
from src.genai.prompt_builder import build_prompt
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("AI_API_KEY")

if not API_KEY:
    raise ValueError("AI api is not found")

client = Groq(api_key = API_KEY)

def explain_event(event):
    
    try:
        prompt = build_prompt(event)



        #"You are an advanced CCTV forensic AI specialized in theft, violence, and suspicious behavior detection."    need ton ujpdate it later

        response = client.chat.completions.create(
            model = "llama-3.1-8b-instant",
            messages=[
                {"role": "system",
                "content": """
                            You are an advanced CCTV forensic AI specialized in:
                            - retail theft analysis
                            - suspicious behavior detection
                            - violence detection
                            - surveillance reasoning

                            Your job is to analyze structured CCTV behavior events.

                            Rules:
                            - Avoid false accusations
                            - Not all loitering is suspicious
                            - Standing still alone is normal in many environments
                            - Use environmental context carefully
                            - Be professional and realistic
                            - Focus on behavioral patterns over time

                            Your responses must sound like a real security monitoring system.
                            """},
                {"role": "user", "content": prompt}
            ],
            temperature = 0.3
        )

        return response.choices[0].message.content
    except Exception as e:
        return f"Groq Error: {e}"