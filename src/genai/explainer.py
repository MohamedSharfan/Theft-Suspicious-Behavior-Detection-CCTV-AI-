from openai import OpenAI
from src.genai.prompt_builder import build_prompt
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("OPEN_AI_API_KEY")

if not API_KEY:
    raise ValueError("OPENAI api is not found")

client = OpenAI(api_key = API_KEY)

def explain_event(event):
    prompt = build_prompt(event)



    #"You are an advanced CCTV forensic AI specialized in theft, violence, and suspicious behavior detection."    need ton ujpdate it later

    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages = [
            {"role": "system", "content": "You are a strict security analysis AI"},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content