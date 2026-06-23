import requests
import os
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

def send_message(text: str):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": text,
        "parse_mode": "Markdown"
    }
    response = requests.post(url, json=payload)
    return response.json()

def format_job(job: dict) -> str:
    return f"""*🎯 {job['title']}*

*Score:* {job['score']}/100
*Rationale:* {job['rationale']}

[View Job]({job['url']})"""

if __name__ == "__main__":
    jobs = [
        {"title": "AI Engineer Intern – Mistral AI", "score": 88, "rationale": "Strong RAG and LangGraph match.", "url": "https://example.com/job1"},
        {"title": "LLM Engineer Intern – Hugging Face", "score": 76, "rationale": "Good NLP background, missing fine-tuning experience.", "url": "https://example.com/job2"},
        {"title": "NLP Research Intern – Inria", "score": 65, "rationale": "Academic profile matches but missing French B2.", "url": "https://example.com/job3"},
    ]

    digest = "🗞 *JobRadar Daily Digest*\n\n"
    digest += "\n\n──────────────────\n\n".join(format_job(job) for job in jobs)

    send_message(digest)