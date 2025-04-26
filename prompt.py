import os
import requests
from dotenv import load_dotenv

def get_analysis_prompt(user_input):
    load_dotenv()
    prompt_url = os.getenv('SYSTEM_PROMPT')
    if not prompt_url:
        raise ValueError("SYSTEM_PROMPT environment variable not found")
    
    try:
        response = requests.get(prompt_url)
        response.raise_for_status()
        system_prompt = response.text
    except requests.RequestException as e:
        raise ValueError(f"Failed to fetch system prompt from URL: {e}")
    
    return f"{system_prompt}\n\n---\nUSER PROMPT: {user_input}"