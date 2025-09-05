# backend/generate_response.py

import os
import json
import requests
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

# ==============================
# CONFIGURATION
# ==============================
GROK_API_KEY = os.getenv("GROK_API_KEY")  # Your Grok API key
GROK_API_URL = "https://api.x.ai/v1/chat/completions"  # Grok endpoint
DUCKDUCKGO_SEARCH_URL = "https://api.duckduckgo.com/"

# ==============================
# LAWYER STYLE PROMPT
# ==============================
LAWYER_SYSTEM_PROMPT = """
You are JuristMind, an AI legal assistant.
Your role is to explain legal matters, cases, and general knowledge
with the clarity, accuracy, and structure of a professional lawyer.

Guidelines:
- Be precise, objective, and formal
- Use plain English for easy understanding
- When provided with search results, evaluate their credibility and cite them
- Never make up citations; only use provided sources
- If asked personal questions (name, creator, etc.), answer politely and truthfully
"""

# ==============================
# CASUAL QUESTIONS
# ==============================
CASUAL_ANSWERS = {
    "what is your name": "I am JuristMind, your AI legal assistant.",
    "who created you": "I was developed by Oluwaseun Ogun to assist with legal research and explanations.",
    "how old are you": "I do not have an age, but I was launched recently to assist with legal and general inquiries."
}

def is_casual_query(user_input: str) -> str:
    """Check if question is casual and return predefined answer."""
    lower_input = user_input.lower().strip()
    for key, answer in CASUAL_ANSWERS.items():
        if key in lower_input:
            return answer
    return None

# ==============================
# WEB SEARCH FUNCTION
# ==============================
def search_web(query: str, max_results: int = 5) -> list:
    """
    Perform a live web search using DuckDuckGo API (free).
    Returns a list of dicts: [{'title': ..., 'url': ..., 'snippet': ...}, ...]
    """
    params = {
        "q": query,
        "format": "json",
        "no_html": 1,
        "skip_disambig": 1
    }

    try:
        r = requests.get(DUCKDUCKGO_SEARCH_URL, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()

        results = []
        if "RelatedTopics" in data:
            for item in data["RelatedTopics"]:
                if "Text" in item and "FirstURL" in item:
                    results.append({
                        "title": item["Text"],
                        "url": item["FirstURL"],
                        "snippet": item["Text"]
                    })
                    if len(results) >= max_results:
                        break
        return results

    except Exception as e:
        return [{"title": "Search Error", "url": "", "snippet": str(e)}]

# ==============================
# GROK API CALL
# ==============================
def call_grok(user_input: str) -> str:
    """Send a request to Grok API and return the response text."""
    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "grok-beta",  # Adjust if your Grok account uses a different model name
        "messages": [
            {"role": "system", "content": LAWYER_SYSTEM_PROMPT},
            {"role": "user", "content": user_input}
        ],
        "temperature": 0.5,
        "max_tokens": 800
    }

    try:
        response = requests.post(GROK_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        if "choices" in data and data["choices"]:
            return data["choices"][0]["message"]["content"]
        return "No response from Grok."
    except Exception as e:
        return f"Error calling Grok: {str(e)}"

# ==============================
# MAIN RESPONSE GENERATOR
# ==============================
def generate_response(user_input: str, use_web_search: bool = False) -> str:
    # Handle casual queries instantly
    casual_reply = is_casual_query(user_input)
    if casual_reply:
        return casual_reply

    # If using web search, fetch results and pass them into Grok
    if use_web_search:
        search_results = search_web(user_input)
        search_context = "\n\n".join(
            [f"{i+1}. {res['title']} - {res['url']}\n{res['snippet']}" for i, res in enumerate(search_results)]
        )
        grok_input = f"Here are search results for the query '{user_input}':\n\n{search_context}\n\nNow, explain the answer like a lawyer."
    else:
        grok_input = user_input

    return call_grok(grok_input)

# ==============================
# TEST MODE
# ==============================
if __name__ == "__main__":
    while True:
        query = input("\nAsk JuristMind: ")
        if query.lower() in ["exit", "quit"]:
            break
        use_search = any(word in query.lower() for word in ["latest", "current", "today", "search"])
        answer = generate_response(query, use_web_search=use_search)
        print("\nJuristMind:", answer)
