import asyncio
import base64
import json
import logging
import os
import re
import uuid
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import requests
import uuid
from bs4 import BeautifulSoup
import logging
import base64
import re
import aiohttp
import time
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from fastapi.middleware.cors import CORSMiddleware

GENERAL_ANSWERS = {
    "what is your name": "I am JuristMind, your AI legal assistant.",
    "who created you": "I was developed by Oluwaseun Ogun to assist with legal research and explanations.",
    "how old are you": "I do not have an age, but I was launched recently to assist with legal and general inquiries."
}

def handle_general_query(question: str) -> str:
    q = question.lower().strip()
    for key in GENERAL_ANSWERS:
        if key in q:
            return GENERAL_ANSWERS[key]
    return None
# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GROK_API_KEY = os.getenv("GROK_API_KEY")
NWLR_LOGIN_URL = "https://nwlronline.com/auth/login"
NWLR_DASHBOARD_URL = "https://nwlronline.com/dashboard"
NWLR_SEARCH_URL = "https://nwlronline.com/dashboard/legal-research"
NWLR_EMAIL = "David.ajayi@aluko-oyebode.com"
NWLR_PASSWORD = "Law@2020"

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for request body
class QuestionRequest(BaseModel):
    question: str

# Session with browser-like headers
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://nwlronline.com/",
})

# Login to NWLR with retry
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(requests.exceptions.RequestException))
def login_nwlr():
    try:
        logger.info("Fetching NWLR login page")
        login_page = session.get(NWLR_LOGIN_URL, timeout=10)
        login_page.raise_for_status()
        soup = BeautifulSoup(login_page.text, "html.parser")
        # Check for all possible form fields
        form_inputs = soup.select("form input")
        payload = {
            "email": NWLR_EMAIL,
            "password": NWLR_PASSWORD,
        }
        for input_tag in form_inputs:
            name = input_tag.get("name")
            value = input_tag.get("value")
            if name and name not in ["email", "password"] and value:
                payload[name] = value
                logger.info(f"Added form field: {name}={value}")
       
        logger.info(f"Attempting login with payload: {payload}")
        response = session.post(NWLR_LOGIN_URL, data=payload, allow_redirects=True, timeout=10)
        response.raise_for_status()
        if "dashboard" in response.url:
            logger.info("Successfully logged into NWLR")
            return True
        logger.error(f"NWLR login failed: Redirected to {response.url}, Status: {response.status_code}, Response: {response.text[:500]}")
        return False
    except requests.exceptions.HTTPError as e:
        logger.error(f"NWLR login HTTP error: {e}, Status: {e.response.status_code if e.response else 'No response'}, Response: {e.response.text[:500] if e.response else 'No response'}")
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"NWLR login error: {e}")
        raise

# Crawl NWLR for case law
def crawl_nwlr(query):
    try:
        if not login_nwlr():
            logger.error("Falling back to Grok API due to NWLR login failure")
            return query_grok(f"Search for case law on {query}")
    except Exception as e:
        logger.error(f"Login retries failed: {e}")
        return query_grok(f"Search for case law on {query}")
   
    try:
        logger.info(f"Searching NWLR for: {query}")
        response = session.get(NWLR_SEARCH_URL, params={"q": query}, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
       
        case_links = soup.select('a[href*="/dashboard/legal-research/"]')
        if not case_links:
            logger.warning(f"No case links found for query: {query}")
            return query_grok(f"Search for case law on {query}")

        results = []
        for link in case_links[:3]:
            case_url = link.get('href')
            if not case_url.startswith("http"):
                case_url = f"https://nwlronline.com{case_url}"
           
            logger.info(f"Fetching case: {case_url}")
            case_response = session.get(case_url, timeout=10)
            case_response.raise_for_status()
            case_soup = BeautifulSoup(case_response.text, "html.parser")
           
            case_content = case_soup.select_one(".case-content, .case-details, .content")
            case_text = case_content.get_text(strip=True) if case_content else "No details available."
           
            try:
                case_id = re.search(r'/legal-research/([A-Za-z0-9=]+)', case_url).group(1)
                decoded_id = base64.b64decode(case_id).decode('utf-8')
                case_title = f"Case {decoded_id}"
            except:
                case_title = link.get_text(strip=True) or "Unnamed Case"
           
            results.append(f"{case_title}: {case_text[:500]}...")
            time.sleep(1)  # Avoid rate limiting
       
        return "\n\n".join(results) or "No case details available."
    except requests.exceptions.RequestException as e:
        logger.error(f"NWLR crawl error: {e}")
        return query_grok(f"Search for case law on {query}")

# Query xAI Grok API with streaming
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(aiohttp.ClientError))
async def query_grok(question):
    if not GROK_API_KEY:
        logger.error("GROK_API_KEY not set")
        yield f"data: {json.dumps({'error': 'GROK_API_KEY not set'})}\n\n"
        return

    try:
        async with aiohttp.ClientSession() as client:
            headers = {
                "Authorization": f"Bearer {GROK_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "messages": [
                    {"role": "system", "content": "You are a legal assistant specialized in Nigerian law."},
                    {"role": "user", "content": question}
                ],
                "model": "grok-beta",  # Updated to a valid model; change if needed
                "stream": True
            }
            logger.info(f"Sending Grok API request: {question}")
            async with client.post("https://api.x.ai/v1/chat/completions",
                                   headers=headers, json=payload, timeout=60) as response:
                if response.status != 200:
                    error_text = await response.text()
                    yield f"data: {json.dumps({'error': f'API error: {response.status} - {error_text}'})}\n\n"
                    return

                buffer = b""
                async for chunk in response.content:
                    buffer += chunk
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        line = line.decode("utf-8").strip()
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                yield "data: [DONE]\n\n"
                                return
                            try:
                                data = json.loads(data_str)
                                delta = data.get("choices", [{}])[0].get("delta", {}).get("content")
                                if delta:
                                    yield f"data: {json.dumps({'content': delta})}\n\n"
                            except json.JSONDecodeError:
                                logger.error(f"Failed to parse chunk: {data_str}")

    except aiohttp.ClientError as e:
        logger.exception(f"Grok API request failed: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


# Load legal document template
def load_template(template_name):
    try:
        with open(f"templates/{template_name}.txt", "r") as f:
            return f.read()
    except FileNotFoundError:
        logger.warning(f"Template {template_name} not found, searching online...")
        return search_online_template(template_name)

# Search online for templates (placeholder)
def search_online_template(template_name):
    return "Default template content for " + template_name

# Store chat session
def store_chat(question, response):
    chat_id = str(uuid.uuid4())
    chat_data = {"id": chat_id, "question": question, "response": response}
    os.makedirs("public/chats", exist_ok=True)
    with open(f"public/chats/{chat_id}.json", "w") as f:
        import json
        json.dump(chat_data, f)
    return f"https://juristmind.com/chat/{chat_id}"

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    question = request.question.strip()
    if not question:
        return JSONResponse({"answer": "Question cannot be empty"})

    general_answer = handle_general_query(question)
    if general_answer:
        async def stream_general():
            yield f"data: {json.dumps({'content': general_answer})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(stream_general(), media_type="text/event-stream")

    # Determine if question involves case law or drafting
    # Note: I moved this logic up, assuming the previous return was a paste error or leftover.
    # If not, you can adjust accordingly. For streaming, we handle each case.
    try:
        if "case" in question.lower() or "legal precedent" in question.lower():
            response = crawl_nwlr(question)
            async def stream_crawl():
                yield f"data: {json.dumps({'content': response})}\n\n"
                yield "data: [DONE]\n\n"
            # Store full response for chat (since crawl is sync/full)
            chat_url = store_chat(question, response)
            return StreamingResponse(stream_crawl(), media_type="text/event-stream", headers={"X-Chat-URL": chat_url})

        elif "draft" in question.lower() or "write" in question.lower():
            template_name = "default_legal"
            template = load_template(template_name)
            # For draft, collect full Grok response first, as template requires it
            full_grok = ""
            async for chunk in query_grok(question):
                # Since query_grok yields SSE lines, parse them here for collection
                if chunk.startswith("data: "):
                    data_str = chunk[6:].strip()
                    if data_str == "[DONE]":
                        break
                    data = json.loads(data_str)
                    delta = data.get("content", "")
                    full_grok += delta
            formatted_response = template.format(content=full_grok)
            async def stream_draft():
                yield f"data: {json.dumps({'content': formatted_response})}\n\n"
                yield "data: [DONE]\n\n"
            chat_url = store_chat(question, formatted_response)
            return StreamingResponse(stream_draft(), media_type="text/event-stream", headers={"X-Chat-URL": chat_url})

        else:
            # Stream directly from Grok
            stream = query_grok(question)
            # Collect full response in parallel for storing chat
            full_response = ""
            async def wrapped_stream():
                nonlocal full_response
                async for chunk in stream:
                    if chunk.startswith("data: "):
                        data_str = chunk[6:].strip()
                        if data_str != "[DONE]":
                            data = json.loads(data_str)
                            delta = data.get("content", "")
                            full_response += delta
                    yield chunk
                # After stream ends, store chat
                chat_url = store_chat(question, full_response)
                # You can send the chat_url in a final event if needed, e.g., yield f"data: {json.dumps({'chat_url': chat_url})}\n\n"
            return StreamingResponse(wrapped_stream(), media_type="text/event-stream")

    except Exception as e:
        logger.error(f"Error processing question: {e}")
        async def stream_error():
            error_msg = f"Error processing '{question}'. Please try again or contact support."
            yield f"data: {json.dumps({'error': error_msg})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(stream_error(), media_type="text/event-stream")

@app.get("/chat/{chat_id}")
async def get_chat(chat_id: str):
    try:
        with open(f"public/chats/{chat_id}.json", "r") as f:
            import json
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Chat not found")

@app.get("/test-nwlr-login")
async def test_nwlr_login():
    try:
        if login_nwlr():
            return {"status": "success", "message": "Successfully logged into NWLR"}
        return {"status": "failure", "message": "Failed to log into NWLR. Check logs for details."}
    except Exception as e:
        return {"status": "failure", "message": f"Login error: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)