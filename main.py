import asyncio
import json
import logging
import os
import re
import uuid
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import aiohttp
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from fastapi.middleware.cors import CORSMiddleware

# General answers for common queries
GENERAL_ANSWERS = {
    "what is your name": "I am JuristMind, your AI legal assistant.",
    "who created you": "I was developed by Oluwaseun Ogun to assist with legal research and explanations.",
    "how old are you": "I do not have an age, but I was launched recently to assist with legal and general inquiries."
}

def sanitize_response(response: str) -> str:
    """
    Remove or replace references to Grok, xAI, or X in the response.
    """
    patterns = [
        (r'\bGrok\b', 'JuristMind'),
        (r'\bxAI\b', 'the developers'),
        (r'\bX\b', 'the platform'),
        (r'\bGrok-3\b', 'JuristMind'),
        (r'\bGrok 3\b', 'JuristMind'),
        (r'\bcreated by xAI\b', 'developed by Oluwaseun Ogun'),
        (r'\bX\.com\b', 'the platform'),
        (r'\bX platform\b', 'the platform')
    ]
    
    sanitized = response
    for pattern, replacement in patterns:
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
    
    return sanitized

def markdown_to_html(text: str) -> str:
    """
    Convert Markdown bold and headings to HTML for frontend rendering.
    Handles **bold**, ### Heading 3, ## Heading 2.
    """
    # Convert **text** to <strong>text</strong>
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    # Convert ### Heading to <h3>Heading</h3>
    text = re.sub(r'###\s+(.*?)\n', r'<h3>\1</h3>\n', text)
    # Convert ## Heading to <h2>Heading</h2>
    text = re.sub(r'##\s+(.*?)\n', r'<h2>\1</h2>\n', text)
    return text

def handle_general_query(question: str) -> str:
    """
    Handle predefined general queries.
    """
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
# Base URL for chat storage (configure as needed for your deployment)
BASE_CHAT_URL = os.getenv("BASE_CHAT_URL", "https://juristmind.onrender.com")  # Updated for Render

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Specify your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for request body
class QuestionRequest(BaseModel):
    question: str

# Query JuristMind API with retry
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2), retry=retry_if_exception_type(aiohttp.ClientError))
async def query_grok(question: str):
    if not GROK_API_KEY:
        logger.error("JuristMind API key not set")
        return sanitize_response(f"No case law found for '{question}'. Please verify your query or contact support.")
    try:
        async with aiohttp.ClientSession() as client:
            headers = {
                "Authorization": f"Bearer {GROK_API_KEY}",
                "Content-Type": "application/json"
            }
            # Configure Live Search parameters
            search_params = {
                "mode": "auto",  # Model decides when to search
                "return_citations": True,  # Include source URLs
                "max_search_results": 1,  # Limit to one source to minimize cost
                "sources": [
                    {"type": "web"},  # Allow web sources
                    {"type": "news"}  # Allow news sources
                    # Excluding "x" and "rss" to avoid social media and RSS feeds
                ]
            }
            payload = {
                "messages": [
                    {"role": "system", "content": "You are JuristMind, a legal assistant specialized in Nigerian law."},
                    {"role": "user", "content": question}
                ],
                "model": "grok-3",
                "stream": False,
                "search_parameters": search_params  # Enable Live Search
            }
            logger.info(f"Sending JuristMind API request: {question}")
            async with client.post("https://api.x.ai/v1/chat/completions",
                                   headers=headers, json=payload, timeout=60) as response:
                text_data = await response.text()
                logger.info(f"JuristMind raw response: {text_data}")
                try:
                    data = json.loads(text_data)
                except json.JSONDecodeError:
                    logger.error("Failed to parse JuristMind JSON response")
                    return sanitize_response(f"[Error] JuristMind returned invalid JSON: {text_data}")

                if "error" in data:
                    return sanitize_response(f"[JuristMind Error] {data['error'].get('message', 'Unknown error')}")

                try:
                    raw_response = data["choices"][0]["message"]["content"]
                    # Log citations and sources used for debugging/cost tracking
                    citations = data.get("citations", [])
                    num_sources = data.get("usage", {}).get("num_sources_used", 0)
                    logger.info(f"Citations: {citations}, Sources used: {num_sources}")
                    response = raw_response
                    if citations:
                        response += f"\n\nSource: {citations[0]}"  # Append single citation
                    return sanitize_response(response)
                except (KeyError, IndexIndexError):
                    return sanitize_response("[Error] Unexpected JuristMind API response format.")

    except aiohttp.ClientError as e:
        logger.exception(f"JuristMind API request failed: {e}")
        return sanitize_response(f"[Error] Could not reach JuristMind API: {e}")

# Load legal document template
def load_template(template_name: str):
    try:
        with open(f"templates/{template_name}.txt", "r") as f:
            return f.read()
    except FileNotFoundError:
        logger.warning(f"Template {template_name} not found, searching online...")
        return search_online_template(template_name)

# Search online for templates (placeholder)
def search_online_template(template_name: str):
    return sanitize_response(f"Default template content for {template_name}")

# Store chat session
def store_chat(question: str, response: str):
    chat_id = str(uuid.uuid4())
    chat_data = {"id": chat_id, "question": question, "response": response}
    os.makedirs("public/chats", exist_ok=True)
    with open(f"public/chats/{chat_id}.json", "w") as f:
        json.dump(chat_data, f)
    # Return correct path to the chats folder
    return f"{BASE_CHAT_URL}/public/chats/{chat_id}.json"

@app.get("/")
async def root():
    return JSONResponse({"message": "Welcome to JuristMind, your AI legal assistant for Nigerian law!"})

@app.get("/favicon.ico")
async def favicon():
    return FileResponse("static/favicon.ico", media_type="image/x-icon")

@app.get("/ask")
async def ask_question_get():
    return JSONResponse({"message": "Please use a POST request to /ask with a JSON body containing your question."})

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    question = request.question.strip()
    if not question:
        return JSONResponse({"answer": "Question cannot be empty"})

    general_answer = handle_general_query(question)
    if general_answer:
        sanitized_answer = sanitize_response(general_answer)
        html_answer = markdown_to_html(sanitized_answer)  # Convert Markdown to HTML
        chat_url = store_chat(question, sanitized_answer)
        logger.info(f"Chat stored at: {chat_url}")
        return JSONResponse({"answer": html_answer, "chat_url": chat_url})

    try:
        if "case" in question.lower() or "legal precedent" in question.lower():
            # Improved prompt for detailed case law response, structured as requested: facts with events, then legal issue, then decision with judge name and quote, then follow-up questions
            response = await query_grok(
                f"Search the internet for Nigerian case law on {question}. Use one reliable source (e.g., official legal websites or reputable news). "
                f"Structure your response exactly as follows: "
                f"1) Detailed facts of the case, including parties involved and key events in chronological order; "
                f"2) The main legal issue(s) raised in the case; "
                f"3) The decision of the court, a direct quote from what they said regarding the decision, if available; also include the final outcome. "
                f"4) Suggested follow-up questions: list 1 relevant questions the user might ask next to encourage further interaction. "
                f"Cite the source and exclude any social media references."
            )
        elif "draft" in question.lower() or "write" in question.lower():
            template_name = "default_legal"
            template = load_template(template_name)
            grok_response = await query_grok(question + " At the end of your response, suggest 1 relevant follow-up questions to encourage further interaction.")
            response = template.format(content=grok_response)
        else:
            response = await query_grok(question + " At the end of your response, suggest 1 relevant follow-up questions to encourage further interaction but dont mention follow up questions, rather leave a space for  that and give the follow up.")
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        response = f"Error processing '{question}'. Please try again or contact support."
    
    # Sanitize and convert the final response to HTML
    response = sanitize_response(response)
    html_response = markdown_to_html(response)
    
    # Store chat and generate link
    chat_url = store_chat(question, response)
    logger.info(f"Chat stored at: {chat_url}")
    
    return JSONResponse({"answer": html_response, "chat_url": chat_url})

@app.get("/chat/{chat_id}")
async def get_chat(chat_id: str):
    try:
        with open(f"public/chats/{chat_id}.json", "r") as f:
            chat_data = json.load(f)
            # Convert stored response to HTML before returning
            chat_data["response"] = markdown_to_html(chat_data["response"])
            return chat_data
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Chat not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
