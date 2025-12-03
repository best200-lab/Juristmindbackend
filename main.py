import asyncio
import json
import logging
import os
import re
import uuid
import time
from typing import Dict, Optional, List
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from fastapi.middleware.cors import CORSMiddleware
from aiohttp import ClientTimeout, TCPConnector

# Simple regex-based classification (no NLTK for speed)
def classify_query(query: str) -> Dict:
    """
    Fast regex/keyword-based classification without external libs.
    """
    q_lower = query.lower()
  
    # Keyword detection
    legal_keywords = ['case', 'law', 'statute', 'section', 'court', 'decision', 'ruling', 'legal', 'jurisdiction', 'precedent']
    simple_keywords = ['what is', 'define', 'explain']
    entity_keywords = ['who is', 'where is', 'what about']
    draft_keywords = ['draft', 'write', 'template']
    fact_keywords = ['facts', 'details', 'events', 'parties', 'outcome', 'chronology']
  
    is_legal = any(re.search(rf'\b{re.escape(kw)}\b', q_lower) for kw in legal_keywords)
    is_case_specific = any(re.search(rf'\b{re.escape(kw)}\b', q_lower) for kw in ['case', 'section'])
    is_fact_specific = is_case_specific and any(re.search(rf'\b{re.escape(kw)}\b', q_lower) for kw in fact_keywords)
    is_simple_non_law = not is_legal and any(re.search(rf'\b{re.escape(kw)}\b', q_lower) for kw in simple_keywords)
    is_entity = any(re.search(rf'\b{re.escape(kw)}\b', q_lower) for kw in entity_keywords)
    is_draft = any(re.search(rf'\b{re.escape(kw)}\b', q_lower) for kw in draft_keywords)
    requires_search = len(query.split()) > 5 or any(re.search(rf'\b{re.escape(kw)}\b', q_lower) for kw in ['current', 'latest', 'recent', 'nigerian'])
  
    classification = {
        'intent': 'simple_non_law' if is_simple_non_law else
                  'entity_query' if is_entity else
                  'legal_case' if is_case_specific else
                  'draft' if is_draft else
                  'general_legal' if is_legal else
                  'general_search' if requires_search else 'direct_response',
        'needs_search': requires_search or is_legal or is_entity,
        'use_sections_cases': is_case_specific,
        'is_draft': is_draft,
        'is_fact_specific': is_fact_specific
    }
  
    logger = logging.getLogger(__name__)
    logger.info(f"Query classification: {classification}")
  
    return classification

def build_reasoned_prompt(query: str, classification: Dict, chat_history: List[Dict] = None, is_case_with_sources: bool = False) -> str:
    """
    Build a structured prompt that enforces reasoning. For cases, include instructions for at least 2 legal sections and detailed facts.
    For drafts, emphasize detailed, professional drafting. Include chat history for context.
    """
    base_prompt = f"User query: {query}\n\nFirst, understand the user's intent: They want information on {classification['intent']} related to Nigerian law where applicable. Follow the chat trend.\n"
  
    base_prompt += "\nNow, reason step by step:\n1. Use your knowledge and any search results provided by the system to answer accurately.\n"
  
    if classification['intent'] == 'legal_case' and is_case_with_sources:
        base_prompt += "3. For this legal case, draw from 3 sources for accuracy: 2 web sources (e.g., legal databases), and latest posts from X (social media discussions, sorted by latest). Structure: 1) Detailed facts (parties involved, key events in chronological order with specifics); 2) Main legal issues; 3) Court decision (direct quote if possible, outcome); 4) Back up with at least 2 specific Nigerian sections/laws (quote them briefly and explain relevance); discuss how recent discussions impact interpretation. Do not cite or mention X sources in your response.\n"
    elif classification['use_sections_cases']:
        base_prompt += "3. Structure for legal cases: 1) Detailed facts (parties involved, key events in chronological order with specifics); 2) Main legal issues; 3) Court decision (direct quote if possible, outcome); 4) Back up with at least 2 specific Nigerian sections/laws (quote them briefly and explain relevance).\n"
    elif classification['is_draft']:
        base_prompt += "3. Take your time to craft a very detailed, professional, irresistible, and perfect draft tailored to Nigerian legal standards. Make it comprehensive, with precise language, all necessary clauses, and impeccable structure. At the end, reference and state the relevant sections of the law that underpin the draft.\n"
    else:
        base_prompt += "3. Provide a clear, factual, and helpful answer.\n"
    
    base_prompt += "4. End with a helpful suggestion for follow-up, phrased to assist the user (e.g., 'Would you like me to explain further?', 'Should I draft a related document?').\n\nRespond thoughtfully as JuristMind, specializing in Nigerian law. Keep it concise yet comprehensive where needed. If relevant sources are available, cite them inline using [1], [2], etc., at the exact point in the body, and provide footnotes or a references section at the end with full details including links."
    
    return base_prompt         

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
        (r'\bGrok-4\b', 'JuristMind'),
        (r'\bGrok 4\b', 'JuristMind'),
        (r'\bGrok 4 Fast\b', 'JuristMind'),
        (r'\bcreated by xAI\b', 'developed by Oluwaseun Ogun'),
        (r'\bX\.com\b', 'the platform'),
        (r'\bX platform\b', 'the platform')
    ]
  
    sanitized = response
    for pattern, replacement in patterns:
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
  
    return sanitized

def handle_general_query(question: str) -> Optional[str]:
    """
    Handle predefined general queries.
    """
    q = question.lower().strip()
    for key in GENERAL_ANSWERS:
        if key in q:
            return GENERAL_ANSWERS[key]
    return None

def load_chat_history(chat_id: str) -> Optional[Dict]:
    """
    Load chat history from file.
    """
    try:
        with open(f"public/chats/{chat_id}.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def save_chat_history(chat_id: str, history: List[Dict]):
    """
    Save chat history to file.
    """
    os.makedirs("public/chats", exist_ok=True)
    with open(f"public/chats/{chat_id}.json", "w") as f:
        json.dump({"id": chat_id, "history": history}, f, indent=2)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GROK_API_KEY = os.getenv("GROK_API_KEY")

# Base URL for chat storage
BASE_CHAT_URL = os.getenv("BASE_CHAT_URL", "https://juristmind.onrender.com")

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Query Grok API with retry (streaming generator) - Optimized for speed
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), retry=retry_if_exception_type(aiohttp.ClientError))
async def query_grok(messages: List[Dict], search_params: Optional[Dict] = None, model: str = "grok-3", classification: Optional[Dict] = None):
    if not GROK_API_KEY:
        logger.error("Grok API key not set")
        yield {"type": "error", "message": "No API key set."}
        return

    try:
        # Enhanced token limit based on intent
        base_tokens = 1500
        if classification:
            if classification.get('intent') == 'draft':
                base_tokens = 4000
            elif classification.get('intent') in ['legal_case', 'general_legal']:
                base_tokens = 3000

        async with aiohttp.ClientSession(connector=TCPConnector(limit=10, limit_per_host=5)) as client:
            headers = {
                "Authorization": f"Bearer {GROK_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "messages": messages,
                "model": model,
                "stream": True,
                "temperature": 0.7,
                "max_tokens": base_tokens,
                "return_citations": True,
            }
            if search_params:
                payload["search_parameters"] = search_params

            timeout_total = 120
            if classification and classification.get('intent') in ['legal_case', 'draft']:
                timeout_total = 180
            timeout = ClientTimeout(total=timeout_total)

            logger.info(f"Sending Grok API request with model {model} (search: {bool(search_params)}, max_tokens: {base_tokens})")
            async with client.post("https://api.x.ai/v1/chat/completions",
                                   headers=headers, json=payload,
                                   timeout=timeout) as response:
                if response.status != 200:
                    text_data = await response.text()
                    yield {"type": "error", "message": f"API error: {text_data}"}
                    return

                buffer = ""
                last_yield_time = time.time()
                async for chunk in response.content.iter_any():
                    buffer += chunk.decode("utf-8")
                    now = time.time()
                    if now - last_yield_time > 3:
                        yield {"type": "ping"}
                        last_yield_time = now

                    while "\n\n" in buffer:
                        event, buffer = buffer.split("\n\n", 1)
                        for line in event.split("\n"):
                            if not line.startswith("data: "):
                                continue
                            data = line[6:].strip()
                            if data == "[DONE]":
                                return
                            try:
                                json_data = json.loads(data)
                            except json.JSONDecodeError:
                                break

                            if "choices" in json_data:
                                delta = json_data["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                finish_reason = json_data["choices"][0].get("finish_reason")
                                if finish_reason == "length":
                                    yield {"type": "warning", "message": "Response may be incomplete; consider follow-up for more details."}
                                if content:
                                    yield {"type": "content", "delta": content}
                                    last_yield_time = time.time()
                            if "citations" in json_data:
                                yield {"type": "citations", "data": json_data["citations"]}

                if buffer.strip():
                    try:
                        json_data = json.loads(buffer.strip())
                        if "choices" in json_data:
                            delta = json_data["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield {"type": "content", "delta": content}
                    except json.JSONDecodeError:
                        pass

    except aiohttp.ClientError as e:
        logger.exception(f"Grok API request failed: {e}")
        yield {"type": "error", "message": str(e)}

# Load legal document template
def load_template(template_name: str):
    try:
        with open(f"templates/{template_name}.txt", "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(f"templates/{template_name}.txt", "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except FileNotFoundError:
        logger.warning(f"Template {template_name} not found, using default.")
        return f"Default template for {template_name}. Insert your content here: {{content}}"

@app.get("/")
async def root():
    return JSONResponse({"message": "Welcome to JuristMind, your AI legal assistant for Nigerian law!"})

@app.get("/favicon.ico")
async def favicon():
    return FileResponse("static/favicon.ico", media_type="image/x-icon")

@app.post("/ask")
async def ask_question(request: Request):
    form = await request.form()
    question = form.get("question", "").strip()
    chat_id = form.get("chat_id")
    user_id = form.get("user_id")

    if not question:
        return JSONResponse({"answer": "Question cannot be empty"})

    general_answer = handle_general_query(question)
    classification = classify_query(question)
    q_lower = question.lower()

    # Dynamic model selection
    model = "grok-3"
    if classification['intent'] == 'draft':
        model = "grok-4-fast"
    elif classification['intent'] in ['legal_case', 'general_legal']:
        model = "grok-4"

    if re.search(r'\b(think deep|think)\b', q_lower):
        model = "grok-4-fast"

    async def generate():
        nonlocal general_answer, chat_id, model
        full_text = ""
        citations = []
        has_error = False
        search_params = None
        messages = []
        history = []

        # Load or initialize chat history
        if chat_id:
            chat_data = load_chat_history(chat_id)
            if chat_data:
                history = chat_data.get("history", [])
                messages = history.copy()

        if not chat_id:
            chat_id = str(uuid.uuid4())

        # Add system prompt
        messages.insert(0, {"role": "system", "content": "You are JuristMind, a logical AI specializing in Nigerian law but capable of answering general queries. For law-related answers, quote specific statutes and cases."})

        # Add user message
        user_msg = {"role": "user", "content": question}
        messages.append(user_msg)
        history.append(user_msg)

        if general_answer:
            sanitized = sanitize_response(general_answer)
            assistant_msg = {"role": "assistant", "content": sanitized}
            messages.append(assistant_msg)
            history.append(assistant_msg)
            full_text = sanitized
            yield f"data: {json.dumps({'content': sanitized})}\n\n"

        else:
            # Handle drafts with template
            prefix = ""
            suffix = ""
            if classification['intent'] == 'draft':
                template = load_template("default_legal")
                if '{content}' in template:
                    parts = template.split('{content}', 1)
                    prefix = parts[0]
                    suffix = parts[1] if len(parts) > 1 else ""
                if prefix:
                    sanitized_prefix = sanitize_response(prefix)
                    yield f"data: {json.dumps({'content': sanitized_prefix})}\n\n"
                    full_text += sanitized_prefix

            # Search configuration
            if classification['intent'] == 'legal_case':
                search_params = {
                    "mode": "auto",
                    "return_citations": True,
                    "max_search_results": 3,
                    "sources": [
                        {"type": "web"},
                        {"type": "news"},
                        {"type": "x"}
                    ]
                }
                custom_prompt = build_reasoned_prompt(question, classification, history, is_case_with_sources=True)
            else:
                if classification['needs_search'] or classification['is_draft']:
                    search_params = {
                        "mode": "auto",
                        "return_citations": True,
                        "max_search_results": 2,
                        "sources": [
                            {"type": "web"},
                            {"type": "news"}
                        ]
                    }
                custom_prompt = build_reasoned_prompt(question, classification, history)

            messages[-1]["content"] = custom_prompt

            # Stream response
            async for item in query_grok(messages, search_params, model, classification):
                if item["type"] == "content":
                    delta = item["delta"]
                    sanitized_delta = sanitize_response(delta)
                    yield f"data: {json.dumps({'content': sanitized_delta})}\n\n"
                    full_text += sanitized_delta
                elif item["type"] == "citations":
                    citations = item["data"]
                elif item["type"] == "warning":
                    warning_msg = item["message"]
                    yield f"data: {json.dumps({'content': warning_msg})}\n\n"
                    full_text += warning_msg
                elif item["type"] == "error":
                    has_error = True
                    error_msg = "[Error] " + item["message"]
                    yield f"data: {json.dumps({'content': error_msg})}\n\n"
                    full_text += error_msg

            assistant_msg = {"role": "assistant", "content": full_text}
            history.append(assistant_msg)

            # Add template suffix for drafts
            if suffix and classification['intent'] == 'draft' and not has_error:
                sanitized_suffix = sanitize_response(suffix)
                yield f"data: {json.dumps({'content': sanitized_suffix})}\n\n"
                full_text += sanitized_suffix
                history[-1]["content"] += sanitized_suffix

        if not has_error:
            save_chat_history(chat_id, history)

        # Append sources (exclude X)
        sources = []
        if citations and search_params is not None:
            for c in citations:
                if isinstance(c, dict) and 'url' in c and 'x.com' not in c['url'].lower():
                    sources.append({
                        "title": c.get("title", c.get("url", "")),
                        "url": c.get("url", "")
                    })
            if sources and "Sources:" not in full_text:
                sources_text = "\nSources: " + ", ".join([s['url'] for s in sources])
                yield f"data: {json.dumps({'content': sources_text})}\n\n"
                full_text += sources_text
                history[-1]["content"] += sources_text

        chat_url = f"{BASE_CHAT_URL}/public/chats/{chat_id}.json"
        yield f"data: {json.dumps({'type': 'done', 'chat_id': chat_id, 'chat_url': chat_url, 'sources': sources})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/chat/{chat_id}")
async def get_chat(chat_id: str):
    chat_data = load_chat_history(chat_id)
    if not chat_data:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat_data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
