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

# ==================== CONFIG & LOGGING ====================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
load_dotenv()
GROK_API_KEY = os.getenv("GROK_API_KEY")
BASE_CHAT_URL = os.getenv("BASE_CHAT_URL", "https://juristmind.onrender.com")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== CLASSIFICATION (UNCHANGED) ====================
def classify_query(query: str) -> Dict:
    q_lower = query.lower()
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
    logger.info(f"Query classification: {classification}")
    return classification

# ==================== PROMPT BUILDER (UNCHANGED) ====================
def build_reasoned_prompt(query: str, classification: Dict, chat_history: List[Dict] = None, is_case_with_sources: bool = False) -> str:
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

# ==================== SANITIZE & GENERAL ANSWERS ====================
def sanitize_response(response: str) -> str:
    patterns = [
        (r'\bGrok\b', 'JuristMind'), (r'\bxAI\b', 'the developers'), (r'\bX\b', 'the platform'),
        (r'\bGrok-?3\b', 'JuristMind'), (r'\bGrok-?4\b', 'JuristMind'), (r'\bGrok 4 Fast\b', 'JuristMind'),
        (r'\bcreated by xAI\b', 'developed by Oluwaseun Ogun'), (r'\bX\.com\b', 'the platform')
    ]
    sanitized = response
    for pattern, replacement in patterns:
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
    return sanitized

# Expanded instant responses (ZERO API cost)
INSTANT_RESPONSES = {
    "hello": "Hello! I'm JuristMind, your Nigerian law specialist. How can I assist you today?",
    "hi": "Hi there! Ready to help with any legal question.",
    "thank you": "You're very welcome! Happy to help anytime.",
    "thanks": "My pleasure! Anything else?",
    "good morning": "Good morning! How can I assist you today?",
    "good afternoon": "Good afternoon! Ready when you are.",
    "good evening": "Good evening! How may I help?",
    "ok": "Alright! Let me know if you need anything else.",
    "continue": "Yes, please go ahead — I'm listening.",
    "bye": "Goodbye! Feel free to return anytime.",
}

def handle_instant_query(question: str) -> Optional[str]:
    q = question.lower().strip()
    for trigger, response in INSTANT_RESPONSES.items():
        if trigger in q:
            return response
    # Simple definitions or explanations without deep context
    if re.match(r'^(what is|define|explain briefly)\s', q):
        if any(word in q for word in ['section', 'act', 'constitution', 'evidence act', 'criminal code', 'penal code']):
            return None  # Let model handle legal sections
        return None  # For now, let model handle others
    return None

# ==================== CHAT HISTORY IO ====================
def load_chat_history(chat_id: str) -> Optional[List[Dict]]:
    try:
        with open(f"public/chats/{chat_id}.json", "r") as f:
            data = json.load(f)
            return data.get("history", [])
    except FileNotFoundError:
        return None

def save_chat_history(chat_id: str, history: List[Dict]):
    os.makedirs("public/chats", exist_ok=True)
    with open(f"public/chats/{chat_id}.json", "w") as f:
        json.dump({"id": chat_id, "history": history}, f, indent=2)

# ==================== SUMMARIZATION WITH GROK-3 (CHEAP) ====================
async def summarize_history(history: List[Dict]) -> str:
    if len(history) < 6:
        return ""
    
    past_messages = history[:-4]  # Everything except last 4 messages
    text_to_summarize = "\n".join([
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:500]}"
        for m in past_messages
    ])

    summary_prompt = f"""Summarize the key points and ongoing context from this conversation in 2-3 concise paragraphs. 
Focus on unresolved questions, user's goals, legal topics discussed, and any facts mentioned.
Do not say "previous conversation" or meta comments. Write as if continuing naturally.

Conversation:
{text_to_summarize}

Summary:"""

    messages = [
        {"role": "system", "content": "You are a concise summarizer."},
        {"role": "user", "content": summary_prompt}
    ]

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.x.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"},
                json={"model": "grok-3", "messages": messages, "max_tokens": 600, "temperature": 0.3},
                timeout=ClientTimeout(total=60)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"].strip()
    except:
        pass
    return "Previous conversation context was summarized for efficiency."

# ==================== GROK QUERY (REUSABLE) ====================
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def query_grok_stream(messages: List[Dict], search_params: Optional[Dict], model: str, classification: Dict):
    if not GROK_API_KEY:
        yield {"type": "error", "message": "API key missing"}
        return

    base_tokens = {"grok-3": 4000, "grok-4-fast": 8000, "grok-4": 12000}.get(model, 6000)
    if classification.get('is_draft'):
        base_tokens = 12000 if 'fast' in model else 16000

    async with aiohttp.ClientSession(connector=TCPConnector(limit=10)) as client:
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

        timeout = ClientTimeout(total=180 if 'draft' in classification.get('intent', '') or classification.get('intent') == 'legal_case' else 120)

        async with client.post("https://api.x.ai/v1/chat/completions", headers={
            "Authorization": f"Bearer {GROK_API_KEY}",
            "Content-Type": "application/json"
        }, json=payload, timeout=timeout) as response:

            if response.status != 200:
                yield {"type": "error", "message": await response.text()}
                return

            buffer = ""
            last_yield = time.time()
            async for chunk in response.content.iter_any():
                buffer += chunk.decode("utf-8", errors="ignore")
                now = time.time()
                if now - last_yield > 3:
                    yield {"type": "ping"}
                    last_yield = now

                while "\n\n" in buffer:
                    event, buffer = buffer.split("\n\n", 1)
                    for line in event.split("\n"):
                        if not line.startswith("data: "):
                            continue
                        data = line[6:].strip()
                        if data == "[DONE]":
                            return
                        try:
                            js = json.loads(data)
                            if "choices" in js:
                                delta = js["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield {"type": "content", "delta": content}
                                if js["choices"][0].get("finish_reason") == "length":
                                    yield {"type": "warning", "message": "Response truncated due to length."}
                            if "citations" in js:
                                yield {"type": "citations", "data": js["citations"]}
                        except:
                            continue

# ==================== MAIN ENDPOINT ====================
@app.post("/ask")
async def ask_question(request: Request):
    form = await request.form()
    question = form.get("question", "").strip()
    chat_id = form.get("chat_id")
    if not question:
        return JSONResponse({"answer": "Question cannot be empty"})

    classification = classify_query(question)

    # === INSTANT RESPONSE (0 tokens) ===
    instant = handle_instant_query(question)
    if instant:
        if not chat_id:
            chat_id = str(uuid.uuid4())
        history = load_chat_history(chat_id) or []
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": instant})
        save_chat_history(chat_id, history)
        return StreamingResponse(
            (f"data: {json.dumps({'content': instant + '\n\n'})}\n\n" +
             f"data: {json.dumps({'type': 'done', 'chat_id': chat_id, 'sources': []})}\n\n").split("\n\n"),
            media_type="text/event-stream"
        )

    # === LOAD FULL HISTORY ===
    history = load_chat_history(chat_id) if chat_id else []
    if not chat_id:
        chat_id = str(uuid.uuid4())

    # Save user message immediately
    user_msg = {"role": "user", "content": question}
    history.append(user_msg)

    async def generate():
        full_text = ""
        citations = []
        has_error = False
        sources = []

        # === INTENT-BASED CONFIG ===
        intent = classification['intent']
        config = {
            'simple_non_law':   {'recent_count': 2, 'model': 'grok-3',       'needs_summary': False},
            'entity_query':     {'recent_count': 3, 'model': 'grok-3',       'needs_summary': False},
            'general_legal':    {'recent_count': 3, 'model': 'grok-4-fast',  'needs_summary': True},
            'general_search':   {'recent_count': 3, 'model': 'grok-4-fast',  'needs_summary': True},
            'legal_case':       {'recent_count': 4, 'model': 'grok-4',       'needs_summary': True},
            'draft':            {'recent_count': 3, 'model': 'grok-4-fast',  'needs_summary': True},
            'direct_response':  {'recent_count': 3, 'model': 'grok-3',       'needs_summary': False},
        }.get(intent, {'recent_count': 3, 'model': 'grok-4-fast', 'needs_summary': True})

        recent_count = config['recent_count']
        model = "grok-4-fast" if "think deep" in question.lower() or "think" in question.lower() else config['model']
        needs_summary = config['needs_summary'] and len(history) > recent_count + 3

        # === BUILD FINAL MESSAGES (TINY CONTEXT) ===
        messages = [{"role": "system", "content": "You are JuristMind, a logical AI specializing in Nigerian law but capable of answering general queries. For law-related answers, quote specific statutes and cases."}]

        # Add summary if needed
        if needs_summary:
            summary = await summarize_history(history)
            if summary:
                messages.append({"role": "user", "content": "Here is a summary of our past conversation:"})
                messages.append({"role": "assistant", "content": summary})

        # Add recent real messages
        recent_messages = history[-recent_count:]
        messages.extend([m for m in recent_messages if m["role"] != "assistant" or m is recent_messages[-1]])  # avoid duplicate assistant if last

        # Custom prompt
        custom_prompt = build_reasoned_prompt(
            question,
            classification,
            history,
            is_case_with_sources=(intent == 'legal_case')
        )
        messages.append({"role": "user", "content": custom_prompt})

        # Search params
        search_params = None
        if classification['needs_search'] or intent in ['legal_case', 'draft']:
            sources_list = [{"type": "web"}, {"type": "news"}]
            if intent == 'legal_case':
                sources_list.append({"type": "x"})
            search_params = {
                "mode": "auto",
                "return_citations": True,
                "max_search_results": 3 if intent == 'legal_case' else 2,
                "sources": sources_list
            }

        # === STREAM RESPONSE ===
        template_prefix = ""
        template_suffix = ""
        if intent == 'draft':
            template = load_template("default_legal")
            if '{content}' in template:
                template_prefix, template_suffix = template.split('{content}', 1)
                if template_prefix:
                    sanitized = sanitize_response(template_prefix)
                    yield f"data: {json.dumps({'content': sanitized})}\n\n"
                    full_text += sanitized

        async for item in query_grok_stream(messages, search_params, model, classification):
            if item["type"] == "content":
                delta = sanitize_response(item["delta"])
                yield f"data: {json.dumps({'content': delta})}\n\n"
                full_text += delta
            elif item["type"] == "citations":
                citations = item["data"]
            elif item["type"] == "error":
                has_error = True
                yield f"data: {json.dumps({'content': '[Error] ' + item['message']})}\n\n"

        # Save assistant response
        assistant_msg = {"role": "assistant", "content": full_text}
        history.append(assistant_msg)
        if template_suffix and intent == 'draft' and not has_error:
            sanitized_suffix = sanitize_response(template_suffix)
            yield f"data: {json.dumps({'content': sanitized_suffix})}\n\n"
            full_text += sanitized_suffix
            history[-1]["content"] += sanitized_suffix

        # Add sources
        if citations and search_params:
            for c in citations:
                if isinstance(c, dict) and 'url' in c and 'x.com' not in c['url'].lower():
                    sources.append({"title": c.get("title", c.get("url")), "url": c["url"]})
            if sources and "Sources:" not in full_text:
                sources_text = "\nSources: " + ", ".join([s['url'] for s in sources])
                yield f"data: {json.dumps({'content': sources_text})}\n\n"
                full_text += sources_text
                history[-1]["content"] += sources_text

        save_chat_history(chat_id, history)
        chat_url = f"{BASE_CHAT_URL}/public/chats/{chat_id}.json"
        yield f"data: {json.dumps({'type': 'done', 'chat_id': chat_id, 'chat_url': chat_url, 'sources': sources})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

# ==================== UTILS ====================
def load_template(template_name: str) -> str:
    try:
        with open(f"templates/{template_name}.txt", "r", encoding="utf-8") as f:
            return f.read()
    except:
        return "{content}"

@app.get("/")
async def root():
    return {"message": "JuristMind API – Optimized & Cost-Effective"}

@app.get("/favicon.ico")
async def favicon():
    return FileResponse("static/favicon.ico", media_type="image/x-icon")

@app.get("/chat/{chat_id}")
async def get_chat(chat_id: str):
    history = load_chat_history(chat_id)
    if not history:
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"id": chat_id, "history": history}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
