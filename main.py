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
from tenacity import retry, stop_after_attempt, wait_fixed, wait_exponential, retry_if_exception_type
from fastapi.middleware.cors import CORSMiddleware
from aiohttp import ClientTimeout, TCPConnector
import io
from fastapi import UploadFile
import PyPDF2
import docx
import base64

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
    project_keywords = ['law project', 'thesis', 'dissertation', 'final year project', 'research paper', 'write project']
    fact_keywords = ['facts', 'details', 'events', 'parties', 'outcome', 'chronology']
   
    is_legal = any(re.search(rf'\b{re.escape(kw)}\b', q_lower) for kw in legal_keywords)
    is_case_specific = any(re.search(rf'\b{re.escape(kw)}\b', q_lower) for kw in ['case', 'section'])
    is_fact_specific = is_case_specific and any(re.search(rf'\b{re.escape(kw)}\b', q_lower) for kw in fact_keywords)
    is_simple_non_law = not is_legal and any(re.search(rf'\b{re.escape(kw)}\b', q_lower) for kw in simple_keywords)
    is_entity = any(re.search(rf'\b{re.escape(kw)}\b', q_lower) for kw in entity_keywords)
    is_draft = any(re.search(rf'\b{re.escape(kw)}\b', q_lower) for kw in draft_keywords)
    is_project = any(re.search(rf'\b{re.escape(kw)}\b', q_lower) for kw in project_keywords)
    requires_search = len(query.split()) > 5 or any(re.search(rf'\b{re.escape(kw)}\b', q_lower) for kw in ['current', 'latest', 'recent', 'nigerian'])
   
    classification = {
        'intent': 'law_project' if is_project else
                  'simple_non_law' if is_simple_non_law else
                  'entity_query' if is_entity else
                  'legal_case' if is_case_specific else
                  'draft' if is_draft else
                  'general_legal' if is_legal else
                  'general_search' if requires_search else 'direct_response',
        'needs_search': requires_search or is_legal or is_entity or is_project,
        'use_sections_cases': is_case_specific or is_project,
        'is_draft': is_draft,
        'is_project': is_project,
        'is_fact_specific': is_fact_specific
    }
   
    # Internal logging
    logger = logging.getLogger(__name__)
    logger.info(f"Query classification: {classification}")
   
    return classification

def build_reasoned_prompt(query: str, classification: Dict, chat_history: List[Dict] = None, is_case_with_sources: bool = False, document_texts: List[str] = None, has_images: bool = False) -> str:
    """
    Build a structured prompt that enforces reasoning. For cases, include instructions for at least 2 legal sections and detailed facts.
    For drafts, emphasize detailed, professional drafting. Include chat history for context.
    If documents are provided, include instructions to analyze them.
    If images are provided, include instructions to recognize text and analyze them.
    """
    # No explicit history_context added; rely on messages list for implicit context
   
    base_prompt = f"User query: {query}\n\nFirst, understand the user's intent: They want information on {classification['intent']} related to Nigerian law where applicable. Follow the chat trend.\n"
   
    base_prompt += "\nNow, reason step by step:\n1. Use your knowledge and any search results provided by the system to answer accurately.\n"
   
    if document_texts or has_images:
        base_prompt += "\nThe user has attached documents and/or images; always analyze them to answer the query, even if not explicitly mentioned in the query.\n"
   
    if document_texts:
        base_prompt += "\nAnalyze and incorporate the following attached documents in your response:\n" + "\n\n".join(document_texts) + "\n"
   
    if has_images:
        base_prompt += "\nAnalyze the attached images, recognize any text within them using OCR-like capabilities, describe the visual content, and incorporate that into your response to answer the query accordingly.\n"
   
    if classification['intent'] == 'law_project':
        base_prompt += "2. For this law project, generate section by section in detail. If the title is missing from context, first ask for the title of the project. Then, ask for other details like number of chapters (default 5), key objectives, research questions if needed. Start with Title, Abstract, Table of Contents as the first section if not already generated. For subsequent sections, generate one at a time based on the user's request or the next logical chapter. Use markdown for formatting: # for main titles, ## for subsections, bullet points for Table of Contents. Use OSCOLA referencing style for all citations: footnotes for in-text references (e.g., party v party [year] report, pinpoint), and a full bibliography at the end of the complete project. Follow OSCOLA rules for cases, statutes (title (date) series), books, journals, and websites. Use at most 3 relevant sources (Nigerian laws, cases, journals) per section, fewer if not needed. Structure chapters typically as: Introduction, Literature Review, Methodology, Main Analysis (multiple chapters), Findings/Discussion, Conclusion, Recommendations. Ensure original, academic tone tailored to Nigerian law.\n"
    elif classification['use_sections_cases'] and is_case_with_sources:
        if classification['is_fact_specific']:
            base_prompt += "2. For this fact-specific query, emphasize detailed facts: 1) Parties involved, key events in chronological order with specifics; 2) Main legal issues; 3) Court decision (direct quote if possible, outcome); 4) Back up with at least 2 specific Nigerian sections/laws (quote them briefly and explain relevance); draw from up to 3 sources for accuracy. Discuss how recent discussions impact interpretation. Do not cite or mention X sources in your response.\n"
        else:
            base_prompt += "2. For this general legal query, provide a clear explanation: 1) Overview; 2) Key legal issues; 3) Relevant decisions or outcomes; 4) Back up with at least 2 specific Nigerian sections/laws (quote them briefly and explain relevance); draw from up to 3 sources.\n"
    elif classification['use_sections_cases']:
        if classification['is_fact_specific']:
            base_prompt += "2. Structure for fact-specific: 1) Detailed facts (parties involved, key events in chronological order with specifics); 2) Main legal issues; 3) Court decision (direct quote if possible, outcome); 4) Back up with at least 2 specific Nigerian sections/laws (quote them briefly and explain relevance).\n"
        else:
            base_prompt += "2. Structure for general: 1) Overview; 2) Main legal issues; 3) Decision or outcome; 4) Back up with at least 2 specific Nigerian sections/laws (quote them briefly and explain relevance).\n"
    elif classification['is_draft']:
        base_prompt += "2. Take your time to craft a very detailed, professional, irresistible, and perfect draft tailored to Nigerian legal standards. Make it comprehensive, with precise language, all necessary clauses, and impeccable structure. At the end, reference and state the relevant sections of the law that underpin the draft.\n"
    else:
        base_prompt += "2. Provide a clear, factual, and helpful answer. If applicable to law, back it up with provisions from Nigerian statutes or cases.\n"
   
    base_prompt += "3. End with a helpful suggestion for follow-up, phrased to assist the user (e.g., 'Would you like me to explain further?', 'Should I draft a related document?', 'What is the next section for your project?').\n\nRespond thoughtfully as JuristMind, specializing in Nigerian law. Keep it concise yet comprehensive where needed. If relevant sources are available, mention and cite them at the end; otherwise, do not."
   
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

async def extract_document_text(file: UploadFile) -> str:
    """
    Extract text from uploaded document based on file type.
    """
    contents = await file.read()
    filename = file.filename.lower()
    try:
        if filename.endswith('.pdf'):
            reader = PyPDF2.PdfReader(io.BytesIO(contents))
            text = ''
            for page in reader.pages:
                text += page.extract_text() + '\n'
        elif filename.endswith('.docx') or filename.endswith('.doc'):
            doc = docx.Document(io.BytesIO(contents))
            text = ''
            for para in doc.paragraphs:
                text += para.text + '\n'
        elif filename.endswith('.txt'):
            text = contents.decode('utf-8')
        else:
            text = ''
        return f"Document '{file.filename}':\n{text}\n"
    except Exception as e:
        logger.error(f"Error extracting text from {file.filename}: {e}")
        return f"Document '{file.filename}': [Error extracting text]\n"

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
    allow_origins=["*"], # Specify your frontend URL in production
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
        base_tokens = 1500 # Bump default
        if classification and classification.get('intent') == 'law_project':
            base_tokens = 6000  # Higher for detailed project sections
        elif any('draft' in msg.get('content', '').lower() for msg in messages if msg.get('role') == 'user'):
            base_tokens = 4000
        elif classification and classification.get('intent') in ['legal_case', 'general_legal']:
            base_tokens = 3000
        # For other intents, keep at 1500
        async with aiohttp.ClientSession(connector=TCPConnector(limit=10, limit_per_host=5)) as client:
            headers = {
                "Authorization": f"Bearer {GROK_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "messages": messages,
                "model": model, # Dynamic model selection
                "stream": True,
                "temperature": 0.7, # Balanced for legal accuracy
                "max_tokens": base_tokens,
                "return_citations": True, # Ensure citations are enabled
            }
            if search_params:
                payload["search_parameters"] = search_params
            # Dynamic timeout
            timeout_total = 120
            if classification and classification.get('intent') in ['legal_case', 'draft', 'law_project']:
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
                    # Faster heartbeat (every 3s)
                    now = time.time()
                    if now - last_yield_time > 3:
                        yield {"type": "ping"}
                        last_yield_time = now
                    # Process SSE events efficiently
                    while "\n\n" in buffer:
                        event, buffer = buffer.split("\n\n", 1)
                        for line in event.split("\n"):
                            if not line.startswith("data: "):
                                continue
                            data = line[6:].strip()
                            if data == "[DONE]":
                                buffer = ""
                                return
                            try:
                                json_data = json.loads(data)
                            except json.JSONDecodeError:
                                buffer = data + "\n\n" + buffer
                                break
                            if "choices" in json_data:
                                delta = json_data["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                finish_reason = json_data["choices"][0].get("finish_reason")
                                if finish_reason == "length":
                                    logger.warning("Response truncated due to max_tokens")
                                    yield {"type": "warning", "message": "Response may be incomplete; consider follow-up for more details."}
                                if content:
                                    yield {"type": "content", "delta": content}
                                    last_yield_time = time.time()
                            if "citations" in json_data:
                                yield {"type": "citations", "data": json_data["citations"]}
                                last_yield_time = time.time()
                # Drain buffer quickly
                if buffer.strip():
                    try:
                        json_data = json.loads(buffer.strip())
                        if "choices" in json_data:
                            delta = json_data["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            finish_reason = json_data["choices"][0].get("finish_reason")
                            if finish_reason == "length":
                                logger.warning("Response truncated due to max_tokens")
                                yield {"type": "warning", "message": "Response may be incomplete; consider follow-up for more details."}
                            if content:
                                yield {"type": "content", "delta": content}
                    except json.JSONDecodeError:
                        pass
                logger.info(f"Stream ended with buffer len: {len(buffer)}")
    except aiohttp.ClientError as e:
        logger.exception(f"Grok API request failed: {e}")
        yield {"type": "error", "message": str(e)}

# Load legal document template (no search, keep simple)
def load_template(template_name: str):
    try:
        # Try reading the file in UTF-8 first
        with open(f"templates/{template_name}.txt", "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        # Fallback: replace problematic characters instead of crashing
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

@app.get("/ask")
async def ask_question_get():
    return JSONResponse({"message": "Please use a POST request to /ask with a form body containing your question, optional chat_id, user_id, and files."})

@app.post("/ask")
async def ask_question(request: Request):
    form = await request.form()
    question = form.get("question", "").strip()
    chat_id = form.get("chat_id")
    user_id = form.get("user_id") # Not used currently, but captured
    files = form.getlist("files")
    if not question and not files:
        return JSONResponse({"answer": "Question or files cannot be empty"})
    # Extract document texts and image base64 if files are uploaded
    document_texts = []
    image_contents = []
    if files:
        for file in files:
            if isinstance(file, UploadFile):
                contents = await file.read()
                filename = file.filename.lower()
                if filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp')):
                    # Handle images for Grok vision
                    ext = filename.split('.')[-1]
                    if ext == 'jpg':
                        ext = 'jpeg'
                    base64_data = base64.b64encode(contents).decode('utf-8')
                    image_contents.append((ext, base64_data))
                else:
                    # Handle documents as before
                    doc_text = await extract_document_text(file)
                    document_texts.append(doc_text)
    has_images = len(image_contents) > 0
    general_answer = handle_general_query(question)
    classification = classify_query(question)
    q_lower = question.lower()
    # Dynamic model selection
    model = "grok-3" # Default
    if classification['intent'] in ['draft', 'law_project'] or document_texts or has_images: # Use grok-4-fast for drafts, projects, document analysis, or images
        model = "grok-4-fast"
    elif classification['intent'] in ['legal_case', 'general_legal']: # For cases and solving questions
        model = "grok-4"
    if re.search(r'\b(think deep|think)\b', q_lower):
        model = "grok-4-fast" # Fast reasoning override for deep thinks
    async def generate():
        nonlocal general_answer, chat_id, model
        full_text = ""
        citations = []
        has_error = False
        search_params = None # Default: No search for speed
        messages = []
        history = []
        # Load or initialize chat history
        if chat_id:
            chat_data = load_chat_history(chat_id)
            if chat_data:
                history = chat_data.get("history", [])
                messages = history.copy()
            else:
                # If invalid chat_id provided, reset history but keep same chat_id
                history = []
        else:
            # New conversation → generate a fresh chat_id
            chat_id = str(uuid.uuid4())
            history = []
        # Add system prompt
        messages.insert(0, {"role": "system", "content": "You are JuristMind, a logical AI specializing in Nigerian law but capable of answering general queries. For law-related answers, quote specific statutes and cases."})
        # Append user question (include file mentions if any)
        user_content = question
        if document_texts:
            user_content += "\n\nAttached documents for analysis."
        if has_images:
            user_content += "\n\nAttached images for analysis."
        # For vision, user content must be a list if images are present
        if has_images:
            user_content_list = [{"type": "text", "text": user_content}]
            for ext, base64_data in image_contents:
                user_content_list.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{ext};base64,{base64_data}"}
                })
            user_msg = {"role": "user", "content": user_content_list}
        else:
            user_msg = {"role": "user", "content": user_content}
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
            # Handle drafts or projects with template
            template_name = "default_legal"
            if classification['intent'] == 'law_project':
                template_name = "law_project"
            if classification['intent'] in ['draft', 'law_project']:
                template = load_template(template_name)
                if '{content}' in template:
                    parts = template.split('{content}', 1)
                    prefix = parts[0]
                    suffix = parts[1] if len(parts) > 1 else ""
                else:
                    prefix = ""
                    suffix = ""
                if prefix:
                    sanitized_prefix = sanitize_response(prefix)
                    yield f"data: {json.dumps({'content': sanitized_prefix})}\n\n"
                    full_text += sanitized_prefix
            # Special handling for legal cases or projects: Enable search
            if classification['intent'] in ['legal_case', 'law_project']:
                search_params = {
                    "mode": "auto",
                    "return_citations": True,
                    "max_search_results": 3, # Fixed at 3 for projects and cases
                    "sources": [
                        {"type": "web"}, # 1. General web for legal facts
                        {"type": "news"}, # 2. News websites for updates
                        {"type": "x"} # 3. X for discussions (not cited)
                    ]
                }
                custom_prompt = build_reasoned_prompt(question, classification, history, is_case_with_sources=True, document_texts=document_texts, has_images=has_images)
                if has_images:
                    messages[-1]["content"][0]["text"] = custom_prompt # Override the text part
                else:
                    messages[-1]["content"] = custom_prompt # Override user content with full prompt
            else:
                # Set search only if needed (including for drafts now; fixed at 3 results)
                if classification['needs_search'] or classification['is_draft']:
                    if classification['intent'] == 'legal_case':
                        # Already handled above
                        pass
                    else:
                        # Lighter search: web+news only, max=2
                        search_params = {
                            "mode": "auto",
                            "return_citations": True,
                            "max_search_results": 2,
                            "sources": [
                                {"type": "web"}, # Primary: General web
                                {"type": "news"} # Secondary: News websites
                            ]
                        }
                custom_prompt = build_reasoned_prompt(question, classification, history, document_texts=document_texts, has_images=has_images)
                if has_images:
                    messages[-1]["content"][0]["text"] = custom_prompt # Override the text part
                else:
                    messages[-1]["content"] = custom_prompt # Override user content with full prompt
            # Stream response
            async for item in query_grok(messages, search_params, model, classification):
                if item["type"] == "content":
                    delta = item["delta"]
                    sanitized_delta = sanitize_response(delta)
                    logger.info(f"Yielded {len(delta)} chars; total so far: {len(full_text)}")
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
            # Append assistant response to history
            assistant_msg = {"role": "assistant", "content": full_text}
            history.append(assistant_msg)
            # Add suffix for drafts or projects
            if 'suffix' in locals() and suffix and classification['intent'] in ['draft', 'law_project'] and not has_error:
                sanitized_suffix = sanitize_response(suffix)
                yield f"data: {json.dumps({'content': sanitized_suffix})}\n\n"
                full_text += sanitized_suffix
                history[-1]["content"] += sanitized_suffix
        if not has_error:
            logger.info("Processing complete, storing chat.")
            save_chat_history(chat_id, history)
        # Prepare sources from citations (filter out X) and append if not in content and search was performed
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
        logger.info(f"Chat stored at: {chat_url}")
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
