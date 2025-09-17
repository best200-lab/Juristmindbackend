import asyncio
import json
import logging
import os
import re
import uuid
import time
import sqlite3
from typing import Dict, Optional, List, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import aiohttp
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from fastapi.middleware.cors import CORSMiddleware
from aiohttp import ClientTimeout

# Session Management Class
class IntelligentChatSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.user_info: Dict[str, Any] = {}
        self.conversation_history: List[Dict[str, Any]] = []
        self.custom_facts: Dict[str, Any] = {}
        
        # Load existing data from database
        self.load_from_db()
    
    def load_from_db(self):
        """Load session data from database"""
        try:
            conn = sqlite3.connect('data/chat_sessions.db')
            cursor = conn.cursor()
            cursor.execute(
                'SELECT user_info, conversation_history, custom_facts FROM sessions WHERE session_id = ?', 
                (self.session_id,)
            )
            
            result = cursor.fetchone()
            if result:
                self.user_info = json.loads(result[0]) if result[0] else {}
                self.conversation_history = json.loads(result[1]) if result[1] else []
                self.custom_facts = json.loads(result[2]) if result[2] else {}
            
            conn.close()
        except sqlite3.Error as e:
            logging.error(f"Error loading session from database: {e}")
    
    def save_to_db(self):
        """Save session data to database"""
        try:
            conn = sqlite3.connect('data/chat_sessions.db')
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO sessions 
                (session_id, user_info, conversation_history, custom_facts, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (
                self.session_id,
                json.dumps(self.user_info),
                json.dumps(self.conversation_history),
                json.dumps(self.custom_facts)
            ))
            
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            logging.error(f"Error saving session to database: {e}")
    
    def extract_and_remember(self, message: str) -> bool:
        """
        Extract personal information from message and store it
        Returns True if information was found and stored
        """
        message_lower = message.lower()
        extracted = False
        
        # Pattern matching for different types of information
        patterns = {
            'name': [
                r"(my name is|i am|call me) ([A-Za-z\s]+)(?:\.|$)",
                r"(^|\.|\s)([A-Z][a-z]+ [A-Z][a-z]+) is my name"
            ],
            'age': [
                r"(i am|i'm) (\d+) years old",
                r"(my age is|i am) (\d+)(?:\.|$)"
            ],
            'location': [
                r"(i live in|i'm from) ([A-Za-z\s]+)(?:\.|$)",
                r"(my city is|my country is) ([A-Za-z\s]+)(?:\.|$)"
            ],
            'email': [
                r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})"
            ],
            'phone': [
                r"(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})"
            ]
        }
        
        # Try to extract information using patterns
        for info_type, regex_list in patterns.items():
            for pattern in regex_list:
                match = re.search(pattern, message_lower)
                if match:
                    # Extract the value (different patterns might have different group numbers)
                    value = match.group(2) if len(match.groups()) >= 2 else match.group(1)
                    if value:
                        self.user_info[info_type] = value.strip()
                        extracted = True
                        self.save_to_db()  # Save immediately when info is found
        
        # Handle custom facts (e.g., "I like pizza")
        fact_patterns = [
            r"(i like|i love|i enjoy) ([^\.]+)(?:\.|$)",
            r"(my favorite [^ ]+ is) ([^\.]+)(?:\.|$)",
            r"(i have|i own) ([^\.]+)(?:\.|$)",
            r"(i work as|i am a) ([^\.]+)(?:\.|$)"
        ]
        
        for pattern in fact_patterns:
            match = re.search(pattern, message_lower)
            if match:
                fact_key = match.group(1).strip()
                fact_value = match.group(2).strip()
                self.custom_facts[fact_key] = fact_value
                extracted = True
                self.save_to_db()  # Save immediately when info is found
        
        return extracted
    
    def remember_fact(self, key: str, value: Any):
        """Manually remember a fact"""
        self.custom_facts[key] = value
        self.save_to_db()
    
    def recall(self, key: str) -> Optional[Any]:
        """Recall information from memory"""
        if key in self.user_info:
            return self.user_info[key]
        elif key in self.custom_facts:
            return self.custom_facts[key]
        else:
            return None
    
    def get_conversation_context(self, max_messages: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation history for context"""
        return self.conversation_history[-max_messages:]
    
    def add_to_history(self, user_message: str, ai_response: str):
        """Add a message exchange to history"""
        self.conversation_history.append({
            'user': user_message,
            'ai': ai_response,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep history manageable
        if len(self.conversation_history) > 100:  # Keep last 100 exchanges
            self.conversation_history = self.conversation_history[-100:]
        
        # Save to database
        self.save_to_db()
    
    def handle_memory_queries(self, user_input: str) -> Optional[str]:
        """Handle queries about remembered information"""
        user_input_lower = user_input.lower()
        
        memory_queries = {
            'name': ['what is my name', 'do you know my name', 'what did i say my name was'],
            'age': ['how old am i', 'what is my age', 'did i tell you my age'],
            'location': ['where do i live', 'what is my location', 'where am i from'],
            'email': ['what is my email', 'do you know my email'],
            'phone': ['what is my phone number', 'do you know my phone number']
        }
        
        for info_type, queries in memory_queries.items():
            for query in queries:
                if query in user_input_lower:
                    value = self.recall(info_type)
                    if value:
                        if info_type == 'name':
                            return f"Your name is {value}!"
                        elif info_type == 'age':
                            return f"You are {value} years old!"
                        elif info_type == 'location':
                            return f"You live in {value}!"
                        elif info_type == 'email':
                            return f"Your email is {value}!"
                        elif info_type == 'phone':
                            return f"Your phone number is {value}!"
                    else:
                        return f"I don't know your {info_type} yet. Could you tell me?"
        
        # Handle custom fact queries
        for fact_key, fact_value in self.custom_facts.items():
            if fact_key in user_input_lower:
                return f"Yes, you told me that {fact_key} {fact_value}!"
        
        # Check for "what do you know about me" type questions
        if 'what do you know about me' in user_input_lower or 'what have i told you' in user_input_lower:
            if not self.user_info and not self.custom_facts:
                return "I don't know anything about you yet. Tell me something about yourself!"
            
            response = "Here's what I know about you:\n"
            if self.user_info:
                response += "\nPersonal information:\n"
                for key, value in self.user_info.items():
                    response += f"- Your {key}: {value}\n"
            
            if self.custom_facts:
                response += "\nOther facts:\n"
                for key, value in self.custom_facts.items():
                    response += f"- {key}: {value}\n"
            
            return response
        
        return None

# Global session storage
chat_sessions: Dict[str, IntelligentChatSession] = {}

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
    
    is_legal = any(re.search(rf'\b{re.escape(kw)}\b', q_lower) for kw in legal_keywords)
    is_case_specific = any(re.search(rf'\b{re.escape(kw)}\b', q_lower) for kw in ['case', 'section'])
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
        'is_draft': is_draft
    }
    
    # Internal logging
    logger = logging.getLogger(__name__)
    logger.info(f"Query classification: {classification}")
    
    return classification

def build_reasoned_prompt(query: str, classification: Dict, conversation_history: List[Dict] = None, is_case_with_sources: bool = False) -> str:
    """
    Build a structured prompt that enforces reasoning with proper conversation context.
    """
    # Build conversation context from history
    history_context = ""
    if conversation_history:
        # Format the conversation history for the prompt
        history_context = "\n\nPrevious conversation:\n"
        for msg in conversation_history[-6:]:  # Last 6 exchanges (3 back-and-forths)
            if 'user' in msg and 'ai' in msg:
                history_context += f"User: {msg['user']}\n"
                history_context += f"Assistant: {msg['ai']}\n\n"
            elif 'role' in msg and 'content' in msg:
                # Handle the old format too
                role = "User" if msg['role'] == 'user' else "Assistant"
                history_context += f"{role}: {msg['content']}\n\n"
    
    base_prompt = f"{history_context}Current query: {query}\n\n"
    
    base_prompt += "First, understand the user's intent based on the conversation history and current query.\n"
    
    base_prompt += "\nNow, reason step by step:\n1. Consider the full conversation context and what was previously discussed.\n2. Restate what the user is asking in your own words, considering the ongoing discussion.\n3. Use your knowledge and any search results provided by the system to answer accurately.\n"
    
    if classification['use_sections_cases'] and is_case_with_sources:
        base_prompt += "4. For this legal case, draw from 3 sources for accuracy: 1 web source (e.g., legal databases), 1 news source (recent developments), and latest posts from X (social media discussions, sorted by latest). Structure: 1) Detailed facts (parties involved, key events in chronological order with specifics); 2) Main legal issues; 3) Court decision (direct quote if possible, outcome); 4) Back up with at least 2 specific Nigerian sections/laws (quote them briefly and explain relevance); discuss how recent discussions impact interpretation.\n"
    elif classification['use_sections_cases']:
        base_prompt += "4. Structure for legal cases: 1) Detailed facts (parties involved, key events in chronological order with specifics); 2) Main legal issues; 3) Court decision (direct quote if possible, outcome); 4) Back up with at least 2 specific Nigerian sections/laws (quote them briefly and explain relevance).\n"
    elif classification['is_draft']:
        base_prompt += "4. Take your time to craft a very detailed, professional, irresistible, and perfect draft tailored to Nigerian legal standards. Make it comprehensive, with precise language, all necessary clauses, and impeccable structure. At the end, reference and state the relevant sections of the law that underpin the draft.\n"
    else:
        base_prompt += "4. Provide a clear, factual, and helpful answer that continues the conversation naturally.\n"
    
    base_prompt += "5. End with a helpful suggestion for follow-up, phrased to assist the user (e.g., 'Would you like me to explain further?', 'Should I draft a related document?').\n\nRespond thoughtfully as JuristMind, specializing in Nigerian law. Keep it concise yet comprehensive where needed. Do not mention sources, reasoning, or search processes in the response. Maintain conversation continuity."
    
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
    allow_origins=["*"],  # Specify your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for request body
class QuestionRequest(BaseModel):
    question: str
    chat_id: Optional[str] = None

# Query Grok API with retry (streaming generator) - Optimized for speed
@retry(stop=stop_after_attempt(3), wait=wait_fixed(1), retry=retry_if_exception_type(aiohttp.ClientError))
async def query_grok(messages: List[Dict], search_params: Optional[Dict] = None):
    if not GROK_API_KEY:
        logger.error("Grok API key not set")
        yield {"type": "error", "message": "No API key set."}
        return

    try:
        async with aiohttp.ClientSession() as client:
            headers = {
                "Authorization": f"Bearer {GROK_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "messages": messages,
                "model": "grok-3",  # Use grok-3 for speed; switch to grok-4 if needed for Premium
                "stream": True,
                "temperature": 0.7,  # Balanced for legal accuracy
                "max_tokens": 2000 if any(msg.get('role') == 'user' and 'draft' in msg['content'].lower() for msg in messages) else 1000,  # Higher for drafts
            }
            if search_params:
                payload["search_parameters"] = search_params

            logger.info(f"Sending Grok API request (search: {bool(search_params)})")
            async with client.post("https://api.x.ai/v1/chat/completions",
                                   headers=headers, json=payload,
                                   timeout=ClientTimeout(total=60)) as response:  # Shorter timeout for speed

                if response.status != 200:
                    text_data = await response.text()
                    yield {"type": "error", "message": f"API error: {text_data}"}
                    return

                buffer = ""
                last_yield_time = time.time()

                async for chunk in response.content.iter_any():
                    buffer += chunk.decode("utf-8")

                    # Faster heartbeat (every 5s instead of 10)
                    now = time.time()
                    if now - last_yield_time > 5:
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
                            if content:
                                yield {"type": "content", "delta": content}
                    except json.JSONDecodeError:
                        pass

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
    return JSONResponse({"message": "Please use a POST request to /ask with a JSON body containing your question and optional chat_id."})

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    question = request.question.strip()
    chat_id = request.chat_id
    if not question:
        return JSONResponse({"answer": "Question cannot be empty"})

    # Get or create session
    if chat_id and chat_id in chat_sessions:
        session = chat_sessions[chat_id]
    else:
        if not chat_id:
            chat_id = str(uuid.uuid4())
        session = IntelligentChatSession(chat_id)
        chat_sessions[chat_id] = session

    # Check for memory queries first
    memory_response = session.handle_memory_queries(question)
    if memory_response:
        session.add_to_history(question, memory_response)
        return JSONResponse({"answer": memory_response, "chat_id": chat_id})

    # Extract and remember information from the question
    session.extract_and_remember(question)

    general_answer = handle_general_query(question)
    classification = classify_query(question)

    async def generate():
        nonlocal general_answer, chat_id
        full_text = ""
        citations = []
        has_error = False
        search_params = None
        messages = []

        # Prepare conversation history for the prompt
        conversation_history = session.conversation_history
        
        # Add system prompt with memory context
        memory_context = ""
        if session.user_info or session.custom_facts:
            memory_context = "\nUser information to remember: "
            if session.user_info:
                memory_context += " ".join([f"{k}: {v}" for k, v in session.user_info.items()])
            if session.custom_facts:
                memory_context += " ".join([f"{k}: {v}" for k, v in session.custom_facts.items()])
            memory_context += "\n"
        
        system_prompt = f"You are JuristMind, a logical legal AI specializing in Nigerian law. Always reason before answering and quote both specific statutes and cases applicable in your responses.{memory_context}"
        messages.append({"role": "system", "content": system_prompt})

        if general_answer:
            # For general answers, just return them directly
            sanitized = sanitize_response(general_answer)
            session.add_to_history(question, sanitized)
            yield f"data: {json.dumps({'content': sanitized})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'chat_id': chat_id, 'chat_url': f'{BASE_CHAT_URL}/public/chats/{chat_id}.json'})}\n\n"
            return
        else:
            # Build the reasoned prompt with conversation history
            custom_prompt = build_reasoned_prompt(question, classification, conversation_history)
            
            # Special handling for legal cases: Enable search with 2 sources
            if classification['intent'] == 'legal_case':
                search_params = {
                    "mode": "auto",
                    "return_citations": True,
                    "max_search_results": 2,
                    "sources": [
                        {"type": "web"},
                        {"type": "news"}
                    ]
                }
                # Add the case-specific instructions to the prompt
                custom_prompt += "\n\nNote: This is a legal case query. Please provide detailed analysis with specific legal references."

            # Add the user message with the custom prompt
            messages.append({"role": "user", "content": custom_prompt})

            # Handle drafts with template (no search)
            if classification['is_draft']:
                template_name = "default_legal"
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

            # Stream response from Grok
            async for item in query_grok(messages, search_params):
                if item["type"] == "content":
                    delta = item["delta"]
                    sanitized_delta = sanitize_response(delta)
                    yield f"data: {json.dumps({'content': sanitized_delta})}\n\n"
                    full_text += sanitized_delta
                elif item["type"] == "citations":
                    citations = item["data"]
                elif item["type"] == "error":
                    has_error = True
                    error_msg = "[Error] " + item["message"]
                    yield f"data: {json.dumps({'content': error_msg})}\n\n"
                    full_text += error_msg

            # Add suffix for drafts
            if 'suffix' in locals() and suffix and classification['is_draft'] and not has_error:
                sanitized_suffix = sanitize_response(suffix)
                yield f"data: {json.dumps({'content': sanitized_suffix})}\n\n"
                full_text += sanitized_suffix

        if not has_error:
            # Add to conversation history
            session.add_to_history(question, full_text)
            
            # Also save to the JSON chat history for compatibility
            chat_data = load_chat_history(chat_id) or {"id": chat_id, "history": []}
            chat_data["history"].extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": full_text}
            ])
            save_chat_history(chat_id, chat_data["history"])
            
        chat_url = f"{BASE_CHAT_URL}/public/chats/{chat_id}.json"
        yield f"data: {json.dumps({'type': 'done', 'chat_id': chat_id, 'chat_url': chat_url})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/chat/{chat_id}")
async def get_chat(chat_id: str):
    chat_data = load_chat_history(chat_id)
    if not chat_data:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat_data

@app.get("/session/{chat_id}/memory")
async def get_session_memory(chat_id: str):
    """Get the memory information for a session"""
    if chat_id in chat_sessions:
        session = chat_sessions[chat_id]
        return {
            "user_info": session.user_info,
            "custom_facts": session.custom_facts,
            "conversation_count": len(session.conversation_history)
        }
    else:
        # Try to load from database
        session = IntelligentChatSession(chat_id)
        if session.user_info or session.custom_facts:
            chat_sessions[chat_id] = session
            return {
                "user_info": session.user_info,
                "custom_facts": session.custom_facts,
                "conversation_count": len(session.conversation_history)
            }
        else:
            raise HTTPException(status_code=404, detail="Session not found")

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("public/chats", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    
    # Initialize database
    try:
        conn = sqlite3.connect('data/chat_sessions.db')
        cursor = conn.cursor()
        
        # Create sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_info TEXT,
                conversation_history TEXT,
                custom_facts TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at)')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully!")
    except sqlite3.Error as e:
        logger.error(f"Error initializing database: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
