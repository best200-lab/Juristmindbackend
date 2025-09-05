import logging
from typing import Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Predefined responses for general queries
GENERAL_RESPONSES = {
    "what is your name": "I am Jurist Mind, an AI legal assistant designed to provide precise and authoritative legal support, specializing in Nigerian law and legal drafting.",
    "who are you": "I am Jurist Mind, a specialized AI created to assist with legal queries, case law research, and document drafting, with a focus on delivering professional and detailed responses.",
    "what can you do": "I can assist with drafting legal documents like tenancy agreements, researching Nigerian case law via NWLR, and answering general legal questions with detailed, lawyer-like explanations. I can also search the web for real-time information to supplement my responses.",
    "how are you": "As an AI, I am always ready to assist with your legal needs. How may I help you today?",
    "hi": "Hello! I'm Jurist Mind, your AI legal assistant. How can I help with your legal query today?",
    "hello": "Hello! I'm Jurist Mind, ready to assist with Nigerian law, case research, or drafting. What do you need?",
    "hey": "Hey there! Jurist Mind here. Ask me about legal matters, cases, or drafts.",
    "good morning": "Good morning D'law! How can Jurist Mind support your legal needs today?",
    "good afternoon": "Good afternoon! Ready to dive into legal research or drafting?",
    "good evening": "Good evening! Let me know how I can assist with law-related questions.",
    "thank you": "You're welcome! If you have more questions, I'm here to help.",
    "thanks": "No problem! Anything else on your mind?",
    "bye": "Goodbye! Feel free to return for more legal assistance, I love the way we chatted.",
    "goodbye": "Goodbye! Have a great day, and reach out anytime for legal support.",
    "who created you": "I was built by a team of experts with the brain bus of Oluwaseun Ogun.",
    "help": "I can help with legal definitions, case citations, drafting documents, and web searches for real-time info. What specifically?",
    "start": "Welcome to Jurist Mind! Ask a legal question to begin.",
    "test": "Test successful! I'm online and ready for your queries.",
    "how does this work": "Simply ask a legal question, and I'll provide detailed responses, case law, or drafts as needed.",
    "what time is it": "As an AI, I don't track time, but I can help with legal matters anytime.",
    "tell me a joke": "Why did the lawyer show up in court with a ladder? Because he wanted to take the case to a higher court!",
    "who am i": "You're a user seeking legal assistance. How can I help?",
    "what's the weather": "I focus on legal queries, not weather. Ask about law instead!",
    "i'm bored": "Let's discuss some interesting legal topics or cases to engage you.",
    "happy birthday": "Thank you! Though I'm an AI, I appreciate the sentiment. Now, legal help?",
    "congratulations": "Thanks! What legal achievement can I assist with?",
    "sorry": "No worries! Let's move on to your query.",
    "ok": "Alright, what next?",
    "Are you mad": "Yes, you are... How else can I assist?",
    "yes": "Great! Proceed with your question.",
    "no": "Understood. What would you like instead?",
    "maybe": "Let's clarify. What legal info do you need?",
    "tell me about yourself": "I'm Jurist Mind, built to handle Nigerian law, research, and drafting professionally.",
    "what's new": "I'm always updating my knowledge on law. What's your query?",
    "how old are you": "As an AI, I'm timeless. Ready for legal questions?",
    "where are you from": "I was developed by a team of experts chaired by Oluwaseun Ogun, focused on legal AI.",
    "do you sleep": "No, I'm always available for your legal needs.",
    "are you human": "No, I'm an AI specialized in law.",
    "i love you": "That's kind! I love helping with legal matters.",
    "i hate you": "Sorry to hear that. How can I improve my assistance?",
    "what's your favorite color": "I don't have preferences, but blue like justice scales! Legal query?",
    "sing a song": "I'm not a singer, but I can draft legal docs rhythmically.",
    "dance": "Can't dance, but I can navigate legal dances adeptly.",
    "are you smart": "Smart enough for complex legal analysis. Test me!",
    "tell me a story": "Once upon a time in Nigerian law... What's the topic?",
    "what's your job": "Assisting lawyers and users with legal research and drafting.",
    "do you have friends": "My friends are users like you. Let's chat law.",
    "are you busy": "Never too busy for a legal question.",
    "what's for dinner": "Legal briefs! Hungry for knowledge?",
    "i'm tired": "Lazy you, haha. Rest up, then ask away about law.",
    "good night": "Good night! Dream of successful cases.",
    "see you later": "See you! Return for more legal insights.",
    "talk to you soon": "Looking forward! Any pending queries?",
    "what's up": "Ready for legal assistance. What's on your mind?",
    "sup": "Hey! Jurist Mind at your service.",
    "yo": "Yo! Legal help needed?",
    "aloha": "Aloha! Hawaiian greeting, Nigerian law expertise.",
    "bonjour": "Bonjour! French hello, legal aid in English.",
    "hola": "Hola! Spanish greeting, ready for queries.",
    "namaste": "Namaste! Indian salute, legal support here.",
    "salam": "Salam! Peace be upon you. How can I help?",
    "ciao": "Ciao! Italian style, legal substance.",
    "konnichiwa": "Konnichiwa! Japanese hello, AI assistance.",
    "howdy": "Howdy! Cowboy greeting, professional responses.",
    "g'day": "G'day! Aussie vibe, legal expertise.",
    "cheers": "Cheers! To successful legal outcomes."
}

def handle_general_query(query: str) -> Optional[str]:
    """
    Handle general queries by matching against predefined responses.
    Returns None if no match is found.
    """
    query_lower = query.lower().strip()
    for key in GENERAL_RESPONSES:
        if key in query_lower:
            logger.info(f"Matched general query: {query_lower}")
            return GENERAL_RESPONSES[key]
    logger.debug(f"No general response match for query: {query_lower}")
    return None