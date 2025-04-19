# llm_interface.py (Version pour LangChain avec Google Gemini - Vérifiée)

import google.generativeai as genai
import config
from typing import List, Dict, Optional, Any
import logging
import time
from langchain_google_genai import ChatGoogleGenerativeAI
# Assurez-vous que typing.Optional est bien importé
from typing import Optional

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')
log = logging.getLogger(__name__)

# Configuration Globale de l'API Google
try:
    if not config.GOOGLE_API_KEY: raise ValueError("GOOGLE_API_KEY not found.")
    genai.configure(api_key=config.GOOGLE_API_KEY)
    log.info("Google AI (Gemini) API configured successfully.")
except Exception as e_cfg: log.critical(f"Failed to configure Google AI API: {e_cfg}", exc_info=True); raise

# Fonction pour obtenir l'instance LLM LangChain
def get_langchain_llm(temperature: float = 0.5, max_tokens: Optional[int] = 2048):
    """Crée et retourne une instance du LLM Gemini configurée pour LangChain."""
    if not config.GOOGLE_API_KEY: log.critical("GOOGLE_API_KEY missing."); return None
    try:
        llm = ChatGoogleGenerativeAI(
            model=config.GEMINI_CHAT_MODEL_NAME,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=temperature,
            max_output_tokens=max_tokens,
            convert_system_message_to_human=True
        )
        log.info(f"LangChain LLM instance created for model: {config.GEMINI_CHAT_MODEL_NAME}")
        return llm
    except Exception as e: log.critical(f"Failed to init LangChain LLM: {e}", exc_info=True); return None