# llm_interface.py (Ajout LLM spécifique pour analyse JSON)

import google.generativeai as genai
import config
from typing import List, Dict, Optional, Any
import logging
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models.ollama import ChatOllama
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')
log = logging.getLogger(__name__)

# (configure_google_api et get_google_llm inchangés)
def configure_google_api():
    if config.LLM_PROVIDER == "google":
        try:
            if not config.GOOGLE_API_KEY: raise ValueError("GOOGLE_API_KEY non trouvée...")
            genai.configure(api_key=config.GOOGLE_API_KEY); log.info("API Google AI configurée."); return True
        except Exception as e_cfg: log.critical(f"Échec config API Google: {e_cfg}", exc_info=True); return False
    return True
if not configure_google_api(): log.warning("Impossible de configurer API Google.")

def get_google_llm(temperature: float = 0.5, max_tokens: Optional[int] = 2048):
    if not config.GOOGLE_API_KEY: log.critical("Clé API Google manquante."); return None
    try:
        llm = ChatGoogleGenerativeAI(model=config.GEMINI_CHAT_MODEL_NAME, google_api_key=config.GOOGLE_API_KEY, temperature=temperature, max_output_tokens=max_tokens, convert_system_message_to_human=True)
        log.info(f"Instance LLM Google créée: {config.GEMINI_CHAT_MODEL_NAME}"); return llm
    except Exception as e: log.critical(f"Échec init LLM Google: {e}", exc_info=True); return None

# --- MODIFICATION : Fonctions LLM Ollama distinctes ---

def get_ollama_llm_generic(temperature: float = 0.5, stop: Optional[List[str]] = None):
    """Fonction de base pour obtenir une instance ChatOllama."""
    if not config.OLLAMA_MODEL_NAME: log.critical("Nom modèle Ollama manquant."); return None
    try:
        num_cpu_threads = 12
        llm = ChatOllama(
            base_url=config.OLLAMA_BASE_URL,
            model=config.OLLAMA_MODEL_NAME,
            temperature=temperature,
            num_thread=num_cpu_threads,
            stop=stop, # Ajout du paramètre stop
            # num_ctx=4096, # Augmenter si nécessaire
            # request_timeout=120
        )
        # Test de connexion rapide
        try:
             llm.invoke("Test")
             log.info(f"Instance LLM Ollama ({config.OLLAMA_MODEL_NAME}) connectée.")
             return llm
        except Exception as e_invoke:
             log.error(f"Échec connexion Ollama ({config.OLLAMA_BASE_URL}, {config.OLLAMA_MODEL_NAME}): {e_invoke}", exc_info=True)
             return None
    except Exception as e:
        log.critical(f"Échec init LLM Ollama: {e}", exc_info=True); return None

def get_ollama_llm_for_analysis():
    """Instance Ollama optimisée pour l'extraction JSON."""
    log.info("Configuration LLM Ollama pour Analyse (temp=0.0, stop=<FIN_JSON>)...")
    return get_ollama_llm_generic(temperature=0.0, stop=["<FIN_JSON>"])

def get_ollama_llm_for_writing():
    """Instance Ollama pour la rédaction (température standard)."""
    log.info("Configuration LLM Ollama pour Rédaction (temp=0.5)...")
    return get_ollama_llm_generic(temperature=0.5)

# --- FIN MODIFICATION ---


# --- Fonction unifiée pour obtenir le LLM basé sur la config ---
# (Modifiée pour utiliser les nouvelles fonctions Ollama)
def get_configured_llm(purpose: str = "generic", provider: Optional[str] = None, temperature: float = 0.5, max_tokens: Optional[int] = 2048):
    """
    Obtient l'instance LLM appropriée basée sur le fournisseur et l'usage.
    'purpose' peut être 'analysis', 'writing', ou 'generic'.
    """
    selected_provider = provider if provider else config.LLM_PROVIDER
    log.info(f"Tentative chargement LLM pour fournisseur: '{selected_provider}', usage: '{purpose}'")

    if selected_provider == "google":
        # Google gère bien différentes tâches avec une seule instance pour l'instant
        return get_google_llm(temperature=temperature, max_tokens=max_tokens)
    elif selected_provider == "ollama":
        if purpose == "analysis":
            return get_ollama_llm_for_analysis()
        elif purpose == "writing":
            return get_ollama_llm_for_writing()
        else: # generic ou autre
            return get_ollama_llm_generic(temperature=temperature) # Utilise la température passée
    else:
        log.error(f"Fournisseur LLM non supporté : '{selected_provider}'.")
        raise ValueError(f"Fournisseur LLM non supporté : {selected_provider}")