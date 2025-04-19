# llm_interface.py (Version avec import Ollama corrigé)

import google.generativeai as genai
import config
from typing import List, Dict, Optional, Any
import logging
import time
from langchain_google_genai import ChatGoogleGenerativeAI
# --- MODIFICATION : Import corrigé pour Ollama Chat Model ---
from langchain_community.chat_models.ollama import ChatOllama
# --- FIN MODIFICATION ---
from typing import Optional

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')
log = logging.getLogger(__name__)

# --- Configuration Google (si utilisé) ---
def configure_google_api():
    """Configure l'API Google si nécessaire."""
    if config.LLM_PROVIDER == "google":
        try:
            if not config.GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY non trouvée dans config/env pour le fournisseur 'google'.")
            genai.configure(api_key=config.GOOGLE_API_KEY)
            log.info("API Google AI (Gemini) configurée avec succès.")
            return True
        except Exception as e_cfg:
            log.critical(f"Échec de la configuration de l'API Google AI : {e_cfg}", exc_info=True)
            return False
    return True # Pas besoin si pas google

if not configure_google_api():
     log.warning("Impossible de configurer l'API Google. L'utilisation de Gemini échouera.")


# --- Fonction pour obtenir l'instance LLM Google LangChain ---
def get_google_llm(temperature: float = 0.5, max_tokens: Optional[int] = 2048):
    """Crée et retourne une instance du LLM Gemini configurée pour LangChain."""
    if not config.GOOGLE_API_KEY:
        log.critical("Clé API Google manquante pour get_google_llm.")
        return None
    try:
        llm = ChatGoogleGenerativeAI(
            model=config.GEMINI_CHAT_MODEL_NAME,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=temperature,
            max_output_tokens=max_tokens,
            convert_system_message_to_human=True
        )
        log.info(f"Instance LLM LangChain (Google) créée pour le modèle : {config.GEMINI_CHAT_MODEL_NAME}")
        return llm
    except Exception as e:
        log.critical(f"Échec de l'initialisation du LLM LangChain (Google) : {e}", exc_info=True)
        return None

# --- Fonction pour obtenir l'instance LLM Ollama LangChain ---
def get_ollama_llm(temperature: float = 0.5):
    """Crée et retourne une instance d'un LLM Ollama local configuré pour LangChain."""
    if not config.OLLAMA_MODEL_NAME:
        log.critical("Nom du modèle Ollama (OLLAMA_MODEL_NAME) manquant dans la configuration.")
        return None
    try:
        # --- MODIFICATION : Utiliser ChatOllama ---
        llm = ChatOllama(
            base_url=config.OLLAMA_BASE_URL,
            model=config.OLLAMA_MODEL_NAME,
            temperature=temperature,
            # request_timeout=120 # Décommenter si nécessaire
        )
        # --- FIN MODIFICATION ---

        try:
             # Test simple pour vérifier la connexion et le modèle
             log.info(f"Test de connexion à Ollama ({config.OLLAMA_BASE_URL}) avec le modèle '{config.OLLAMA_MODEL_NAME}'...")
             llm.invoke("Bonjour") # Fait un appel rapide
             log.info(f"Instance LLM LangChain (Ollama) créée et connectée pour le modèle : {config.OLLAMA_MODEL_NAME} sur {config.OLLAMA_BASE_URL}")
        except Exception as e_invoke:
             log.error(f"Échec de la connexion au serveur Ollama ({config.OLLAMA_BASE_URL}) pour le modèle '{config.OLLAMA_MODEL_NAME}'. Vérifiez que Ollama est lancé et que le modèle est téléchargé (`ollama pull {config.OLLAMA_MODEL_NAME}`). Erreur: {e_invoke}", exc_info=True)
             return None

        return llm
    except Exception as e:
        log.critical(f"Échec de l'initialisation du LLM LangChain (Ollama) : {e}", exc_info=True)
        return None


# --- Fonction unifiée pour obtenir le LLM basé sur la config ---
def get_configured_llm(provider: Optional[str] = None, temperature: float = 0.5, max_tokens: Optional[int] = 2048):
    """
    Obtient l'instance LLM LangChain appropriée (Google ou Ollama)
    basée sur le fournisseur spécifié ou celui dans config.py.
    """
    selected_provider = provider if provider else config.LLM_PROVIDER
    log.info(f"Tentative de chargement du LLM pour le fournisseur : '{selected_provider}'")

    if selected_provider == "google":
        return get_google_llm(temperature=temperature, max_tokens=max_tokens)
    elif selected_provider == "ollama":
        return get_ollama_llm(temperature=temperature)
    else:
        log.error(f"Fournisseur LLM non supporté : '{selected_provider}'. Vérifiez config.py ou l'argument --llm.")
        raise ValueError(f"Fournisseur LLM non supporté : {selected_provider}")