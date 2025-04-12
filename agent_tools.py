# agent_tools.py

import logging
from typing import List, Dict, Any, Optional
# Importer les composants nécessaires dont les outils dépendent
from vector_database import VectorDBManager
# Potentiellement d'autres imports si d'autres outils sont ajoutés
# from memory_manager import MemoryManager

# Configuration du logger pour ce module
log = logging.getLogger(__name__)
# Assurer une configuration de base si non déjà faite par main.py (utile si testé seul)
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')


# --- Initialisation des Dépendances ---
# On initialise VectorDBManager ici pour que les fonctions outils puissent l'utiliser.
# Dans une application plus complexe, l'injection de dépendances serait préférable.
try:
    vector_db = VectorDBManager()
    log.info("VectorDBManager initialized successfully within agent_tools.")
except Exception as e:
    log.critical(f"FATAL: Failed to initialize VectorDBManager in agent_tools: {e}", exc_info=True)
    # Rendre les outils inutilisables si la DB n'est pas prête
    vector_db = None

# --- Définition des Fonctions Outils ---

def search_journal_entries(query: str, k: int = 5) -> str:
    """
    Searches the apprentice's YEAR 2 journal entries vector store for text chunks
    semantically relevant to the given query. Returns the concatenated text of
    the top 'k' most relevant chunks found, separated by '---'.
    Use this tool to find specific details, examples, or context about projects,
    tasks, skills learned, challenges faced, or people mentioned during the Year 2 apprenticeship.
    Provide a specific and descriptive query.

    Args:
        query: The natural language query describing the information needed.
        k: The maximum number of relevant chunks to return (default: 5).

    Returns:
        A string containing the concatenated relevant text chunks, or an
        informative message if no results are found or an error occurs.
    """
    log.info(f"[Tool Called] search_journal_entries(query='{query[:50]}...', k={k})")
    if not vector_db:
        log.error("search_journal_entries tool cannot execute: Vector Database not initialized.")
        return "Error: Vector Database is not available for journal search."
    if not query or not isinstance(query, str):
        log.warning("search_journal_entries tool called with empty or invalid query.")
        return "Error: Please provide a valid, specific query string."
    if not isinstance(k, int) or k <= 0:
        log.warning(f"search_journal_entries tool called with invalid k={k}. Using default k=5.")
        k = 5

    try:
        # Utiliser la méthode dédiée pour chercher dans les journaux
        results: List[Dict[str, Any]] = vector_db.search_journals(query=query, k=k)

        if not results:
            log.info("No relevant journal entries found for the query.")
            return "No relevant journal entries found for this query."

        # Extraire et formater le contenu des documents trouvés
        found_texts = [res.get("document", "") for res in results if res.get("document")]
        # Filtrer les textes vides au cas où
        found_texts = [text for text in found_texts if text.strip()]

        if not found_texts:
             log.info("Journal search returned results but contained no actual text content.")
             return "Found potentially relevant entries, but could not extract text content."

        # Concaténer les résultats avec un séparateur clair pour le LLM
        context_string = "\n\n---\n\n".join(found_texts)
        log.info(f"Returning {len(found_texts)} journal chunks, total length {len(context_string)} chars.")
        return context_string

    except Exception as e:
        log.error(f"Error during search_journal_entries execution: {e}", exc_info=True)
        # Retourner un message d'erreur clair au LLM orchestrateur
        return f"Error executing journal search: {e}"

def search_guidelines(topic: str, k: int = 3) -> str:
    """
    Searches the official apprenticeship report guidelines vector store (derived from the PDF)
    for sections relevant to the given topic or section title. Returns the concatenated text
    of the top 'k' most relevant guideline chunks found, separated by '---'.
    Use this to check requirements, expected content, or structure for a specific part of the report.
    Provide the topic or section title as the query.

    Args:
        topic: The topic or section title to search for in the guidelines.
        k: The maximum number of relevant chunks to return (default: 3).

    Returns:
        A string containing the concatenated relevant guideline text chunks, or an
        informative message if no results are found or an error occurs.
    """
    log.info(f"[Tool Called] search_guidelines(topic='{topic[:50]}...', k={k})")
    if not vector_db:
        log.error("search_guidelines tool cannot execute: Vector Database not initialized.")
        return "Error: Vector Database is not available for guideline search."
    if not topic or not isinstance(topic, str):
        log.warning("search_guidelines tool called with empty or invalid topic.")
        return "Error: Please provide a valid topic or section title string."
    if not isinstance(k, int) or k <= 0:
        log.warning(f"search_guidelines tool called with invalid k={k}. Using default k=3.")
        k = 3

    try:
        # Utiliser la méthode dédiée pour chercher dans les références
        results: List[Dict[str, Any]] = vector_db.search_references(query=topic, k=k)

        if not results:
            log.info("No relevant guidelines found for the topic.")
            return "No relevant guidelines found for this topic."

        # Extraire et formater le contenu
        found_texts = [res.get("document", "") for res in results if res.get("document")]
        found_texts = [text for text in found_texts if text.strip()]

        if not found_texts:
             log.info("Guideline search returned results but contained no actual text content.")
             return "Found potentially relevant guidelines, but could not extract text content."

        context_string = "\n\n---\n\n".join(found_texts)
        log.info(f"Returning {len(found_texts)} guideline chunks, total length {len(context_string)} chars.")
        return context_string

    except Exception as e:
        log.error(f"Error during search_guidelines execution: {e}", exc_info=True)
        return f"Error executing guidelines search: {e}"

# --- Autres Outils Potentiels (Placeholders/À implémenter) ---

# def get_report_plan_structure() -> str:
#     """Returns the current structure (section titles, hierarchy, status) of the report plan."""
#     # ... (Implémentation nécessiterait MemoryManager) ...
#     return "Placeholder: get_report_plan_structure tool not implemented."

# def get_pending_sections() -> str:
#     """Returns a comma-separated list of report sections that are not yet 'completed' or 'approved'."""
#     # ... (Implémentation nécessiterait MemoryManager, ProgressTracker) ...
#     return "Placeholder: get_pending_sections tool not implemented."

# def update_section_status(section_id: str, new_status: str) -> str:
#      """Updates the status of a specific section in the report plan."""
#      # ... (Implémentation nécessiterait MemoryManager) ...
#      # Retourner "Success" ou un message d'erreur
#      return "Placeholder: update_section_status tool not implemented."