# agent_tools.py

import logging
from typing import List, Dict, Any, Optional
# Importer les composants nécessaires dont les outils dépendent
from vector_database import VectorDBManager
# Importer d'autres modules si nécessaire (ex: report_planner, progress_tracker)
# from report_planner import ReportPlanner
# from memory_manager import MemoryManager

log = logging.getLogger(__name__)

# Initialiser les dépendances nécessaires UNE SEULE FOIS (ou les passer en argument)
# C'est mieux de les passer en argument pour éviter les états globaux,
# mais pour simplifier ici, on les initialise.
# Attention : S'assurer que l'initialisation est compatible si ce module est importé ailleurs.
try:
    vector_db = VectorDBManager()
    # memory = MemoryManager() # Si les outils doivent accéder à la mémoire (plan, etc.)
    # planner = ReportPlanner() # Si des outils doivent manipuler le plan
except Exception as e:
    log.critical(f"Failed to initialize dependencies for agent_tools: {e}", exc_info=True)
    # Gérer l'erreur de manière appropriée, peut-être en levant une exception
    vector_db = None # Ou définir des versions 'dummy' pour permettre l'importation ?

# --- Définition des Outils ---

def search_journal_entries(query: str, k: int = 5) -> str:
    """
    Searches the apprentice's journal entries for passages relevant to the given query.
    Returns the top k relevant text chunks found. Use this to find details about specific projects,
    tasks, skills learned, or challenges faced during the apprenticeship year 2.
    Provide a specific and detailed query.
    """
    log.info(f"[Tool Called] search_journal_entries(query='{query[:50]}...', k={k})")
    if not vector_db:
        return "Error: Vector Database is not available."
    if not query:
        return "Error: Please provide a specific query."
    try:
        results = vector_db.search_journals(query=query, k=k)
        if not results:
            return "No relevant journal entries found for this query."

        # Formatter les résultats pour le LLM (concaténer les documents)
        context = "\n---\n".join([res.get("document", "") for res in results])
        return context
    except Exception as e:
        log.error(f"Error during search_journal_entries: {e}", exc_info=True)
        return f"Error executing journal search: {e}"

def search_guidelines(topic: str, k: int = 3) -> str:
    """
    Searches the official apprenticeship report guidelines (PDF content) for information
    relevant to the given topic or section title. Use this to check the requirements,
    expected content, or structure for a specific part of the report.
    Provide the topic or section title as the query.
    """
    log.info(f"[Tool Called] search_guidelines(topic='{topic[:50]}...', k={k})")
    if not vector_db:
        return "Error: Vector Database is not available."
    if not topic:
        return "Error: Please provide a topic or section title."
    try:
        # Utiliser la nouvelle méthode pour chercher dans les références
        results = vector_db.search_references(query=topic, k=k)
        if not results:
            return "No relevant guidelines found for this topic."

        # Formatter les résultats
        context = "\n---\n".join([res.get("document", "") for res in results])
        return context
    except Exception as e:
        log.error(f"Error during search_guidelines: {e}", exc_info=True)
        return f"Error executing guidelines search: {e}"

# --- Outils Potentiels Supplémentaires (à implémenter si besoin) ---

# def get_report_plan_structure() -> str:
#     """Returns the current structure (section titles and hierarchy) of the report plan."""
#     log.info("[Tool Called] get_report_plan_structure()")
#     if not memory or not memory.get_report_plan():
#         return "Error: Report plan not loaded or available."
#     plan = memory.get_report_plan()
#     # Formatter la structure pour le LLM (ex: liste indentée)
#     structure_str = ""
#     def format_sections(sections, indent=""):
#         nonlocal structure_str
#         for sec in sections:
#             structure_str += f"{indent}- {sec.title} (Status: {sec.status})\n"
#             if sec.subsections:
#                 format_sections(sec.subsections, indent + "  ")
#     format_sections(plan.structure)
#     return structure_str.strip() if structure_str else "Plan structure is empty."

# def get_pending_sections() -> str:
#     """Returns a comma-separated list of report sections that are not yet drafted or completed."""
#     log.info("[Tool Called] get_pending_sections()")
#     # Nécessiterait ProgressTracker et MemoryManager
#     # tracker = ProgressTracker()
#     # plan = memory.get_report_plan()
#     # if not plan: return "Error: Report plan not loaded."
#     # pending = tracker.get_pending_sections(plan)
#     # return ", ".join(pending) if pending else "No pending sections found."
#     return "Placeholder: Pending sections tool not fully implemented."

# def summarize_journal_entries(start_date: str, end_date: str) -> str:
#      """Summarizes the key activities and learnings from journal entries within a specific date range."""
#      log.info(f"[Tool Called] summarize_journal_entries(start_date={start_date}, end_date={end_date})")
#      # Cette fonction serait complexe:
#      # 1. Filtrer les entrées par date (via MemoryManager ou recherche DB avec filtre date)
#      # 2. Concaténer le texte des entrées filtrées
#      # 3. Envoyer le texte concaténé (ou des chunks) à l'API DeepSeek pour résumé
#      return "Placeholder: Journal summarization tool not implemented."

# def search_year1_summary(query: str, k: int = 3) -> str:
#     """Searches the summaries or documents related to the first apprenticeship year."""
#     log.info(f"[Tool Called] search_year1_summary(query='{query[:50]}...', k={k})")
#     # Supposerait une collection ou des métadonnées dédiées pour l'année 1
#     # if not vector_db: return "Error: Vector DB not available."
#     # results = vector_db.search_collection("year1_docs", query, k) # Exemple
#     return "Placeholder: Search for year 1 not implemented. No data available yet."