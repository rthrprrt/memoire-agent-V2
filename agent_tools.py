# agent_tools.py (Version avec outils de plan implémentés)

import logging
from typing import List, Dict, Any, Optional
# Importer les composants nécessaires dont les outils dépendent
from vector_database import VectorDBManager
# Imports pour les nouveaux outils
from memory_manager import MemoryManager
from data_models import ReportPlan, ReportSection # Assurez-vous que ReportSection est bien défini dans data_models.py
from progress_tracker import ProgressTracker # Bien que non strictement utilisé pour pending, on peut le garder pour cohérence
import json # Pour charger/sauvegarder le plan (solution temporaire)
import config # Pour le chemin par défaut du plan

# Configuration du logger pour ce module
log = logging.getLogger(__name__)
# Assurer une configuration de base si non déjà faite par main.py
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')


# --- Initialisation des Dépendances ---
# ATTENTION: Crée des instances séparées de celles de main.py.
# La solution de recharger le fichier plan à chaque fois est un contournement.
# Une meilleure architecture utiliserait l'injection de dépendances.
try:
    vector_db = VectorDBManager()
    memory = MemoryManager() # Instance locale pour charger/sauvegarder le plan
    log.info("VectorDBManager and MemoryManager initialized successfully within agent_tools.")
except Exception as e:
    log.critical(f"FATAL: Failed to initialize dependencies in agent_tools: {e}", exc_info=True)
    vector_db = None
    memory = None

# --- Fonctions Outils Existantes ---

def search_journal_entries(query: str, k: int = 5) -> str:
    """
    Searches YEAR 2 journal entries vector store for chunks relevant to the query.
    Returns concatenated text of top 'k' chunks. Use for details on projects, tasks, skills, challenges.
    """
    log.info(f"[Tool Called] search_journal_entries(query='{query[:50]}...', k={k})")
    if not vector_db: return "Error: Vector DB not available for journal search."
    if not query or not isinstance(query, str): return "Error: Please provide a valid query string."
    if not isinstance(k, int) or k <= 0: k = 5; log.warning("Using default k=5 for journal search.")
    try:
        results: List[Dict[str, Any]] = vector_db.search_journals(query=query, k=k)
        if not results: return "No relevant journal entries found for this query."
        found_texts = [res.get("document", "") for res in results if res.get("document")]
        found_texts = [text for text in found_texts if text.strip()]
        if not found_texts: return "Found potentially relevant entries, but could not extract text content."
        context_string = "\n\n---\n\n".join(found_texts)
        log.info(f"Returning {len(found_texts)} journal chunks, length {len(context_string)}.")
        return context_string
    except Exception as e: log.error(f"Error during journal search: {e}", exc_info=True); return f"Error executing journal search: {e}"

def search_guidelines(topic: str, k: int = 3) -> str:
    """
    Searches the official guidelines vector store for chunks relevant to the topic/section title.
    Returns concatenated text of top 'k' chunks. Use for requirements, expected content, structure.
    """
    log.info(f"[Tool Called] search_guidelines(topic='{topic[:50]}...', k={k})")
    if not vector_db: return "Error: Vector DB not available for guideline search."
    if not topic or not isinstance(topic, str): return "Error: Please provide a valid topic string."
    if not isinstance(k, int) or k <= 0: k = 3; log.warning("Using default k=3 for guideline search.")
    try:
        results: List[Dict[str, Any]] = vector_db.search_references(query=topic, k=k)
        if not results: return "No relevant guidelines found for this topic."
        found_texts = [res.get("document", "") for res in results if res.get("document")]
        found_texts = [text for text in found_texts if text.strip()]
        if not found_texts: return "Found potentially relevant guidelines, but could not extract text content."
        context_string = "\n\n---\n\n".join(found_texts)
        log.info(f"Returning {len(found_texts)} guideline chunks, length {len(context_string)}.")
        return context_string
    except Exception as e: log.error(f"Error during guideline search: {e}", exc_info=True); return f"Error executing guidelines search: {e}"

# --- NOUVEAUX Outils pour le Plan ---

def get_report_plan_structure() -> str:
    """
    Retrieves the current structure of the report plan from the JSON file,
    including section titles, hierarchy (level), IDs, and current status.
    Use this to understand the overall report outline and see section statuses.
    """
    log.info("[Tool Called] get_report_plan_structure()")
    if not memory: return "Error: Memory Manager instance not available in agent_tools."

    # Recharger systématiquement depuis le fichier JSON pour refléter l'état le plus récent
    report_plan = memory.load_report_plan(config.DEFAULT_PLAN_FILE)

    if not report_plan or not report_plan.structure:
        log.warning(f"Report plan file not found or empty: {config.DEFAULT_PLAN_FILE}")
        return f"Error: Report plan file is missing or empty ({config.DEFAULT_PLAN_FILE}). Run 'create_plan' first."

    # Formatage pour le LLM
    structure_str = f"Report Title: {report_plan.title}\nSections:\n"
    processed_sections = set()

    def format_sections(sections: List[ReportSection], indent_level=0):
        nonlocal structure_str
        indent = "  " * indent_level
        if not sections: return # Cas où subsections est vide ou None
        for sec in sections:
            sec_id = getattr(sec, 'section_id', None)
            if sec_id and sec_id in processed_sections: continue # Eviter boucle infinie

            status_info = f"(Status: {getattr(sec, 'status', 'unknown')})"
            id_info = f"(ID: {sec_id})" if sec_id else "(ID: MISSING!)"
            title = getattr(sec, 'title', 'Untitled Section')
            structure_str += f"{indent}- {title} {id_info} {status_info}\n"

            if sec_id: processed_sections.add(sec_id)
            if hasattr(sec, 'subsections') and sec.subsections:
                format_sections(sec.subsections, indent_level + 1)

    try:
        format_sections(report_plan.structure)
        log.info(f"Returning report plan structure (length: {len(structure_str)}).")
        return structure_str.strip()
    except Exception as e:
        log.error(f"Error formatting plan structure: {e}", exc_info=True)
        return f"Error formatting plan: {e}"

def get_pending_sections() -> str:
    """
    Retrieves a list of section IDs marked as 'pending' or 'failed' in the report plan file.
    Returns a comma-separated list of section IDs, or a specific message if none are pending/failed.
    """
    log.info("[Tool Called] get_pending_sections()")
    if not memory: return "Error: Memory Manager not available."

    report_plan = memory.load_report_plan(config.DEFAULT_PLAN_FILE) # Recharger
    if not report_plan: return "Error: Report plan not loaded."

    try:
        pending_ids = []
        pending_statuses = {"pending", "failed"}
        processed_check = set()

        def find_pending_ids(sections: List[ReportSection]):
             if not sections: return
             for section in sections:
                 sec_id = getattr(section, 'section_id', None)
                 if not sec_id or sec_id in processed_check: continue

                 current_status = getattr(section, 'status', 'pending') # Défaut à pending
                 if current_status in pending_statuses:
                     pending_ids.append(sec_id)

                 processed_check.add(sec_id)
                 if hasattr(section, 'subsections') and section.subsections:
                     find_pending_ids(section.subsections)

        find_pending_ids(report_plan.structure)

        if not pending_ids:
            log.info("No pending or failed sections found in the plan.")
            return "No sections are currently marked as pending or failed."
        else:
            result = ", ".join(pending_ids)
            log.info(f"Found pending/failed sections: {result}")
            return result
    except Exception as e:
        log.error(f"Error getting pending sections: {e}", exc_info=True)
        return f"Error getting pending sections: {e}"

def update_section_status(section_id: str, new_status: str) -> str:
     """
     Updates the status of a specific section (by section_id) in the report plan JSON file.
     Allowed statuses: 'pending', 'drafting', 'drafted', 'failed', 'reviewing', 'approved'.

     Args:
         section_id: The unique ID of the section to update.
         new_status: The new status to set.

     Returns:
         Confirmation message ("Success: ...") or error message ("Error: ...").
     """
     log.info(f"[Tool Called] update_section_status(section_id='{section_id}', new_status='{new_status}')")
     if not memory: return "Error: Memory Manager not available."

     VALID_STATUSES = {"pending", "drafting", "drafted", "failed", "reviewing", "approved"}
     if new_status not in VALID_STATUSES:
         return f"Error: Invalid status '{new_status}'. Allowed: {', '.join(VALID_STATUSES)}"
     if not section_id or not isinstance(section_id, str):
         return f"Error: Invalid or missing section_id provided."

     # Charger, Modifier, Sauvegarder
     report_plan = memory.load_report_plan(config.DEFAULT_PLAN_FILE)
     if not report_plan: return "Error: Report plan could not be loaded."

     updated = False
     processed_update = set()
     try:
         def find_and_update(sections: List[ReportSection]) -> bool:
             nonlocal updated
             if not sections: return False
             for section in sections:
                 current_sec_id = getattr(section, 'section_id', None)
                 if current_sec_id and current_sec_id not in processed_update:
                     if current_sec_id == section_id:
                         log.info(f"Updating status for section '{section_id}' ('{getattr(section, 'title', 'N/A')}') to '{new_status}'.")
                         section.status = new_status
                         updated = True
                         processed_update.add(current_sec_id)
                         return True # Trouvé et mis à jour

                     processed_update.add(current_sec_id)
                     if hasattr(section, 'subsections') and section.subsections:
                         if find_and_update(section.subsections):
                             return True # Trouvé dans sous-section
             return False

         find_and_update(report_plan.structure)

         if updated:
             memory.save_report_plan(report_plan, config.DEFAULT_PLAN_FILE)
             return f"Success: Status for section '{section_id}' updated to '{new_status}' in the plan file."
         else:
             log.warning(f"Section ID '{section_id}' not found for update.")
             return f"Error: Section ID '{section_id}' not found in the plan."
     except Exception as e:
          log.error(f"Error updating section status for '{section_id}': {e}", exc_info=True)
          return f"Error updating status for section '{section_id}': {e}"