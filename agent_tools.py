# agent_tools.py (Version finale avec tous les outils implémentés)

import logging
from typing import List, Dict, Any, Optional
# Imports nécessaires
from vector_database import VectorDBManager
from memory_manager import MemoryManager
from data_models import ReportPlan, ReportSection
# ProgressTracker n'est plus nécessaire ici car la logique est simple
# from progress_tracker import ProgressTracker
import json # Utilisé par MemoryManager implicitement
import config # Pour le chemin par défaut du plan
from llm_interface import GeminiLLM # Importé pour l'outil draft_single_section

# Configuration du logger
log = logging.getLogger(__name__)
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')

# --- Initialisation des Dépendances ---
# ATTENTION : Instances locales. Rechargement du plan depuis fichier nécessaire dans les outils.
try:
    vector_db = VectorDBManager()
    memory = MemoryManager() # Utilisé pour charger/sauvegarder le plan dans les outils
    llm_tool_instance = GeminiLLM() # Instance LLM pour l'outil de rédaction
    log.info("VectorDBManager, MemoryManager, and GeminiLLM initialized within agent_tools.")
except Exception as e:
    log.critical(f"FATAL: Failed to initialize dependencies in agent_tools: {e}", exc_info=True)
    vector_db = None
    memory = None
    llm_tool_instance = None

# --- Fonctions Outils ---

def search_journal_entries(query: str, k: int = 5) -> str:
    """
    Searches YEAR 2 journal entries vector store for chunks relevant to the query.
    Returns concatenated text of top 'k' chunks, separated by '---'.
    Use for details on projects, tasks, skills, challenges from Year 2.
    """
    log.info(f"[Tool Called] search_journal_entries(query='{query[:50]}...', k={k})")
    if not vector_db: return "Error: Vector DB unavailable."
    if not query or not isinstance(query, str): return "Error: Invalid query string."
    if not isinstance(k, int) or k <= 0: k = 5; log.warning("Using default k=5.")
    try:
        results = vector_db.search_journals(query=query, k=k)
        if not results: return "No relevant journal entries found."
        found_texts = [res.get("document", "") for res in results if res.get("document")]
        found_texts = [text for text in found_texts if text.strip()]
        if not found_texts: return "Found entries, but no text content."
        context_string = "\n\n---\n\n".join(found_texts)
        log.info(f"Returning {len(found_texts)} journal chunks, len {len(context_string)}.")
        return context_string
    except Exception as e: log.error(f"Search error: {e}", exc_info=True); return f"Error: {e}"

def search_guidelines(topic: str, k: int = 3) -> str:
    """
    Searches the official guidelines vector store for chunks relevant to the topic/section title.
    Returns concatenated text of top 'k' chunks, separated by '---'.
    Use for requirements, expected content, structure.
    """
    log.info(f"[Tool Called] search_guidelines(topic='{topic[:50]}...', k={k})")
    if not vector_db: return "Error: Vector DB unavailable."
    if not topic or not isinstance(topic, str): return "Error: Invalid topic string."
    if not isinstance(k, int) or k <= 0: k = 3; log.warning("Using default k=3.")
    try:
        results = vector_db.search_references(query=topic, k=k)
        if not results: return "No relevant guidelines found."
        found_texts = [res.get("document", "") for res in results if res.get("document")]
        found_texts = [text for text in found_texts if text.strip()]
        if not found_texts: return "Found guidelines, but no text content."
        context_string = "\n\n---\n\n".join(found_texts)
        log.info(f"Returning {len(found_texts)} guideline chunks, len {len(context_string)}.")
        return context_string
    except Exception as e: log.error(f"Search error: {e}", exc_info=True); return f"Error: {e}"

def get_report_plan_structure() -> str:
    """
    Retrieves the current report plan structure from the JSON file (titles, hierarchy, IDs, status).
    Use to understand the outline and section statuses.
    """
    log.info("[Tool Called] get_report_plan_structure()")
    if not memory: return "Error: Memory Manager unavailable."
    report_plan = memory.load_report_plan(config.DEFAULT_PLAN_FILE) # Recharge depuis fichier
    if not report_plan or not report_plan.structure: return f"Error: Report plan missing or empty ({config.DEFAULT_PLAN_FILE}). Run 'create_plan'."

    structure_str = f"Report Title: {report_plan.title}\nSections:\n"
    processed = set()
    def format_sections(sections: List[ReportSection], indent_level=0):
        nonlocal structure_str
        indent = "  " * indent_level
        if not sections: return
        for sec in sections:
            sec_id = getattr(sec, 'section_id', None)
            if sec_id and sec_id in processed: continue
            status = f"(Status: {getattr(sec, 'status', 'unknown')})"
            id_info = f"(ID: {sec_id})" if sec_id else "(ID: MISSING!)"
            title = getattr(sec, 'title', 'Untitled')
            structure_str += f"{indent}- {title} {id_info} {status}\n"
            if sec_id: processed.add(sec_id)
            if hasattr(sec, 'subsections') and sec.subsections: format_sections(sec.subsections, indent_level + 1)
    try: format_sections(report_plan.structure); return structure_str.strip()
    except Exception as e: log.error(f"Error formatting plan: {e}", exc_info=True); return f"Error: {e}"

def get_pending_sections() -> str:
    """
    Retrieves section IDs marked as 'pending' or 'failed' from the report plan file.
    Returns a comma-separated list of IDs, or a message if none are pending/failed.
    """
    log.info("[Tool Called] get_pending_sections()")
    if not memory: return "Error: Memory Manager unavailable."
    report_plan = memory.load_report_plan(config.DEFAULT_PLAN_FILE) # Recharge
    if not report_plan: return "Error: Report plan not loaded."
    try:
        pending_ids = []; pending_statuses = {"pending", "failed"}; processed = set()
        def find_pending_ids(sections: List[ReportSection]):
             if not sections: return
             for section in sections:
                 sec_id = getattr(section, 'section_id', None)
                 if not sec_id or sec_id in processed: continue
                 current_status = getattr(section, 'status', 'pending')
                 if current_status in pending_statuses: pending_ids.append(sec_id)
                 processed.add(sec_id)
                 if hasattr(section, 'subsections') and section.subsections: find_pending_ids(section.subsections)
        find_pending_ids(report_plan.structure)
        if not pending_ids: log.info("No pending/failed sections found."); return "No sections are currently marked as pending or failed."
        else: result = ", ".join(pending_ids); log.info(f"Pending/failed sections: {result}"); return result
    except Exception as e: log.error(f"Error getting pending sections: {e}", exc_info=True); return f"Error: {e}"

def update_section_status(section_id: str, new_status: str) -> str:
     """
     Updates the status of a section (by ID) in the report plan JSON file.
     Allowed statuses: 'pending', 'drafting', 'drafted', 'failed', 'reviewing', 'approved'.
     """
     log.info(f"[Tool Called] update_section_status(id='{section_id}', status='{new_status}')")
     if not memory: return "Error: Memory Manager unavailable."
     VALID_STATUSES = {"pending", "drafting", "drafted", "failed", "reviewing", "approved"}
     if new_status not in VALID_STATUSES: return f"Error: Invalid status '{new_status}'. Allowed: {', '.join(VALID_STATUSES)}"
     if not section_id or not isinstance(section_id, str): return f"Error: Invalid section_id."

     report_plan = memory.load_report_plan(config.DEFAULT_PLAN_FILE)
     if not report_plan: return "Error: Report plan could not be loaded."
     updated = False; processed = set()
     try:
         def find_and_update(sections: List[ReportSection]) -> bool:
             nonlocal updated
             if not sections: return False
             for section in sections:
                 current_sec_id = getattr(section, 'section_id', None)
                 if current_sec_id and current_sec_id not in processed:
                     if current_sec_id == section_id:
                         log.info(f"Updating status for '{section_id}' ('{getattr(section, 'title', 'N/A')}') to '{new_status}'.")
                         section.status = new_status; updated = True; processed.add(current_sec_id); return True
                     processed.add(current_sec_id)
                     if hasattr(section, 'subsections') and section.subsections:
                         if find_and_update(section.subsections): return True
             return False
         find_and_update(report_plan.structure)
         if updated: memory.save_report_plan(report_plan, config.DEFAULT_PLAN_FILE); return f"Success: Status for '{section_id}' updated to '{new_status}'."
         else: log.warning(f"Section ID '{section_id}' not found."); return f"Error: Section ID '{section_id}' not found."
     except Exception as e: log.error(f"Error updating status for '{section_id}': {e}", exc_info=True); return f"Error: {e}"

def draft_single_section(section_id: str) -> str:
    """
    Drafts content for a specific section ID. It finds the section title, gets context
    from guidelines and journals using other tools, then calls the LLM (Gemini) to write the draft.
    """
    log.info(f"[Tool Called] draft_single_section(section_id='{section_id}')")
    if not memory or not vector_db or not llm_tool_instance: return "Error: Core components unavailable in tool."
    if not section_id: return "Error: section_id is required."

    # 1. Get Section Title from Plan
    report_plan = memory.load_report_plan(config.DEFAULT_PLAN_FILE)
    if not report_plan: return f"Error: Could not load plan for section '{section_id}'."
    target_section: Optional[ReportSection] = None; section_title = "Unknown Section"
    try:
        processed_find = set()
        def find_section(sections: List[ReportSection]):
            nonlocal target_section, section_title
            if not sections: return None
            for sec in sections:
                current_sec_id = getattr(sec, 'section_id', None)
                if current_sec_id and current_sec_id not in processed_find:
                     if current_sec_id == section_id: target_section = sec; section_title = getattr(sec, 'title', section_id); return sec
                     processed_find.add(current_sec_id)
                     if hasattr(sec, 'subsections') and sec.subsections:
                         found_in_sub = find_section(sec.subsections);
                         if found_in_sub: return found_in_sub
            return None
        find_section(report_plan.structure)
    except Exception as e_find: log.error(f"Error finding section '{section_id}': {e_find}"); return f"Error finding section '{section_id}'."
    if not target_section: return f"Error: Section ID '{section_id}' not found in plan."
    log.info(f"Found section to draft: '{section_title}' (ID: {section_id})")

    # 2. Gather Context
    guideline_context = ""; journal_context = ""
    try:
        log.info(f"Searching guidelines context for '{section_title}'..."); guideline_context = search_guidelines(topic=section_title, k=3)
        log.info(f"Searching journal context for '{section_title}'..."); journal_context = search_journal_entries(query=section_title, k=7)
        log.info("Context gathering complete.")
    except Exception as e_ctx: log.error(f"Error gathering context for '{section_id}': {e_ctx}"); return f"Error gathering context: {e_ctx}"

    # 3. Call LLM for Drafting
    log.info(f"Calling LLM to draft section '{section_title}'...")
    try:
        drafting_instructions = f"Draft the content for the report section titled '{section_title}'. Use the provided context from guidelines and journals. Maintain a professional, academic tone. Synthesize information, do not just list excerpts."
        # Simplification: Envoyer le contexte comme un seul bloc, laisser Gemini trier.
        combined_context = f"GUIDELINES CONTEXT:\n{guideline_context if not guideline_context.startswith('Error:') else 'Not available.'}\n\nJOURNAL ENTRIES CONTEXT:\n{journal_context if not journal_context.startswith('Error:') else 'Not available.'}"
        # Utiliser l'instance LLM initialisée dans ce module
        drafted_content = llm_tool_instance.draft_report_section(section_title=section_title, context_chunks=[combined_context], instructions=drafting_instructions)

        if drafted_content and not drafted_content.startswith("Error:"):
            log.info(f"Successfully drafted section '{section_id}'. Length: {len(drafted_content)}.")
            return drafted_content # Retourne le texte généré
        else:
            log.error(f"LLM failed to draft section '{section_id}'. Response: {drafted_content}")
            return f"Error: LLM failed to generate draft for section '{section_id}'. Details: {drafted_content}"
    except Exception as e_draft: log.error(f"Error calling LLM draft for '{section_id}': {e_draft}"); return f"Error during LLM drafting: {e_draft}"