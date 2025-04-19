# agent_tools.py (Version avec sauvegarde du contenu dans DraftSingleSectionTool)

import logging
from typing import List, Dict, Any, Optional, Type
import time # <-- Importer time

# Imports nécessaires
from vector_database import VectorDBManager
from memory_manager import MemoryManager
from data_models import ReportPlan, ReportSection
import config
import json
from llm_interface import get_langchain_llm
# Imports LangChain
from langchain_core.tools import BaseTool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel

# Configuration du logger
log = logging.getLogger(__name__)
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')

# --- Pas d'initialisation globale des dépendances ---

# --- Schémas d'Arguments Pydantic ---
class SearchInput(BaseModel): query: str = Field(...); k: int = Field(default=5)
class GuidelineSearchInput(BaseModel): topic: str = Field(...); k: int = Field(default=3)
class SectionStatusInput(BaseModel): section_id: str = Field(...); new_status: str = Field(...)
class DraftSectionInput(BaseModel): section_id: str = Field(...)


# --- Définition des Outils au Format LangChain ---

class SearchJournalEntriesTool(BaseTool):
    name: str = "search_journal_entries"; description: str = "Searches YEAR 2 journal entries based on a query. Input: query string. Returns relevant text snippets."
    args_schema: Type[BaseModel] = SearchInput
    vector_db: VectorDBManager
    def _run(self, query: str, k: int = 5) -> str:
        log.info(f"[Tool] {self.name}(query='{query[:50]}...', k={k})");
        if not self.vector_db: return "Error: Vector DB unavailable.";
        try:
            r = self.vector_db.search_journals(query=query, k=k);
            return "\n\n---\n\n".join([d.get("document","") for d in r]) if r else "No relevant journals found."
        except Exception as e:
            log.error(f"{self.name} Error: {e}",exc_info=True);
            return f"Error during journal search: {e}"

class SearchGuidelinesTool(BaseTool):
    name: str = "search_guidelines"; description: str = "Searches official guidelines based on a topic. Input: topic string. Returns relevant guideline snippets."
    args_schema: Type[BaseModel] = GuidelineSearchInput
    vector_db: VectorDBManager
    def _run(self, topic: str, k: int = 3) -> str:
        log.info(f"[Tool] {self.name}(topic='{topic[:50]}...', k={k})");
        if not self.vector_db: return "Error: Vector DB unavailable.";
        try:
            r = self.vector_db.search_references(query=topic, k=k);
            return "\n\n---\n\n".join([d.get("document","") for d in r]) if r else "No relevant guidelines found."
        except Exception as e:
            log.error(f"{self.name} Error: {e}",exc_info=True);
            return f"Error during guideline search: {e}"

class GetReportPlanStructureTool(BaseTool):
    name: str = "get_report_plan_structure"; description: str = "Retrieves the current report plan structure including section titles, IDs, and statuses. Takes no arguments."
    memory_manager: MemoryManager
    def _run(self, *args: Any, **kwargs: Any) -> str:
        log.info(f"[Tool] {self.name}()")
        if not self.memory_manager: return "Error: Memory Manager unavailable."
        try:
            report_plan = self.memory_manager.load_report_plan(config.DEFAULT_PLAN_FILE)
            if not report_plan or not report_plan.structure:
                return f"Error: Report plan missing or empty ({config.DEFAULT_PLAN_FILE})."

            structure_str = f"Report Title: {report_plan.title}\nSections:\n"
            processed = set() # Pour éviter les boucles infinies si structure étrange

            def format_sections(sections: List[ReportSection], indent_level=0):
                nonlocal structure_str
                indent = "  " * indent_level
                if not sections: return

                for section in sections:
                    sec_id = getattr(section, 'section_id', 'NO_ID')
                    if sec_id in processed: continue
                    processed.add(sec_id)

                    title = getattr(section, 'title', 'Untitled')
                    status = getattr(section, 'status', 'unknown')
                    level = getattr(section, 'level', indent_level + 1) # Estimer si non présent
                    structure_str += f"{indent}- L{level}: {title} (ID: {sec_id}, Status: {status})\n"

                    if hasattr(section, 'subsections') and section.subsections:
                        format_sections(section.subsections, indent_level + 1)

            format_sections(report_plan.structure)
            return structure_str.strip()
        except Exception as e:
            log.error(f"{self.name} Error: {e}", exc_info=True)
            return f"Error retrieving report plan structure: {e}"

class GetPendingSectionsTool(BaseTool):
    name: str = "get_pending_sections";
    description: str = (
        "Retrieves the ID of the *first* section marked as 'pending' or 'failed' from the report plan file. "
        "Use this to identify the immediate next section to work on. Takes no arguments. "
        "Returns only the section ID string if found, or the specific message 'COMPLETED - No pending sections found.' if none are pending/failed."
    )
    memory_manager: MemoryManager

    def _run(self, *args: Any, **kwargs: Any) -> str:
        log.info(f"[Tool] {self.name}()")
        if not self.memory_manager: return "Error: Memory Manager unavailable."
        try:
            report_plan = self.memory_manager.load_report_plan(config.DEFAULT_PLAN_FILE)
            if not report_plan or not report_plan.structure:
                 log.error("Report plan file not found or empty when trying to get pending sections.")
                 return "Error: Report plan not loaded or empty."

            pending_ids_list = [] # Liste pour stocker les IDs trouvés
            pending_statuses = {"pending", "failed"}
            processed_ids_for_search = set() # Pour éviter les boucles

            # Fonction interne récursive pour trouver les IDs en attente/échec
            def find_pending_ids_recursive(sections: List[ReportSection]):
                 if not sections: return # Arrêt si pas de sous-sections
                 for section in sections:
                     sec_id = getattr(section, 'section_id', None)
                     # Ignorer si pas d'ID ou déjà traité
                     if not sec_id or sec_id in processed_ids_for_search: continue

                     current_status = getattr(section, 'status', 'pending') # Défaut à pending
                     if current_status in pending_statuses:
                         pending_ids_list.append(sec_id) # Ajouter à la liste

                     processed_ids_for_search.add(sec_id) # Marquer comme traité

                     # Chercher dans les sous-sections seulement si on n'a pas déjà trouvé un ID pending/failed à ce niveau ou au-dessus? Non, on les veut tous puis on prend le premier.
                     if hasattr(section, 'subsections') and section.subsections:
                         find_pending_ids_recursive(section.subsections)
            # --- Fin Fonction Interne ---

            # Lancer la recherche depuis la racine du plan
            find_pending_ids_recursive(report_plan.structure)

            # --- Logique de Retour Corrigée ---
            if not pending_ids_list:
                log.info("No pending or failed sections found in the plan.")
                # Retourner le message spécifique indiquant la complétion
                return "COMPLETED - No pending sections found."
            else:
                # Retourner seulement le PREMIER ID de la liste trouvée
                first_pending_id = pending_ids_list[0]
                log.info(f"First pending section ID found: {first_pending_id} (Total pending/failed: {len(pending_ids_list)})")
                return first_pending_id
            # --- Fin Logique de Retour ---

        except Exception as e:
            log.error(f"Error getting pending sections: {e}", exc_info=True)
            return f"Error getting pending sections: {e}"

class UpdateSectionStatusTool(BaseTool):
    name: str = "update_section_status"; description: str = "Updates the status of a specific section using its unique ID. Input format: 'section_id,new_status'. Valid statuses: pending, drafting, drafted, failed, reviewing, approved.";
    # args_schema non nécessaire car description claire, mais pourrait utiliser SectionStatusInput
    memory_manager: MemoryManager
    def _run(self, tool_input: str) -> str:
        log.info(f"[Tool] {self.name}(input='{tool_input}')")
        if not self.memory_manager: return "Error: Memory Manager unavailable."

        # --- Parsing Input ---
        try:
            parts = tool_input.split(',', 1)
            if len(parts) != 2: raise ValueError("Input must contain exactly one comma.")
            section_id = parts[0].strip()
            new_status = parts[1].strip().lower() # Lowercase for consistent validation
            if not section_id: raise ValueError("Section ID cannot be empty.")
        except Exception as e_parse:
            log.error(f"Invalid input format for {self.name}: '{tool_input}'. Error: {e_parse}")
            return f"Error: Invalid input format. Expected 'section_id,new_status'. Received: '{tool_input}'"

        # --- Validate Status ---
        VALID_STATUSES = {"pending", "drafting", "drafted", "failed", "reviewing", "approved"}
        if new_status not in VALID_STATUSES:
            log.error(f"Invalid status '{new_status}' provided for section '{section_id}'.")
            return f"Error: Invalid status '{new_status}'. Valid statuses are: {', '.join(VALID_STATUSES)}."

        # --- Load Plan and Update ---
        try:
            report_plan = self.memory_manager.load_report_plan(config.DEFAULT_PLAN_FILE)
            if not report_plan:
                log.error("Failed to load report plan for status update.")
                return "Error: Report plan could not be loaded."

            updated = False
            processed_ids_update = set() # Prevent cycles

            # Fonction interne récursive pour trouver et mettre à jour par ID
            def find_and_update_recursive(sections: List[ReportSection]) -> bool:
                nonlocal updated # Allow modification
                if updated or not sections: return updated # Stop if already updated or no sections

                for section in sections:
                    current_sec_id = getattr(section, 'section_id', None)
                    if not current_sec_id or current_sec_id in processed_ids_update: continue
                    processed_ids_update.add(current_sec_id)

                    if current_sec_id == section_id:
                        old_status = getattr(section, 'status', 'unknown')
                        setattr(section, 'status', new_status) # Mise à jour directe de l'attribut
                        log.info(f"Updated status for section ID '{section_id}' from '{old_status}' to '{new_status}'.")
                        updated = True
                        return True # Found and updated

                    # Chercher dans les sous-sections
                    if hasattr(section, 'subsections') and section.subsections:
                        if find_and_update_recursive(section.subsections):
                            return True # Propagate success upwards
                return False # Not found in this branch

            # Lancer la recherche/mise à jour
            find_and_update_recursive(report_plan.structure)

            # --- Save Plan and Return Result ---
            if updated:
                self.memory_manager.save_report_plan(report_plan, config.DEFAULT_PLAN_FILE) # Sauvegarde après modification
                return f"Success: Status for section ID '{section_id}' updated to '{new_status}' in the plan file."
            else:
                log.warning(f"Attempted to update status for section ID '{section_id}', but it was not found in the plan.")
                # Lister les IDs disponibles peut aider au débogage par l'agent
                available_ids = []
                def collect_ids(secs):
                    if not secs: return # Ajout vérification None/empty
                    for s in secs:
                        sid = getattr(s, 'section_id', None);
                        if sid: available_ids.append(sid)
                        if hasattr(s, 'subsections'): collect_ids(s.subsections)
                collect_ids(report_plan.structure)
                log.debug(f"Available section IDs in plan: {available_ids}")
                return f"Error: Section ID '{section_id}' not found in the report plan. Update failed."

        except Exception as e:
            log.error(f"Error updating status for section ID '{section_id}': {e}", exc_info=True)
            return f"Error during status update for '{section_id}': {e}"


class DraftSingleSectionTool(BaseTool):
    name: str = "draft_single_section"; description: str = "Drafts the content for a specific section using its unique ID. Input: section_id string. Returns the drafted text content AND saves it to the plan file." # Description mise à jour
    args_schema: Type[BaseModel] = DraftSectionInput # Utiliser le schéma Pydantic
    vector_db: VectorDBManager; memory_manager: MemoryManager; llm: BaseChatModel

    def _run(self, section_id: str) -> str:
        log.info(f"[Tool] {self.name}(section_id='{section_id}')")

        # --- 1. Check Dependencies ---
        if not self.memory_manager: return "Error: Memory Manager unavailable."
        if not self.vector_db: return "Error: Vector DB unavailable."
        if not self.llm: return "Error: LLM unavailable."
        if not section_id: return "Error: Invalid or empty section_id provided."

        # --- 2. Load Report Plan ---
        try:
            # Charger le plan DANS CETTE MÉTHODE pour s'assurer qu'on a la version la plus récente
            # et pour pouvoir le modifier et le sauvegarder.
            report_plan = self.memory_manager.load_report_plan(config.DEFAULT_PLAN_FILE)
            if not report_plan or not report_plan.structure:
                log.error(f"Failed to load report plan or plan is empty ({config.DEFAULT_PLAN_FILE}).")
                return f"Error: Report plan missing or empty ({config.DEFAULT_PLAN_FILE})."
        except Exception as e:
            log.error(f"Error loading report plan: {e}", exc_info=True)
            return f"Error loading report plan: {e}"

        # --- 3. Find the Section by ID ---
        target_section: Optional[ReportSection] = None
        processed_ids_for_find = set() # To prevent infinite loops

        # Fonction interne récursive pour trouver la section par ID
        def find_section_recursive(sections: List[ReportSection], target_id: str) -> Optional[ReportSection]:
            nonlocal target_section
            if target_section: return target_section
            if not sections: return None
            for section in sections:
                current_sec_id = getattr(section, 'section_id', None)
                if not current_sec_id or current_sec_id in processed_ids_for_find: continue
                processed_ids_for_find.add(current_sec_id)
                if current_sec_id == target_id:
                    log.debug(f"Found section with ID '{target_id}': Title='{getattr(section, 'title', 'N/A')}'")
                    target_section = section
                    return section
                if hasattr(section, 'subsections') and section.subsections:
                    found_in_subs = find_section_recursive(section.subsections, target_id)
                    if found_in_subs: return found_in_subs
            return None

        find_section_recursive(report_plan.structure, section_id)

        if not target_section:
            log.error(f"Section ID '{section_id}' not found within the loaded report plan structure.")
            available_ids = []
            def collect_ids(secs):
                 if not secs: return
                 for s in secs:
                     sid = getattr(s, 'section_id', None)
                     if sid: available_ids.append(sid)
                     if hasattr(s, 'subsections'): collect_ids(s.subsections)
            collect_ids(report_plan.structure)
            log.debug(f"Available section IDs in plan: {available_ids}")
            return f"Error: Section ID '{section_id}' not found in the report plan."

        section_title = getattr(target_section, 'title', None)
        if not section_title:
             log.error(f"Found section ID '{section_id}' but it has no title attribute.")
             return f"Error: Section '{section_id}' found but has no title."

        log.info(f"Found section to draft: ID='{section_id}', Title='{section_title}'")

        # --- 4. Gather Context ---
        log.debug(f"Gathering context for section: '{section_title}'")
        guideline_context = "No relevant guidelines found."
        journal_context = "No relevant journal entries found."
        try:
            guideline_results = self.vector_db.search_references(query=section_title, k=3)
            if guideline_results: guideline_context = "\n---\n".join([r.get("document","") for r in guideline_results])
            journal_results = self.vector_db.search_journals(query=section_title, k=7)
            if journal_results: journal_context = "\n---\n".join([r.get("document","") for r in journal_results])
            combined_context = f"OFFICIAL GUIDELINES CONTEXT:\n{guideline_context}\n\nRELEVANT JOURNAL ENTRIES:\n{journal_context}"
            log.debug(f"Context length for '{section_title}': {len(combined_context)} chars.")
        except Exception as e_ctx:
            log.error(f"Error gathering context for section '{section_title}': {e_ctx}", exc_info=True)
            return f"Error gathering context for '{section_title}': {e_ctx}"

        # --- 5. Prepare Prompt and Call LLM ---
        log.debug(f"Preparing LLM prompt for section '{section_title}'...")
        system_instructions = "You are an AI assistant writing a specific section for a professional MSc apprenticeship report. Base your writing *only* on the provided context (guidelines and journal entries). Synthesize the information clearly and maintain an academic, professional tone. Focus specifically on the requirements of the section title."
        user_prompt = (
            f"Please draft the content for the report section titled: '{section_title}'.\n\n"
            f"CONTEXT:\n"
            f"-------\n"
            f"{combined_context[:15000]}\n" # Limiter la taille du contexte
            f"-------\n\n"
            f"DRAFT for section \"{section_title}\":"
        )

        try:
            log.info(f"Invoking LLM for section '{section_title}'...")
            response_message = self.llm.invoke(user_prompt)
            drafted_content = getattr(response_message, 'content', None)

            if drafted_content and isinstance(drafted_content, str) and drafted_content.strip():
                log.info(f"Successfully drafted content for '{section_title}'. Length: {len(drafted_content)} chars.")
                clean_content = drafted_content.strip()

                # --- !!! MODIFICATION : SAUVEGARDE DU CONTENU DANS LE PLAN !!! ---
                try:
                    # Assigner le contenu généré à l'objet section trouvé dans le plan chargé
                    target_section.content = clean_content
                    # Sauvegarder l'objet report_plan COMPLET (qui contient maintenant la section modifiée)
                    self.memory_manager.save_report_plan(report_plan, config.DEFAULT_PLAN_FILE)
                    log.info(f"Saved drafted content for section ID '{section_id}' to the plan file.")
                except Exception as e_save:
                    # Logguer l'erreur mais continuer (le contenu est généré, juste pas sauvegardé)
                    log.error(f"Failed to save drafted content for section ID '{section_id}' to plan file: {e_save}", exc_info=True)
                    # On pourrait retourner un message d'erreur spécifique ici si on voulait que l'agent réagisse
                    # return f"Error: Content drafted for '{section_id}' but failed to save to plan file: {e_save}"
                # --- FIN MODIFICATION ---

                # --- PAUSE ---
                pause_duration = 15 # secondes
                log.info(f"Pausing for {pause_duration} seconds to respect API rate limits...")
                time.sleep(pause_duration)

                # --- RETURN CONTENT ---
                # Retourner le contenu même si la sauvegarde a échoué (l'agent peut continuer)
                return clean_content
            else:
                log.error(f"LLM returned empty or invalid content for section '{section_title}'. Response object: {response_message}")
                return f"Error: LLM returned empty or invalid response for section '{section_title}'."

        except Exception as e_llm:
            log.error(f"LLM call failed during drafting section '{section_title}': {e_llm}", exc_info=True)
            pause_duration_error = 20 # secondes
            log.info(f"Pausing for {pause_duration_error} seconds after LLM error...")
            time.sleep(pause_duration_error)
            return f"Error during LLM call for section '{section_title}': {e_llm}"


# --- Exporter les CLASSES d'outils et Schémas d'Input ---
__all__ = [
    "SearchJournalEntriesTool", "SearchGuidelinesTool", "GetReportPlanStructureTool",
    "GetPendingSectionsTool", "UpdateSectionStatusTool", "DraftSingleSectionTool",
    "SearchInput", "GuidelineSearchInput", "SectionStatusInput", "DraftSectionInput",
]