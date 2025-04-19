# agent_tools.py (Version Finale Corrigée V5 - Fix GetPendingSectionsTool)

import logging
from typing import List, Dict, Any, Optional, Type
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
    name: str = "search_journal_entries"; description: str = "Searches YEAR 2 journal entries..."; args_schema: Type[BaseModel] = SearchInput
    vector_db: VectorDBManager
    def _run(self, query: str, k: int = 5) -> str:
        # ... (code inchangé) ...
        log.info(f"[Tool] {self.name}(query='{query[:50]}...', k={k})");
        if not self.vector_db: return "Error: Vector DB unavailable.";
        try: r = self.vector_db.search_journals(query=query, k=k); return "\n\n---\n\n".join([d.get("document","") for d in r]) if r else "No relevant journals found."
        except Exception as e: log.error(f"{self.name} Error: {e}",exc_info=True); return f"Error: {e}"

class SearchGuidelinesTool(BaseTool):
    name: str = "search_guidelines"; description: str = "Searches official guidelines..."; args_schema: Type[BaseModel] = GuidelineSearchInput
    vector_db: VectorDBManager
    def _run(self, topic: str, k: int = 3) -> str:
        # ... (code inchangé) ...
        log.info(f"[Tool] {self.name}(topic='{topic[:50]}...', k={k})");
        if not self.vector_db: return "Error: Vector DB unavailable.";
        try: r = self.vector_db.search_references(query=topic, k=k); return "\n\n---\n\n".join([d.get("document","") for d in r]) if r else "No relevant guidelines found."
        except Exception as e: log.error(f"{self.name} Error: {e}",exc_info=True); return f"Error: {e}"

class GetReportPlanStructureTool(BaseTool):
    name: str = "get_report_plan_structure"; description: str = "Retrieves the current report plan structure..."
    memory_manager: MemoryManager
    def _run(self, *args: Any, **kwargs: Any) -> str:
        # ... (code inchangé) ...
        log.info(f"[Tool] {self.name}()")
        if not self.memory_manager: return "Error: Memory Manager unavailable."
        try:
            report_plan = self.memory_manager.load_report_plan(config.DEFAULT_PLAN_FILE)
            if not report_plan or not report_plan.structure: return f"Error: Report plan missing/empty ({config.DEFAULT_PLAN_FILE})."
            structure_str = f"Report Title: {report_plan.title}\nSections:\n"; processed = set()
            def format_sections(sections: List[ReportSection], indent_level=0): ... # Logique interne inchangée
            format_sections(report_plan.structure); return structure_str.strip()
        except Exception as e: log.error(f"{self.name} Error: {e}", exc_info=True); return f"Error: {e}"

class GetPendingSectionsTool(BaseTool):
    name: str = "get_pending_sections";
    description: str = (
        "Retrieves the ID of the *first* section marked as 'pending' or 'failed' from the report plan file. "
        "Use this to identify the immediate next section to work on. Takes no arguments. "
        "Returns only the section ID string if found, or the specific message 'COMPLETED - No pending sections found.' if none are pending/failed."
    )
    memory_manager: MemoryManager

    # !! MÉTHODE _run CORRIGÉE !!
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

                     # Chercher dans les sous-sections
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
    name: str = "update_section_status"; description: str = "Updates the status of a section (by ID)... Input: 'section_id,new_status'.";
    # args_schema non nécessaire
    memory_manager: MemoryManager
    def _run(self, tool_input: str) -> str:
        # ... (Code inchangé et correct) ...
        log.info(f"[Tool] {self.name}(input='{tool_input}')")
        if not self.memory_manager: return "Error: Memory Manager unavailable."
        try: parts=tool_input.split(',',1); section_id=parts[0].strip(); new_status=parts[1].strip()
        except: return f"Error: Invalid input format. Expected 'section_id,new_status'."
        VALID_STATUSES = {"pending","drafting","drafted","failed","reviewing","approved"}
        if new_status not in VALID_STATUSES: return f"Error: Invalid status '{new_status}'."
        if not section_id: return f"Error: Invalid section_id."
        try: report_plan=self.memory_manager.load_report_plan(config.DEFAULT_PLAN_FILE);
        if not report_plan: return "Error: Report plan could not be loaded."
        updated=False; processed=set()
        def find_and_update(sections) -> bool: ... # Logique interne inchangée
        find_and_update(report_plan.structure);
        if updated: self.memory_manager.save_report_plan(report_plan, config.DEFAULT_PLAN_FILE); return f"Success: Status for '{section_id}' updated to '{new_status}'."
        else: log.warning(f"Section ID '{section_id}' not found."); return f"Error: Section ID '{section_id}' not found."
        except Exception as e: log.error(f"Error updating status for '{section_id}': {e}", exc_info=True); return f"Error: {e}"

class DraftSingleSectionTool(BaseTool):
    name: str = "draft_single_section"; description: str = "Drafts content for section ID...";
    # args_schema non nécessaire
    vector_db: VectorDBManager; memory_manager: MemoryManager; llm: BaseChatModel
    def _run(self, section_id: str) -> str:
        # ... (Code inchangé et correct) ...
        log.info(f"[Tool] {self.name}(section_id='{section_id}')")
        # ... (Logique Get Title, Gather Context, Call LLM) ...
        pass # Placeholder pour la logique existante

# --- Exporter les CLASSES d'outils et Schémas d'Input ---
__all__ = [
    "SearchJournalEntriesTool", "SearchGuidelinesTool", "GetReportPlanStructureTool",
    "GetPendingSectionsTool", "UpdateSectionStatusTool", "DraftSingleSectionTool",
    "SearchInput", "GuidelineSearchInput", "SectionStatusInput", "DraftSectionInput",
]