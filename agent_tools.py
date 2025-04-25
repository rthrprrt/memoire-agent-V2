# agent_tools.py (Version simplifiée - Logique dans main.py/LangGraph)

import logging
from typing import List, Dict, Any, Optional, Type, ClassVar

# Imports nécessaires
from vector_database import VectorDBManager
from memory_manager import MemoryManager
from data_models import ReportPlan, ReportSection
import config
import json
from llm_interface import get_configured_llm
# Imports LangChain
from langchain_core.tools import BaseTool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel

# Configuration du logger
log = logging.getLogger(__name__)
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - [%(module)s] - %(message)s"
    )

# --- Schémas d'Arguments Pydantic (conservés si outils utilisés ailleurs) ---
class SearchInput(BaseModel):
    query: str = Field(...)
    k: int = Field(default=5)

class GuidelineSearchInput(BaseModel):
    topic: str = Field(...)
    k: int = Field(default=3)

class SectionStatusInput(BaseModel):
    section_id: str = Field(...)
    new_status: str = Field(...)

# --- Définition des Outils UTILISÉS par LangGraph ---

class SearchJournalEntriesTool(BaseTool):
    name: str = "search_journal_entries"
    description: str = "Recherche dans les entrées de journal (année 2) basé sur une requête. Entrée: chaîne de requête. Retourne des extraits de texte pertinents."
    args_schema: Type[BaseModel] = SearchInput
    vector_db: VectorDBManager

    def _run(self, query: str, k: int = 5) -> str:
        log.info(f"[Outil] {self.name}(query='{query[:50]}...', k={k})")
        if not self.vector_db:
            return "Erreur: Base vectorielle indisponible."
        try:
            r = self.vector_db.search_journals(query=query, k=k)
            return (
                "\n\n---\n\n".join([d.get("document", "") for d in r])
                if r
                else "Aucun journal pertinent trouvé."
            )
        except Exception as e:
            log.error(f"{self.name} Erreur: {e}", exc_info=True)
            return f"Erreur durant la recherche dans les journaux: {e}"


class SearchGuidelinesTool(BaseTool):
    name: str = "search_guidelines"
    description: str = "Recherche dans les guidelines officielles (consignes Epitech) basé sur un sujet. Entrée: chaîne de sujet. Retourne des extraits pertinents des guidelines."
    args_schema: Type[BaseModel] = GuidelineSearchInput
    vector_db: VectorDBManager

    def _run(self, topic: str, k: int = 3) -> str:
        log.info(f"[Outil] {self.name}(topic='{topic[:50]}...', k={k})")
        if not self.vector_db:
            return "Erreur: Base vectorielle indisponible."
        try:
            guideline_query = f"Exigences spécifiques pour la section '{topic}' du mémoire Mission Professionnelle Digi5"
            log.debug(f"Recherche guidelines avec query: {guideline_query}")
            r = self.vector_db.search_references(query=guideline_query, k=k)
            if not r:
                 log.debug(f"Aucun résultat pour query spécifiques, essai avec topic brut: {topic}")
                 r = self.vector_db.search_references(query=topic, k=k)

            return (
                "\n\n---\n\n".join([d.get("document", "") for d in r])
                if r
                else "Aucune guideline pertinente trouvée."
            )
        except Exception as e:
            log.error(f"{self.name} Erreur: {e}", exc_info=True)
            return f"Erreur durant la recherche dans les guidelines: {e}"


class GetReportPlanStructureTool(BaseTool):
    name: str = "get_report_plan_structure"
    description: str = "Récupère la structure actuelle du plan du rapport incluant titres de sections, IDs, et statuts. Ne prend aucun argument."
    memory_manager: MemoryManager

    def _run(self, *args: Any, **kwargs: Any) -> str:
        log.info(f"[Outil] {self.name}()")
        if not self.memory_manager:
            return "Erreur: Memory Manager indisponible."
        try:
            report_plan = self.memory_manager.load_report_plan(config.DEFAULT_PLAN_FILE)
            if not report_plan or not report_plan.structure:
                return f"Erreur: Plan du rapport manquant ou vide ({config.DEFAULT_PLAN_FILE})."
            structure_str = f"Titre du Rapport: {report_plan.title}\nSections:\n"
            processed = set()
            def format_sections(sections: List[ReportSection], indent_level=0):
                nonlocal structure_str
                indent = "  " * indent_level
                if not sections: return
                for section in sections:
                    sec_id = getattr(section, "section_id", "NO_ID")
                    if sec_id in processed: continue
                    processed.add(sec_id)
                    title = getattr(section, "title", "Sans titre")
                    status = getattr(section, "status", "inconnu")
                    level = getattr(section, "level", indent_level + 1)
                    structure_str += (
                        f"{indent}- N{level}: {title} (ID: {sec_id}, Statut: {status})\n"
                    )
                    if hasattr(section, "subsections") and section.subsections:
                        format_sections(section.subsections, indent_level + 1)
            format_sections(report_plan.structure)
            return structure_str.strip()
        except Exception as e:
            log.error(f"{self.name} Erreur: {e}", exc_info=True)
            return f"Erreur lors de la récupération de la structure du plan: {e}"


class GetPendingSectionsTool(BaseTool):
    name: str = "get_pending_sections"
    description: str = (
        "Récupère l'ID de la *première* section marquée comme 'pending' (en attente) ou 'failed' (échouée) depuis le fichier du plan. "
        "Utiliser ceci pour identifier la prochaine section immédiate sur laquelle travailler. Ne prend aucun argument. "
        "Retourne seulement l'ID de la section (chaîne de caractères) si trouvée, ou le message spécifique 'COMPLETED - No pending sections found.' si aucune n'est en attente/échouée."
    )
    memory_manager: MemoryManager

    def _run(self, *args: Any, **kwargs: Any) -> str:
        log.info(f"[Outil] {self.name}()")
        if not self.memory_manager: return "Erreur: Memory Manager indisponible."
        try:
            report_plan = self.memory_manager.load_report_plan(config.DEFAULT_PLAN_FILE)
            if not report_plan or not report_plan.structure:
                 log.error("Fichier du plan non trouvé ou vide lors de la recherche de sections en attente.")
                 return "Erreur: Plan du rapport non chargé ou vide."
            pending_ids_list = []
            pending_statuses = {"pending", "failed"}
            processed_ids_for_search = set()
            def find_pending_ids_recursive(sections: List[ReportSection]):
                 if not sections: return
                 for section in sections:
                     sec_id = getattr(section, 'section_id', None)
                     if not sec_id or sec_id in processed_ids_for_search: continue
                     current_status = getattr(section, 'status', 'pending')
                     if current_status in pending_statuses:
                         pending_ids_list.append(sec_id)
                     processed_ids_for_search.add(sec_id)
                     if hasattr(section, 'subsections') and section.subsections:
                         find_pending_ids_recursive(section.subsections)
            find_pending_ids_recursive(report_plan.structure)
            if not pending_ids_list:
                log.info("Aucune section en attente ou échouée trouvée dans le plan.")
                return "COMPLETED - No pending sections found."
            else:
                first_pending_id = pending_ids_list[0]
                log.info(f"Premier ID de section en attente trouvé: {first_pending_id} (Total en attente/échouées: {len(pending_ids_list)})")
                return first_pending_id
        except Exception as e:
            log.error(f"Erreur lors de la récupération des sections en attente: {e}", exc_info=True)
            return f"Erreur lors de la récupération des sections en attente: {e}"


class UpdateSectionStatusTool(BaseTool):
    name: str = "update_section_status"
    description: str = "Met à jour le statut d'une section spécifique en utilisant son ID unique. Format d'entrée: 'section_id,nouveau_statut'. Statuts valides: pending, drafting, drafted, failed, reviewing, approved."
    memory_manager: MemoryManager

    def _run(self, tool_input: str) -> str:
        log.info(f"[Outil] {self.name}(input='{tool_input}')")
        if not self.memory_manager: return "Erreur: Memory Manager indisponible."
        try:
            parts = tool_input.split(",", 1)
            if len(parts) != 2: raise ValueError("L'entrée doit contenir exactement une virgule.")
            section_id = parts[0].strip()
            new_status = parts[1].strip().lower()
            if not section_id: raise ValueError("L'ID de section ne peut pas être vide.")
        except Exception as e_parse:
            log.error(f"Format d'entrée invalide pour {self.name}: '{tool_input}'. Erreur: {e_parse}")
            return f"Erreur: Format d'entrée invalide. Attendu 'section_id,nouveau_statut'. Reçu: '{tool_input}'"
        VALID_STATUSES = {"pending","drafting","drafted","failed","reviewing","approved"}
        if new_status not in VALID_STATUSES:
            log.error(f"Statut invalide '{new_status}' fourni pour la section '{section_id}'.")
            return f"Erreur: Statut invalide '{new_status}'. Les statuts valides sont: {', '.join(VALID_STATUSES)}."
        try:
            report_plan = self.memory_manager.load_report_plan(config.DEFAULT_PLAN_FILE)
            if not report_plan:
                log.error("Échec du chargement du plan pour la mise à jour du statut.")
                return "Erreur: Le plan du rapport n'a pas pu être chargé."
            updated = False
            processed_ids_update = set()
            def find_and_update_recursive(sections: List[ReportSection]) -> bool:
                nonlocal updated
                if updated or not sections: return updated
                for section in sections:
                    current_sec_id = getattr(section, "section_id", None)
                    if not current_sec_id or current_sec_id in processed_ids_update: continue
                    processed_ids_update.add(current_sec_id)
                    if current_sec_id == section_id:
                        old_status = getattr(section, "status", "inconnu")
                        setattr(section, "status", new_status)
                        log.info(f"Statut mis à jour pour l'ID de section '{section_id}' de '{old_status}' à '{new_status}'.")
                        updated = True
                        return True
                    if hasattr(section, "subsections") and section.subsections:
                        if find_and_update_recursive(section.subsections):
                            return True
                return False
            find_and_update_recursive(report_plan.structure)
            if updated:
                self.memory_manager.save_report_plan(report_plan, config.DEFAULT_PLAN_FILE)
                return f"Succès: Statut pour l'ID de section '{section_id}' mis à jour à '{new_status}' dans le fichier du plan."
            else:
                log.warning(f"Tentative de mise à jour du statut pour l'ID '{section_id}', mais il n'a pas été trouvé dans le plan.")
                available_ids = []
                def collect_ids(secs):
                    if not secs: return
                    for s in secs:
                        sid = getattr(s, "section_id", None);
                        if sid: available_ids.append(sid)
                        if hasattr(s, "subsections"): collect_ids(s.subsections)
                collect_ids(report_plan.structure)
                log.debug(f"IDs de section disponibles dans le plan: {available_ids}")
                return f"Erreur: ID de section '{section_id}' non trouvé dans le plan du rapport. Mise à jour échouée."
        except Exception as e:
            log.error(f"Erreur lors de la mise à jour du statut pour l'ID '{section_id}': {e}", exc_info=True)
            return f"Erreur durant la mise à jour du statut pour '{section_id}': {e}"

# --- Exporter les CLASSES d'outils et Schémas d'Input ---
__all__ = [
    "SearchJournalEntriesTool",
    "SearchGuidelinesTool",
    "GetReportPlanStructureTool",
    "GetPendingSectionsTool",
    "UpdateSectionStatusTool",
    # DraftSingleSectionTool n'est plus utilisé directement par le graphe
    "SearchInput",
    "GuidelineSearchInput",
    "SectionStatusInput",
]