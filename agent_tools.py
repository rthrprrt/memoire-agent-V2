# agent_tools.py (Version sans pause API)

import logging
from typing import List, Dict, Any, Optional, Type
# --- MODIFICATION : time n'est plus nécessaire ici ---
# import time

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
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')

# --- Schémas d'Arguments Pydantic ---
class SearchInput(BaseModel): query: str = Field(...); k: int = Field(default=5)
class GuidelineSearchInput(BaseModel): topic: str = Field(...); k: int = Field(default=3)
class SectionStatusInput(BaseModel): section_id: str = Field(...); new_status: str = Field(...)
class DraftSectionInput(BaseModel): section_id: str = Field(...)


# --- Définition des Outils au Format LangChain ---

# (Les classes SearchJournalEntriesTool, SearchGuidelinesTool, GetReportPlanStructureTool, GetPendingSectionsTool, UpdateSectionStatusTool restent inchangées)
class SearchJournalEntriesTool(BaseTool):
    name: str = "search_journal_entries"; description: str = "Recherche dans les entrées de journal (année 2) basé sur une requête. Entrée: chaîne de requête. Retourne des extraits de texte pertinents."
    args_schema: Type[BaseModel] = SearchInput
    vector_db: VectorDBManager
    def _run(self, query: str, k: int = 5) -> str:
        log.info(f"[Outil] {self.name}(query='{query[:50]}...', k={k})");
        if not self.vector_db: return "Erreur: Base vectorielle indisponible.";
        try:
            r = self.vector_db.search_journals(query=query, k=k);
            return "\n\n---\n\n".join([d.get("document","") for d in r]) if r else "Aucun journal pertinent trouvé."
        except Exception as e:
            log.error(f"{self.name} Erreur: {e}",exc_info=True);
            return f"Erreur durant la recherche dans les journaux: {e}"

class SearchGuidelinesTool(BaseTool):
    name: str = "search_guidelines"; description: str = "Recherche dans les guidelines officielles (consignes Epitech) basé sur un sujet. Entrée: chaîne de sujet. Retourne des extraits pertinents des guidelines."
    args_schema: Type[BaseModel] = GuidelineSearchInput
    vector_db: VectorDBManager
    def _run(self, topic: str, k: int = 3) -> str:
        log.info(f"[Outil] {self.name}(topic='{topic[:50]}...', k={k})");
        if not self.vector_db: return "Erreur: Base vectorielle indisponible.";
        try:
            r = self.vector_db.search_references(query=topic, k=k);
            return "\n\n---\n\n".join([d.get("document","") for d in r]) if r else "Aucune guideline pertinente trouvée."
        except Exception as e:
            log.error(f"{self.name} Erreur: {e}",exc_info=True);
            return f"Erreur durant la recherche dans les guidelines: {e}"

class GetReportPlanStructureTool(BaseTool):
    name: str = "get_report_plan_structure"; description: str = "Récupère la structure actuelle du plan du rapport incluant titres de sections, IDs, et statuts. Ne prend aucun argument."
    memory_manager: MemoryManager
    def _run(self, *args: Any, **kwargs: Any) -> str:
        log.info(f"[Outil] {self.name}()")
        if not self.memory_manager: return "Erreur: Memory Manager indisponible."
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
                    sec_id = getattr(section, 'section_id', 'NO_ID')
                    if sec_id in processed: continue
                    processed.add(sec_id)

                    title = getattr(section, 'title', 'Sans titre')
                    status = getattr(section, 'status', 'inconnu')
                    level = getattr(section, 'level', indent_level + 1)
                    structure_str += f"{indent}- N{level}: {title} (ID: {sec_id}, Statut: {status})\n"

                    if hasattr(section, 'subsections') and section.subsections:
                        format_sections(section.subsections, indent_level + 1)

            format_sections(report_plan.structure)
            return structure_str.strip()
        except Exception as e:
            log.error(f"{self.name} Erreur: {e}", exc_info=True)
            return f"Erreur lors de la récupération de la structure du plan: {e}"

class GetPendingSectionsTool(BaseTool):
    name: str = "get_pending_sections";
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
    name: str = "update_section_status"; description: str = "Met à jour le statut d'une section spécifique en utilisant son ID unique. Format d'entrée: 'section_id,nouveau_statut'. Statuts valides: pending, drafting, drafted, failed, reviewing, approved.";
    memory_manager: MemoryManager
    def _run(self, tool_input: str) -> str:
        log.info(f"[Outil] {self.name}(input='{tool_input}')")
        if not self.memory_manager: return "Erreur: Memory Manager indisponible."

        try:
            parts = tool_input.split(',', 1)
            if len(parts) != 2: raise ValueError("L'entrée doit contenir exactement une virgule.")
            section_id = parts[0].strip()
            new_status = parts[1].strip().lower()
            if not section_id: raise ValueError("L'ID de section ne peut pas être vide.")
        except Exception as e_parse:
            log.error(f"Format d'entrée invalide pour {self.name}: '{tool_input}'. Erreur: {e_parse}")
            return f"Erreur: Format d'entrée invalide. Attendu 'section_id,nouveau_statut'. Reçu: '{tool_input}'"

        VALID_STATUSES = {"pending", "drafting", "drafted", "failed", "reviewing", "approved"}
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
                    current_sec_id = getattr(section, 'section_id', None)
                    if not current_sec_id or current_sec_id in processed_ids_update: continue
                    processed_ids_update.add(current_sec_id)
                    if current_sec_id == section_id:
                        old_status = getattr(section, 'status', 'inconnu')
                        setattr(section, 'status', new_status)
                        log.info(f"Statut mis à jour pour l'ID de section '{section_id}' de '{old_status}' à '{new_status}'.")
                        updated = True
                        return True
                    if hasattr(section, 'subsections') and section.subsections:
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
                        sid = getattr(s, 'section_id', None);
                        if sid: available_ids.append(sid)
                        if hasattr(s, 'subsections'): collect_ids(s.subsections)
                collect_ids(report_plan.structure)
                log.debug(f"IDs de section disponibles dans le plan: {available_ids}")
                return f"Erreur: ID de section '{section_id}' non trouvé dans le plan du rapport. Mise à jour échouée."

        except Exception as e:
            log.error(f"Erreur lors de la mise à jour du statut pour l'ID '{section_id}': {e}", exc_info=True)
            return f"Erreur durant la mise à jour du statut pour '{section_id}': {e}"


class DraftSingleSectionTool(BaseTool):
    name: str = "draft_single_section"; description: str = "Rédige le contenu pour une section spécifique en utilisant son ID unique. Entrée: ID de section (chaîne). Retourne le contenu textuel rédigé ET le sauvegarde dans le fichier du plan."
    args_schema: Type[BaseModel] = DraftSectionInput
    vector_db: VectorDBManager; memory_manager: MemoryManager; llm: BaseChatModel

    def _run(self, section_id: str) -> str:
        log.info(f"[Outil] {self.name}(section_id='{section_id}')")

        if not self.memory_manager: return "Erreur: Memory Manager indisponible."
        if not self.vector_db: return "Erreur: Base vectorielle indisponible."
        if not self.llm: return "Erreur: LLM indisponible."
        if not section_id: return "Erreur: section_id invalide ou vide fourni."

        try:
            report_plan = self.memory_manager.load_report_plan(config.DEFAULT_PLAN_FILE)
            if not report_plan or not report_plan.structure:
                log.error(f"Échec du chargement du plan ou plan vide ({config.DEFAULT_PLAN_FILE}).")
                return f"Erreur: Plan du rapport manquant ou vide ({config.DEFAULT_PLAN_FILE})."
        except Exception as e:
            log.error(f"Erreur lors du chargement du plan: {e}", exc_info=True)
            return f"Erreur lors du chargement du plan: {e}"

        target_section: Optional[ReportSection] = None
        processed_ids_for_find = set()

        def find_section_recursive(sections: List[ReportSection], target_id: str) -> Optional[ReportSection]:
            nonlocal target_section
            if target_section: return target_section
            if not sections: return None
            for section in sections:
                current_sec_id = getattr(section, 'section_id', None)
                if not current_sec_id or current_sec_id in processed_ids_for_find: continue
                processed_ids_for_find.add(current_sec_id)
                if current_sec_id == target_id:
                    log.debug(f"Section trouvée avec ID '{target_id}': Titre='{getattr(section, 'title', 'N/A')}'")
                    target_section = section
                    return section
                if hasattr(section, 'subsections') and section.subsections:
                    found_in_subs = find_section_recursive(section.subsections, target_id)
                    if found_in_subs: return found_in_subs
            return None

        find_section_recursive(report_plan.structure, section_id)

        if not target_section:
            log.error(f"ID de section '{section_id}' non trouvé dans la structure du plan chargé.")
            available_ids = []
            def collect_ids(secs):
                 if not secs: return
                 for s in secs:
                     sid = getattr(s, 'section_id', None)
                     if sid: available_ids.append(sid)
                     if hasattr(s, 'subsections'): collect_ids(s.subsections)
            collect_ids(report_plan.structure)
            log.debug(f"IDs de section disponibles dans le plan: {available_ids}")
            return f"Erreur: ID de section '{section_id}' non trouvé dans le plan du rapport."

        section_title = getattr(target_section, 'title', None)
        if not section_title:
             log.error(f"Section ID '{section_id}' trouvée mais sans attribut titre.")
             return f"Erreur: Section '{section_id}' trouvée mais sans titre."

        log.info(f"Section trouvée pour rédaction: ID='{section_id}', Titre='{section_title}'")

        log.debug(f"Récupération du contexte pour la section: '{section_title}'")
        guideline_context = "Aucune guideline pertinente trouvée."
        journal_context = "Aucune entrée de journal pertinente trouvée."
        try:
            guideline_results = self.vector_db.search_references(query=section_title, k=3)
            if guideline_results: guideline_context = "\n---\n".join([r.get("document","") for r in guideline_results])
            journal_results = self.vector_db.search_journals(query=section_title, k=7)
            if journal_results: journal_context = "\n---\n".join([r.get("document","") for r in journal_results])
            combined_context = f"CONTEXTE GUIDELINES OFFICIELLES:\n{guideline_context}\n\nCONTEXTE ENTRÉES DE JOURNAL PERTINENTES:\n{journal_context}"
            log.debug(f"Taille du contexte pour '{section_title}': {len(combined_context)} caractères.")
        except Exception as e_ctx:
            log.error(f"Erreur lors de la récupération du contexte pour la section '{section_title}': {e_ctx}", exc_info=True)
            return f"Erreur lors de la récupération du contexte pour '{section_title}': {e_ctx}"

        log.debug(f"Préparation du prompt LLM pour la section '{section_title}'...")
        system_instructions = "Vous êtes un assistant IA rédigeant une section spécifique d'un rapport d'alternance MSc professionnel (Mission Professionnelle Digi5). Basez votre rédaction *uniquement* sur le contexte fourni (guidelines et entrées de journal). Synthétisez l'information clairement et maintenez un ton académique et professionnel en **français**. Concentrez-vous spécifiquement sur les exigences du titre de la section."
        user_prompt = (
            f"Veuillez rédiger le contenu pour la section du rapport intitulée : '{section_title}'.\n\n"
            f"CONTEXTE FOURNI:\n"
            f"-------\n"
            f"{combined_context[:15000]}\n"
            f"-------\n\n"
            f"BROUILLON pour la section \"{section_title}\" (en français):"
        )

        try:
            log.info(f"Invocation du LLM pour la section '{section_title}'...")
            response_message = self.llm.invoke(user_prompt)
            drafted_content = getattr(response_message, 'content', None)

            if drafted_content and isinstance(drafted_content, str) and drafted_content.strip():
                log.info(f"Contenu rédigé avec succès pour '{section_title}'. Longueur: {len(drafted_content)} caractères.")
                clean_content = drafted_content.strip()

                try:
                    target_section.content = clean_content
                    self.memory_manager.save_report_plan(report_plan, config.DEFAULT_PLAN_FILE)
                    log.info(f"Contenu rédigé sauvegardé pour l'ID de section '{section_id}' dans le fichier du plan.")
                except Exception as e_save:
                    log.error(f"Échec de la sauvegarde du contenu rédigé pour l'ID '{section_id}' dans le fichier du plan: {e_save}", exc_info=True)

                # --- MODIFICATION : Pause retirée ---
                # pause_duration = 15
                # log.info(f"Pause de {pause_duration} secondes pour respecter les limites de taux de l'API...")
                # time.sleep(pause_duration)
                # --- FIN MODIFICATION ---

                return clean_content
            else:
                log.error(f"Le LLM a retourné un contenu vide ou invalide pour la section '{section_title}'. Objet réponse: {response_message}")
                return f"Erreur: Le LLM a retourné une réponse vide ou invalide pour la section '{section_title}'."

        except Exception as e_llm:
            log.error(f"L'appel LLM a échoué durant la rédaction de la section '{section_title}': {e_llm}", exc_info=True)
            # --- MODIFICATION : Pause retirée (ou réduite fortement) ---
            # pause_duration_error = 20
            # log.info(f"Pause de {pause_duration_error} secondes après l'erreur LLM...")
            # time.sleep(pause_duration_error)
            # --- FIN MODIFICATION ---
            return f"Erreur durant l'appel LLM pour la section '{section_title}': {e_llm}"


# --- Exporter les CLASSES d'outils et Schémas d'Input ---
__all__ = [
    "SearchJournalEntriesTool", "SearchGuidelinesTool", "GetReportPlanStructureTool",
    "GetPendingSectionsTool", "UpdateSectionStatusTool", "DraftSingleSectionTool",
    "SearchInput", "GuidelineSearchInput", "SectionStatusInput", "DraftSectionInput",
]