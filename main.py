import argparse
import logging
import os
import sys
import json
import re
from typing import List, Optional, TypedDict, Dict

from langgraph.graph import StateGraph, END
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers.string import StrOutputParser
from langchain.schema import OutputParserException
from langchain_core.prompts import ChatPromptTemplate

# --- Import des modules du projet ---
import config
from document_processor import process_all_journals
from vector_database import VectorDBManager
from llm_interface import get_configured_llm
from data_models import ReportPlan
from agent_tools import SectionStatusInput as Section
from agent_tools import GetPendingSectionsTool, UpdateSectionStatusTool
from report_planner import ReportPlanner
from reference_manager import ReferenceManager # Non utilisé activement pour l'instant
from memory_manager import MemoryManager

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s' # Nom du logger inclus
)
log = logging.getLogger(__name__)

# --- Schéma Pydantic pour l'Analyse Holistique (inchangé) ---
class ProjetCle(BaseModel):
    nom: str = Field(description="Nom concis du projet ou de la mission principale.")
    description_courte: str = Field(description="Très brève description (1 phrase) du projet.")
    periode_activite: str = Field(description="Période approximative (ex: 'Mars 2024 - Mai 2024', 'Q2 2025').")
    themes_associes: List[str] = Field(description="Liste des thèmes ou mots-clés principaux associés (ex: 'Automatisation', 'IA Générative', 'Gestion de projet').")

class Competence(BaseModel):
    nom_competence: str = Field(description="Nom de la compétence technique, organisationnelle ou interpersonnelle acquise ou renforcée.")
    contexte_developpement: str = Field(description="Contexte spécifique (projet, tâche) où la compétence a été développée.")
    type: str = Field(description="Type de compétence (ex: 'Technique', 'Organisationnelle', 'Relationnelle').")

class ApprentissageCle(BaseModel):
    nom_apprentissage: str = Field(description="Titre concis de l'apprentissage clé ou de la prise de conscience.")
    description: str = Field(description="Brève description de l'apprentissage.")
    contexte: str = Field(description="Contexte (projet, situation) ayant mené à cet apprentissage.")

class HolisticAnalysis(BaseModel):
    projets_cles: List[ProjetCle] = Field(description="Liste des projets ou missions les plus significatifs mentionnés dans les journaux.")
    competences_acquises_renforcees: List[Competence] = Field(description="Liste des compétences clés acquises ou significativement renforcées.")
    defis_majeurs: List[str] = Field(description="Liste des défis ou difficultés majeures rencontrées.")
    apprentissages_cles: List[ApprentissageCle] = Field(description="Liste des apprentissages les plus importants tirés des expériences.")
    fil_conducteur_narratif: str = Field(description="Proposition d'un fil conducteur ou d'une trame narrative pour le rapport, basée sur l'ensemble des journaux.")

# --- État du workflow LangGraph (Mis à jour) ---
class ReportWorkflowState(TypedDict):
    objective: str
    current_section_id: Optional[str]
    current_section_title: Optional[str]
    structured_guidelines: List[str]
    journal_context: Optional[str]
    is_context_relevant: Optional[bool] # Indicateur de pertinence du contexte RAG
    analysis_result: Optional[str] # Plan structuré pour la section
    written_content: Optional[str] # Prose rédigée pour la section
    is_prose_adequate: Optional[bool] # Indicateur de qualité de la prose
    prose_feedback: Optional[str]   # Feedback si la prose n'est pas adéquate
    final_content: Optional[str]   # Contenu final (après garde-fous)
    pending_check_result: str
    iterations: int
    writing_attempts: int # Compteur de tentatives pour la rédaction d'une section
    error_message: Optional[str]

# --- Profil Utilisateur (inchangé) ---
USER_PROFILE = """
🎓 Profil – Étudiant en Master IA & Transformation d’Entreprise | AI Project Officer chez Gecina
... (Profil complet) ...
Communication, vulgarisation et stratégie IA en entreprise
"""

# --- Analyse holistique des journaux (inchangée depuis la version précédente) ---
def run_holistic_analysis(llm: BaseChatModel, journal_dir: str, output_file: str):
    log.info("--- Démarrage de l'Analyse Holistique des Journaux ---")
    raw_output_file = output_file.replace(".json", "_raw.txt") # Pour sauvegarde brute si erreur

    try:
        entries = process_all_journals(journal_dir)
        if not entries:
            log.error("Aucun journal trouvé.")
            print("ERREUR: Aucun journal trouvé.")
            return
        text = "\n\n".join([
            f"Entrée du {e.date.strftime('%Y-%m-%d')}:\n{e.raw_text}" for e in entries
        ])
        max_chars = config.HOLISTIC_ANALYSIS_MAX_CHARS
        if len(text) > max_chars:
            log.warning(f"Troncation du texte des journaux à {max_chars} caractères pour l'analyse holistique.")
            text = text[:max_chars]
    except Exception as e:
        log.error(f"Erreur lors du chargement ou traitement des journaux: {e}", exc_info=True)
        print(f"ERREUR: Impossible de charger les journaux: {e}")
        return

    parser = PydanticOutputParser(pydantic_object=HolisticAnalysis)
    format_instructions = parser.get_format_instructions()

    prompt = f"""System: Tu es un extracteur de données JSON expert. Analyse le corpus de journaux fourni et extrais les informations clés demandées. Réponds *uniquement* avec l'objet JSON valide demandé, sans aucun texte additionnel, commentaire, ou explication avant ou après le JSON.

User:
**ROLE IA:** Analyste stratégique expert dans la rédaction de mémoires Epitech basés sur des journaux de bord d'alternance.
**CONTEXTE AUTEUR:**
{USER_PROFILE}

**CORPUS DE JOURNAUX DE BORD:**
--- DEBUT JOURNAUX ---
{text}
--- FIN JOURNAUX ---

**TACHE:** Analyse l'intégralité du corpus de journaux ci-dessus. Identifie les projets clés, les compétences développées, les défis majeurs, les apprentissages principaux et propose un fil conducteur pour un rapport d'alternance. Structure ta réponse **UNIQUEMENT** en JSON valide, en respectant strictement le format décrit ci-dessous. Ne fournis rien d'autre que l'objet JSON.

**FORMAT JSON ATTENDU (Instructions Pydantic):**
{format_instructions}

**RÈGLES STRICTES:**
1.  **Analyse Globale:** Base-toi sur l'ensemble des entrées fournies.
2.  **Synthèse:** Sois concis et pertinent pour un rapport d'alternance.
3.  **Neutralité:** Rapporte les faits objectivement tels que décrits dans les journaux.
4.  **Anonymisation:** Utilise des rôles génériques (ex: "le Manager", "l'équipe technique") si des noms spécifiques apparaissent, conformément au mapping de `config.py` si disponible (le mapping n'est pas fourni ici, fais au mieux).
5.  **Format JSON Strict:** La sortie doit être *uniquement* l'objet JSON, sans aucun autre texte.

**JSON RÉSULTAT DE L'ANALYSE HOLISTIQUE:**
"""

    try:
        log.info("Invocation du LLM pour l'analyse holistique...")
        resp = llm.invoke(prompt)
        content = getattr(resp, 'content', '')

        if not content.strip():
            log.error("La réponse du LLM pour l'analyse holistique est vide.")
            print("ERREUR: La réponse du LLM est vide.")
            return

        log.info("Tentative de parsing de la réponse LLM avec PydanticOutputParser...")
        with open(raw_output_file, 'w', encoding='utf-8') as f_raw:
            f_raw.write(content)
        log.info(f"Réponse brute du LLM sauvegardée dans: {raw_output_file}")

        try:
            parsed_data: HolisticAnalysis = parser.parse(content)
            with open(output_file, 'w', encoding='utf-8') as f_json:
                if hasattr(parsed_data, 'model_dump_json'):
                     f_json.write(parsed_data.model_dump_json(indent=4))
                else:
                     f_json.write(parsed_data.json(indent=4, ensure_ascii=False))
            log.info(f"Analyse holistique parsée et sauvegardée avec succès: {output_file}")
            print(f"\nAnalyse holistique sauvegardée avec succès: {output_file}\nVérifiez son contenu.")

        except OutputParserException as e_parse:
            log.error(f"Échec du parsing JSON par PydanticOutputParser: {e_parse}", exc_info=True)
            log.warning("Tentative d'extraction manuelle du JSON...")
            json_match = re.search(r'```json\s*(\{.*?\})\s*```|(\{.*?\})', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1) or json_match.group(2)
                if json_str:
                    log.info("JSON potentiel extrait manuellement, nouvelle tentative de parsing...")
                    try:
                        parsed_data: HolisticAnalysis = parser.parse(json_str.strip())
                        with open(output_file, 'w', encoding='utf-8') as f_json:
                             if hasattr(parsed_data, 'model_dump_json'):
                                 f_json.write(parsed_data.model_dump_json(indent=4))
                             else:
                                 f_json.write(parsed_data.json(indent=4, ensure_ascii=False))
                        log.info(f"Analyse holistique extraite manuellement, parsée et sauvegardée: {output_file}")
                        print(f"\nAnalyse holistique sauvegardée (après extraction manuelle): {output_file}\nVérifiez son contenu.")
                    except (OutputParserException, json.JSONDecodeError) as e_reparse:
                         log.critical(f"Échec du parsing même après extraction manuelle: {e_reparse}")
                         print(f"ERREUR CRITIQUE: Impossible de parser le JSON de l'analyse holistique. Vérifiez {raw_output_file}")
                else:
                     log.critical("Impossible d'extraire manuellement un bloc JSON valide.")
                     print(f"ERREUR CRITIQUE: Impossible d'extraire le JSON de l'analyse holistique. Vérifiez {raw_output_file}")
            else:
                 log.critical("Aucun bloc JSON identifiable dans la réponse LLM.")
                 print(f"ERREUR CRITIQUE: Aucun JSON trouvé dans l'analyse holistique. Vérifiez {raw_output_file}")

    except Exception as e_llm:
        log.error(f"Erreur générale durant l'analyse holistique (appel LLM ou autre): {e_llm}", exc_info=True)
        print(f"ERREUR: Une erreur s'est produite lors de l'analyse holistique: {e_llm}")
        if 'content' in locals() and content:
             with open(raw_output_file, 'w', encoding='utf-8') as f_raw_err:
                 f_raw_err.write(content)
             log.info(f"Contenu brut en erreur sauvegardé : {raw_output_file}")

# --- Variables globales pour les composants partagés ---
llm_analysis: Optional[BaseChatModel] = None
llm_writing: Optional[BaseChatModel] = None
vector_db_manager: Optional[VectorDBManager] = None
memory_manager: Optional[MemoryManager] = None
holistic_analysis_content: Optional[Dict] = None # Chargé une fois au début de run_agent

# --- Point d'entrée ---
def main():
    global llm_analysis, llm_writing, vector_db_manager, memory_manager, holistic_analysis_content # Déclare utiliser les globales

    parser = argparse.ArgumentParser(description="Agent IA pour la rédaction de mémoire Epitech")
    parser.add_argument("--llm", choices=config.LLM_PROVIDERS, default=None,
                        help="Force l'utilisation d'un fournisseur LLM spécifique (ignore config.py).")
    sub = parser.add_subparsers(dest="command", required=True, help="Commande à exécuter")

    # --- Définition des commandes ---
    parser_analyze = sub.add_parser("analyze_journals", help="Effectue l'analyse holistique des journaux.")
    parser_analyze.add_argument("--journal_dir", default=config.JOURNAL_DIR, help="Répertoire contenant les fichiers journaux DOCX.")
    parser_analyze.add_argument("--output_file", default=config.HOLISTIC_ANALYSIS_FILE, help="Fichier JSON de sortie pour l'analyse.")

    parser_plan = sub.add_parser("create_plan", help="Crée le plan initial du rapport (structure et statuts).")
    parser_plan.add_argument("--output_plan_file", default=config.DEFAULT_PLAN_FILE, help="Fichier JSON de sortie pour le plan.")

    parser_assemble = sub.add_parser("assemble_report", help="Assemble le rapport DOCX final depuis un plan JSON complété.")
    parser_assemble.add_argument("--plan_file", default=config.DEFAULT_PLAN_FILE, help="Chemin vers le fichier plan JSON (avec contenu).")
    parser_assemble.add_argument("--output_file", default=config.DEFAULT_REPORT_OUTPUT, help="Chemin pour sauvegarder le rapport DOCX assemblé.")

    parser_agent = sub.add_parser("run_agent", help="Lance le workflow de rédaction section par section.")
    parser_agent.add_argument("--max_iterations", type=int, default=50, help="Nombre maximum de sections à traiter.")
    parser_agent.add_argument("--max_writing_attempts", type=int, default=3, help="Nombre maximum de tentatives de réécriture pour une section.")
    parser_agent.add_argument("--objective", default="Générer un premier brouillon du rapport d'alternance.", help="Objectif global pour l'agent.")
    parser_agent.add_argument("--plan_file", default=config.DEFAULT_PLAN_FILE, help="Chemin vers le fichier plan JSON à utiliser/mettre à jour.")

    args = parser.parse_args()
    provider = args.llm or config.LLM_PROVIDER
    log.info(f"Utilisation du fournisseur LLM : {provider}")

    # --- Initialisation basée sur la commande ---
    if args.command == "analyze_journals":
        llm_analysis = get_configured_llm(provider=provider, purpose="analysis")
        if not llm_analysis: sys.exit(1)
        run_holistic_analysis(llm_analysis, args.journal_dir, args.output_file)

    elif args.command == "create_plan":
        log.info("--- Commande: create_plan ---")
        # Utilisation de ReportPlanner ici
        planner = ReportPlanner()
        report_plan = planner.create_base_plan()
        # Utilisation de MemoryManager pour sauvegarder
        temp_memory = MemoryManager(plan_file=args.output_plan_file) # Instance temporaire juste pour sauver
        temp_memory.save_report_plan(report_plan, args.output_plan_file)
        log.info(f"Plan initial sauvegardé : {args.output_plan_file}")
        print(f"\nPlan initial sauvegardé : {args.output_plan_file}")

    elif args.command == "assemble_report":
        log.info("--- Commande: assemble_report ---")
        try:
            from report_generator import assemble_report_from_plan
            assemble_report_from_plan(args.plan_file, args.output_file)
            log.info(f"Rapport assemblé avec succès: {args.output_file}")
            print(f"\nRapport assemblé et sauvegardé : {args.output_file}")
        except ImportError:
             log.critical("Importation échouée pour 'report_generator.assemble_report_from_plan'.")
             sys.exit(1)
        except FileNotFoundError:
             log.error(f"Fichier plan '{args.plan_file}' non trouvé.")
             sys.exit(1)
        except Exception as e_ass:
            log.error(f"Erreur assemblage: {e_ass}", exc_info=True)
            sys.exit(1)

    elif args.command == "run_agent":
        log.info(f"--- Commande: run_agent (Workflow Principal avec Réflexion) ---")
        log.info(f"Utilisation du fichier plan : {args.plan_file}")

        # --- Vérifications et Initialisation pour run_agent ---
        if not os.path.exists(config.HOLISTIC_ANALYSIS_FILE):
            print(f"ERREUR: Fichier analyse holistique '{config.HOLISTIC_ANALYSIS_FILE}' manquant. Lancez 'analyze_journals'.")
            sys.exit(1)
        if not os.path.exists(args.plan_file):
             print(f"ERREUR: Fichier plan '{args.plan_file}' manquant. Lancez 'create_plan'.")
             sys.exit(1)

        try:
            with open(config.HOLISTIC_ANALYSIS_FILE, 'r', encoding='utf-8') as f:
                holistic_analysis_content = json.load(f) # Chargement dans la variable globale
            log.info("Analyse holistique chargée.")

            # Initialisation des composants globaux
            vector_db_manager = VectorDBManager()
            memory_manager = MemoryManager(plan_file=args.plan_file)
            llm_analysis = get_configured_llm(provider=provider, purpose="analysis")
            llm_writing = get_configured_llm(provider=provider, purpose="writing")

            if not all([vector_db_manager, memory_manager, llm_analysis, llm_writing]):
                print("ERREUR: Échec initialisation d'un ou plusieurs composants (DB, Mem, LLMs).")
                sys.exit(1)

            # Vérification DB Vecteurs (optionnel: vectoriser si vide)
            if vector_db_manager.collection is None or vector_db_manager.collection.count() == 0:
                 log.warning("Base vectorielle vide. Contexte RAG limité.")
                 # print("AVERTISSEMENT: Base vectorielle vide...")

        except Exception as e_init:
            print(f"ERREUR critique lors de l'initialisation: {e_init}")
            log.critical(f"Erreur initialisation: {e_init}", exc_info=True)
            sys.exit(1)

        # --- Définition des outils et nœuds du graphe ---
        get_pending = GetPendingSectionsTool(memory_manager=memory_manager)
        update_status = UpdateSectionStatusTool(memory_manager=memory_manager)

        def get_title_from_plan(plan: ReportPlan, sid: str) -> Optional[str]:
            # Fonction utilitaire (inchangée)
            if not plan or not plan.structure: return None
            queue: List[Section] = list(plan.structure)
            while queue:
                s = queue.pop(0)
                if getattr(s, 'section_id', None) == sid: return getattr(s, 'title', None)
                queue.extend(getattr(s, 'subsections', []))
            return None

        # --- Nœuds du workflow LangGraph (avec logique réelle) ---

        def check_pending(state: ReportWorkflowState) -> ReportWorkflowState:
            # (Logique inchangée par rapport à la version précédente)
            log.info(f"Iteration {state.get('iterations', 0) + 1}. Vérification sections...")
            result = get_pending.invoke({})
            log.info(f"Résultat check_pending: {result}")
            section_id = None if 'COMPLETED' in result or 'Erreur' in result else result
            error_msg = result if 'Erreur' in result else None
            return {
                **state,
                'pending_check_result': result,
                'current_section_id': section_id,
                'iterations': state.get('iterations', 0) + 1,
                'error_message': error_msg,
                'current_section_title': None, 'structured_guidelines': [], 'journal_context': None,
                'is_context_relevant': None, 'analysis_result': None, 'written_content': None,
                'is_prose_adequate': None, 'prose_feedback': None, 'final_content': None,
                'writing_attempts': 0, # Réinitialise compteur tentatives écriture
            }

        def prepare_drafting(state: ReportWorkflowState) -> ReportWorkflowState:
            # (Logique inchangée par rapport à la version précédente)
            sid = state['current_section_id']
            if not sid:
                log.error("ID section manquant (prepare_drafting).")
                return {**state, 'error_message': 'ID section manquant (prepare).'}
            log.info(f"Préparation pour la section ID: {sid}")
            update_result = update_status.invoke({'tool_input': f"{sid},drafting"})
            log.info(f"Résultat màj statut (drafting) pour {sid}: {update_result}")
            if 'Erreur' in update_result:
                return {**state, 'error_message': f"Erreur MàJ statut (drafting): {update_result}"}
            try:
                current_plan = memory_manager.load_report_plan()
                title = get_title_from_plan(current_plan, sid)
                if not title: return {**state, 'error_message': f"Titre non trouvé ID {sid}."}
                guidelines = config.STRUCTURED_GUIDELINES.get(title, [])
                log.info(f"Section: '{title}' (ID: {sid}). Guidelines: {len(guidelines)}")
                return {**state, 'current_section_title': title, 'structured_guidelines': guidelines}
            except Exception as e_plan:
                 log.error(f"Erreur chargement plan: {e_plan}", exc_info=True)
                 return {**state, 'error_message': f"Erreur lecture plan: {e_plan}"}

        def gather_focused_context(state: ReportWorkflowState) -> ReportWorkflowState:
            # (Logique inchangée par rapport à la version précédente)
            title = state['current_section_title']
            guidelines = state['structured_guidelines']
            if not title: return {**state, 'error_message': 'Titre manquant (gather_context).'}
            query = f"Extraits de journaux pertinents pour la section '{title}'. Chercher exemples, faits, réalisations, réflexions liés à : {'; '.join(guidelines)}. Contexte global : {state.get('objective', '')}"
            log.info(f"Lancement recherche RAG pour '{title}'...")
            try:
                docs = vector_db_manager.search_journals(query=query, k=config.RAG_NUM_RESULTS)
                if docs:
                    context_text = f"Contexte pertinent trouvé pour '{title}':\n\n" + "\n\n---\n\n".join([f"Source (Date approx.): {d.get('metadata', {}).get('date', 'Inconnue')}\nExtrait: {d.get('document', '')}" for d in docs])
                    log.info(f"{len(docs)} extraits RAG trouvés.")
                else:
                    context_text = f"Aucun exemple spécifique trouvé pour '{title}'."
                    log.warning(f"Aucun résultat RAG pour '{title}'.")
                return {**state, 'journal_context': context_text}
            except Exception as e_rag:
                log.error(f"Erreur RAG: {e_rag}", exc_info=True)
                return {**state, 'error_message': f"Erreur RAG: {e_rag}"}

        # --- Nouveau nœud : Filtrage du contexte RAG ---
        def filter_rag_context(state: ReportWorkflowState) -> ReportWorkflowState:
            title = state['current_section_title']
            guidelines = state['structured_guidelines']
            context = state['journal_context']
            log.info(f"Filtrage/Évaluation du contexte RAG pour la section '{title}'...")

            if not context or "Aucun exemple spécifique trouvé" in context:
                 log.warning(f"Pas de contexte RAG à filtrer pour '{title}'. Marqué comme non pertinent.")
                 return {**state, 'is_context_relevant': False} # Ou True si on accepte de continuer sans contexte? False semble plus sûr.

            prompt_template = ChatPromptTemplate.from_messages([
                ("system", "Tu es un assistant IA évaluant la pertinence d'un contexte pour rédiger une section de rapport d'alternance. Réponds uniquement par 'Oui' ou 'Non', suivi d'une brève justification (1 phrase max)."),
                ("human", """Évalue si le CONTEXTE suivant est réellement utile et pertinent pour rédiger la section de rapport '{section_title}' en suivant les GUIDELINES.

GUIDELINES:
- {guidelines}

CONTEXTE À ÉVALUER:
---
{context}
---

Ce contexte contient-il des informations concrètes, exemples, faits ou réflexions qui aident *directement* à traiter les guidelines pour cette section spécifique ?

Réponse (Oui/Non + Justification brève):""")
            ])
            chain = prompt_template | llm_analysis | StrOutputParser()
            try:
                response = chain.invoke({
                    "section_title": title,
                    "guidelines": "; ".join(guidelines) if guidelines else "Objectif général de la section",
                    "context": context
                })
                log.info(f"Résultat évaluation pertinence contexte: {response}")
                is_relevant = response.lower().strip().startswith('oui')
                return {**state, 'is_context_relevant': is_relevant}
            except Exception as e_filter:
                 log.error(f"Erreur lors de l'évaluation du contexte RAG: {e_filter}", exc_info=True)
                 # En cas d'erreur, on considère le contexte comme non pertinent par défaut? Ou on continue?
                 return {**state, 'is_context_relevant': False, 'error_message': f"Erreur filtre RAG: {e_filter}"}


        def structure_section_content(state: ReportWorkflowState) -> ReportWorkflowState:
            title = state['current_section_title']
            guidelines = state['structured_guidelines']
            context = state['journal_context'] if state.get('is_context_relevant') else "Aucun contexte pertinent fourni par le RAG."
            log.info(f"Structuration du contenu pour la section: {title}")

            holistic_summary = json.dumps(holistic_analysis_content, indent=2, ensure_ascii=False) if holistic_analysis_content else "Analyse holistique non disponible."

            prompt_template = ChatPromptTemplate.from_messages([
                ("system", "Tu es un architecte de contenu expert en mémoires Epitech. Ton rôle est de créer un plan structuré et détaillé (en Markdown) pour une section de rapport, basé sur les guidelines, le contexte RAG fourni et l'analyse holistique globale."),
                ("human", """Crée un plan détaillé en Markdown pour la section de rapport intitulée '{section_title}'.

Objectif global de la section (Guidelines):
- {guidelines}

Contexte extrait des journaux de bord (RAG - à utiliser si pertinent):
---
{rag_context}
---

Analyse holistique globale du parcours de l'étudiant (pour le contexte général):
---
{holistic_analysis}
---

**Instructions:**
1.  **Structure Markdown:** Utilise des titres (##, ###), des listes à puces (-) et des sous-points pour organiser le plan.
2.  **Couverture des Guidelines:** Assure-toi que chaque point des guidelines est adressé dans le plan.
3.  **Intégration Contexte:** Suggère où intégrer les exemples concrets du contexte RAG (s'il est pertinent) ou de l'analyse holistique. Mentionne explicitement [Exemple RAG: ...] ou [Ref Holistique: ...] là où c'est approprié.
4.  **Logique et Cohérence:** Le plan doit présenter une structure logique et fluide pour la future rédaction.
5.  **Contenu Minimal:** Même sans contexte RAG pertinent, propose une structure basée sur les guidelines et l'analyse holistique.
6.  **Sortie:** Ne retourne QUE le plan en Markdown.

Plan Détaillé en Markdown:""")
            ])
            chain = prompt_template | llm_analysis | StrOutputParser()

            try:
                structured_plan = chain.invoke({
                    "section_title": title,
                    "guidelines": "; ".join(guidelines) if guidelines else "Objectif général de la section",
                    "rag_context": context,
                    "holistic_analysis": holistic_summary[:2000] # Tronquer si trop long
                })
                log.info(f"Plan structuré généré pour '{title}'.")
                return {**state, 'analysis_result': structured_plan}
            except Exception as e_struct:
                log.error(f"Erreur lors de la structuration du contenu: {e_struct}", exc_info=True)
                return {**state, 'error_message': f"Erreur structuration: {e_struct}"}


        def write_section_prose(state: ReportWorkflowState) -> ReportWorkflowState:
            title = state['current_section_title']
            structured_plan = state['analysis_result']
            feedback = state.get('prose_feedback') # Récupère le feedback de l'itération précédente
            attempt = state.get('writing_attempts', 0) + 1
            log.info(f"Rédaction de la prose pour '{title}' (Tentative {attempt})...")

            if not structured_plan:
                log.error(f"Plan structuré manquant pour la rédaction de '{title}'.")
                return {**state, 'error_message': f"Plan manquant (write {title})."}

            prompt_list = [
                ("system", f"Tu es un rédacteur académique expert, spécialisé dans les rapports d'alternance Epitech (Mission Professionnelle Digi5). Rédige la section demandée en suivant strictement le plan fourni. Adopte un style formel, neutre, objectif (basé sur les faits implicites dans le plan) et respecte l'anonymisation (utilise les rôles génériques si mentionnés). Rédige directement le contenu de la section, sans introduction ou conclusion superflue sur ton rôle."),
                ("human", """Rédige le contenu de la section de rapport intitulée '{section_title}' en te basant *uniquement* sur le plan détaillé ci-dessous.

**Plan à suivre:**
---
{structured_plan}
---
""" + (f"\n**Feedback de la tentative précédente (à prendre en compte pour améliorer):**\n{feedback}\n" if feedback else "") + """
**Instructions de Rédaction:**
1.  **Suivre le Plan:** Adresse chaque point du plan dans l'ordre.
2.  **Style Académique:** Formel, précis, phrases complètes. Utilise le vouvoiement implicite ou un ton neutre.
3.  **Neutralité & Objectivité:** Base-toi sur les éléments du plan (qui dérivent du contexte). Évite les opinions personnelles non justifiées.
4.  **Anonymisation:** Respecte l'anonymisation mentionnée (ex: 'le Manager').
5.  **Cohérence:** Assure une transition fluide entre les points.
6.  **Contenu Complet:** Rédige un texte suffisamment développé pour couvrir le plan.
7.  **Sortie:** Ne retourne QUE le texte rédigé de la section.

Contenu Ré-écrit de la Section (si feedback) / Contenu de la Section:""")
            ]
            prompt_template = ChatPromptTemplate.from_messages(prompt_list)
            chain = prompt_template | llm_writing | StrOutputParser()

            try:
                written_text = chain.invoke({
                    "section_title": title,
                    "structured_plan": structured_plan,
                    "feedback": feedback or "N/A"
                })
                log.info(f"Prose rédigée pour '{title}'.")
                # Réinitialise le feedback après une tentative réussie
                return {**state, 'written_content': written_text, 'writing_attempts': attempt, 'prose_feedback': None}
            except Exception as e_write:
                 log.error(f"Erreur lors de la rédaction de la prose: {e_write}", exc_info=True)
                 return {**state, 'error_message': f"Erreur rédaction: {e_write}", 'writing_attempts': attempt}

        # --- Nouveau nœud : Évaluation de la prose ---
        def evaluate_written_prose(state: ReportWorkflowState) -> ReportWorkflowState:
            title = state['current_section_title']
            plan = state['analysis_result']
            prose = state['written_content']
            log.info(f"Évaluation de la prose rédigée pour la section '{title}'...")

            if not prose:
                log.error(f"Pas de prose à évaluer pour '{title}'.")
                return {**state, 'is_prose_adequate': False, 'prose_feedback': "Le contenu rédigé est vide."}

            prompt_template = ChatPromptTemplate.from_messages([
                ("system", "Tu es un évaluateur qualité pour rapport d'alternance Epitech. Évalue la prose fournie selon les critères spécifiés. Réponds uniquement par 'Oui' si adéquat, ou 'Non' suivi d'un feedback *constructif et concis* (2-3 points max) si des améliorations sont nécessaires."),
                ("human", """Évalue si la PROSE suivante est une rédaction adéquate pour la section '{section_title}', en se basant sur le PLAN fourni.

Critères d'évaluation:
1.  **Adéquation au Plan:** La prose suit-elle fidèlement la structure et les points du plan ?
2.  **Style Académique:** Le style est-il formel, neutre et objectif ?
3.  **Clarté et Cohérence:** Le texte est-il facile à comprendre et logiquement structuré ?
4.  **Anonymisation:** L'anonymisation semble-t-elle respectée (pas de noms propres évidents)?

PLAN DE LA SECTION:
---
{section_plan}
---

PROSE À ÉVALUER:
---
{written_prose}
---

La prose est-elle adéquate selon ces critères ?

Réponse (Oui / Non + Feedback constructif si Non):""")
            ])
            chain = prompt_template | llm_analysis | StrOutputParser()

            try:
                response = chain.invoke({
                    "section_title": title,
                    "section_plan": plan,
                    "written_prose": prose
                })
                log.info(f"Résultat évaluation prose: {response}")
                is_adequate = response.lower().strip().startswith('oui')
                feedback = None if is_adequate else response.partition('\n')[2].strip() or response.partition(' ')[2].strip() # Extrait le feedback après Oui/Non

                return {**state, 'is_prose_adequate': is_adequate, 'prose_feedback': feedback}
            except Exception as e_eval:
                log.error(f"Erreur lors de l'évaluation de la prose: {e_eval}", exc_info=True)
                # En cas d'erreur, on considère la prose comme non adéquate par défaut
                return {**state, 'is_prose_adequate': False, 'prose_feedback': f"Erreur lors de l'évaluation: {e_eval}", 'error_message': f"Erreur éval prose: {e_eval}"}


        def apply_guardrails_and_save(state: ReportWorkflowState) -> ReportWorkflowState:
            # (Logique interne de garde-fous et sauvegarde inchangée)
            sid = state['current_section_id']
            written_content = state.get('written_content')
            title = state.get('current_section_title', 'Section inconnue')
            if not sid: return {**state, 'error_message': 'ID section manquant (save).'}
            if not written_content:
                 log.warning(f"Contenu écrit manquant pour {sid}. Mise à jour statut en pending.")
                 update_status.invoke({'tool_input': f"{sid},pending"})
                 return {**state, 'error_message': f"Contenu vide pour {sid}"}

            log.info(f"Application garde-fous et sauvegarde pour: {title} (ID: {sid})")
            final_content = written_content
            try:
                mapping = config.ANONYMIZATION_MAPPING
                nb_replacements = 0
                for original, replacement in mapping.items():
                    final_content, count = re.subn(r'\b' + re.escape(original) + r'\b', replacement, final_content, flags=re.IGNORECASE)
                    if count > 0: log.info(f"Anonymisation: '{original}' -> '{replacement}' ({count} fois).")
                    nb_replacements += count
                log.info(f"Anonymisation terminée ({nb_replacements} rempl.) pour '{title}'.")
            except Exception as e_guard:
                 log.error(f"Erreur garde-fous pour {sid}: {e_guard}", exc_info=True)
                 final_content = written_content # Fallback

            status_to_save = 'drafted'
            try:
                memory_manager.update_section_content(sid, final_content, status_to_save)
                log.info(f"Contenu final et statut '{status_to_save}' sauvegardés pour {sid}.")
            except Exception as e_save:
                log.error(f"Erreur sauvegarde plan pour {sid}: {e_save}", exc_info=True)
                status_to_save = 'failed'
                update_status.invoke({'tool_input': f"{sid},{status_to_save}"})
                return {**state, 'error_message': f"Erreur sauvegarde plan ({sid}): {e_save}"}

            # Réinitialisation pour la prochaine itération
            return {
                **state,
                'current_section_id': None, 'current_section_title': None, 'structured_guidelines': [],
                'journal_context': None, 'is_context_relevant': None, 'analysis_result': None,
                'written_content': None, 'is_prose_adequate': None, 'prose_feedback': None,
                'final_content': final_content, # Garde le contenu de la dernière section
                'error_message': None, 'writing_attempts': 0,
            }

        # --- Logique de décision (Mise à jour pour intégrer les nouvelles étapes) ---
        def decide_next_step(state: ReportWorkflowState) -> str:
            log.debug(f"Décision basée sur l'état: ID={state.get('current_section_id')}, Iter={state.get('iterations')}, Err={state.get('error_message')}, Pending={state.get('pending_check_result')}")
            if state.get('error_message'):
                log.error(f"Erreur détectée: {state['error_message']}. Fin.")
                return END
            if state.get('iterations', 0) > args.max_iterations: # > car check_pending incrémente avant décision
                log.warning(f"Limite d'itérations ({args.max_iterations}) atteinte. Arrêt.")
                return END

            pending_result = state.get('pending_check_result', '')
            if 'COMPLETED' in pending_result:
                log.info("Toutes sections traitées. Fin.")
                return END

            # Si on vient de check_pending et qu'il y a une section à traiter
            if state.get('current_section_id') and state.get('current_section_title') is None:
                 log.info(f"Section {state['current_section_id']} trouvée -> 'prepare_drafting'")
                 return 'prepare_drafting'

            # Si on a préparé, on récupère le contexte
            if state.get('current_section_title') and state.get('journal_context') is None:
                 log.info("Préparation OK -> 'gather_focused_context'")
                 return 'gather_focused_context'

            # Si on a le contexte, on le filtre
            if state.get('journal_context') and state.get('is_context_relevant') is None:
                 log.info("Contexte RAG récupéré -> 'filter_rag_context'")
                 return 'filter_rag_context'

            # Si le contexte est filtré (pertinent ou non), on structure
            if state.get('is_context_relevant') is not None and state.get('analysis_result') is None:
                 log.info(f"Contexte RAG filtré (Pertinent: {state['is_context_relevant']}) -> 'structure_section_content'")
                 return 'structure_section_content'

            # Si on a structuré, on écrit la prose
            if state.get('analysis_result') and state.get('written_content') is None:
                 log.info("Structuration OK -> 'write_section_prose'")
                 return 'write_section_prose'

            # Si on a écrit la prose, on l'évalue
            if state.get('written_content') and state.get('is_prose_adequate') is None:
                 log.info("Prose écrite -> 'evaluate_written_prose'")
                 return 'evaluate_written_prose'

            # Si l'évaluation de la prose est terminée
            if state.get('is_prose_adequate') is not None:
                 if state['is_prose_adequate']:
                     log.info("Prose jugée adéquate -> 'apply_guardrails_and_save'")
                     return 'apply_guardrails_and_save'
                 else:
                     # Boucle de correction pour l'écriture
                     max_attempts = args.max_writing_attempts
                     if state['writing_attempts'] < max_attempts:
                         log.warning(f"Prose jugée inadéquate (Tentative {state['writing_attempts']}/{max_attempts}). Retour à 'write_section_prose' avec feedback.")
                         # Réinitialise seulement 'written_content' et 'is_prose_adequate' pour retenter
                         state['written_content'] = None
                         state['is_prose_adequate'] = None
                         # Le feedback est déjà dans l'état
                         return 'write_section_prose' # Retourne directement au nœud d'écriture
                     else:
                         log.error(f"Échec de la rédaction après {max_attempts} tentatives pour '{state['current_section_title']}'. Sauvegarde avec statut 'failed'.")
                         # Marquer comme échoué et sauvegarder quand même? Ou juste arrêter?
                         # Pour l'instant, on sauvegarde en failed.
                         state['error_message'] = f"Echec écriture après {max_attempts} tentatives."
                         # On force le passage à la sauvegarde qui mettra le statut 'failed' (ou gérera l'erreur)
                         return 'apply_guardrails_and_save' # Ou END ? Sauvegarder permet de voir le dernier essai.

            # Si on a sauvegardé, on retourne vérifier les sections restantes
            if state.get('final_content') is not None: # final_content est mis à jour par apply_guardrails_and_save
                 log.info("Sauvegarde OK -> 'check_pending'")
                 # Attention : final_content reste dans l'état, mais check_pending le réinitialise
                 return 'check_pending'


            # Cas d'erreur ou état inattendu
            log.error("État inattendu dans decide_next_step. Arrêt.")
            return END


        # --- Construction et exécution du graphe (Mise à jour) ---
        workflow = StateGraph(ReportWorkflowState)

        # Ajout des nœuds
        workflow.add_node("check_pending", check_pending)
        workflow.add_node("prepare_drafting", prepare_drafting)
        workflow.add_node("gather_focused_context", gather_focused_context)
        workflow.add_node("filter_rag_context", filter_rag_context) # Nouveau
        workflow.add_node("structure_section_content", structure_section_content)
        workflow.add_node("write_section_prose", write_section_prose)
        workflow.add_node("evaluate_written_prose", evaluate_written_prose) # Nouveau
        workflow.add_node("apply_guardrails_and_save", apply_guardrails_and_save)

        # Point d'entrée
        workflow.set_entry_point("check_pending")

        # Ajout des arêtes (Transitions)
        # Utilisation de la fonction decide_next_step pour simplifier
        workflow.add_conditional_edges(
            "check_pending",
            lambda x: decide_next_step(x), # Utilise la fonction de décision
             {
                "prepare_drafting": "prepare_drafting",
                END: END
            }
        )
        workflow.add_conditional_edges(
             "prepare_drafting",
             lambda x: decide_next_step(x),
             {
                 "gather_focused_context": "gather_focused_context",
                 END: END # Si erreur dans prepare_drafting
             }
        )
        workflow.add_conditional_edges(
             "gather_focused_context",
             lambda x: decide_next_step(x),
              {
                 "filter_rag_context": "filter_rag_context",
                 END: END # Si erreur dans gather_focused_context
             }
        )
        workflow.add_conditional_edges(
             "filter_rag_context",
             lambda x: decide_next_step(x),
              {
                 "structure_section_content": "structure_section_content",
                 END: END # Si erreur dans filter_rag_context
             }
        )
        workflow.add_conditional_edges(
             "structure_section_content",
             lambda x: decide_next_step(x),
              {
                 "write_section_prose": "write_section_prose",
                 END: END # Si erreur dans structure_section_content
             }
        )
         # La décision après write_section_prose mène à evaluate_written_prose
        workflow.add_conditional_edges(
             "write_section_prose",
             lambda x: decide_next_step(x),
              {
                 "evaluate_written_prose": "evaluate_written_prose",
                 END: END # Si erreur dans write_section_prose
             }
        )
         # La décision après evaluate_written_prose est clé pour la boucle de correction
        workflow.add_conditional_edges(
             "evaluate_written_prose",
             decide_next_step, # decide_next_step gère la logique Oui/Non/Max Attempts
             {
                 "apply_guardrails_and_save": "apply_guardrails_and_save", # Si Oui ou Max Attempts atteint
                 "write_section_prose": "write_section_prose", # Si Non et tentatives restantes
                 END: END # Si erreur dans evaluate_written_prose
             }
        )
        # Après sauvegarde, on retourne toujours à check_pending (géré par decide_next_step)
        workflow.add_conditional_edges(
             "apply_guardrails_and_save",
             lambda x: decide_next_step(x),
             {
                 "check_pending": "check_pending",
                 END: END # Si erreur pendant la sauvegarde
             }
        )


        # Compilation
        app = workflow.compile()
        log.info("Graphe LangGraph avec réflexion compilé.")

        # Exécution
        print("\n--- Démarrage exécution workflow avec réflexion ---")
        initial_state: ReportWorkflowState = {
            'objective': args.objective, 'current_section_id': None, 'current_section_title': None,
            'structured_guidelines': [], 'journal_context': None, 'is_context_relevant': None,
            'analysis_result': None, 'written_content': None, 'is_prose_adequate': None,
            'prose_feedback': None, 'final_content': None, 'pending_check_result': '',
            'iterations': 0, 'writing_attempts': 0, 'error_message': None,
        }
        recursion_limit = args.max_iterations * 15 + 30 # Augmenté pour les boucles potentielles
        log.info(f"Limite récursion: {recursion_limit}")

        try:
            final_state = app.invoke(initial_state, config={"recursion_limit": recursion_limit})
            print("\n--- Exécution workflow terminée ---")
            log.info(f"État final: {final_state}")
            # Affichage résultat (inchangé)
            if final_state.get('error_message'): print(f"\nSORTIE: Erreur: {final_state['error_message']}")
            elif final_state.get('pending_check_result') and 'COMPLETED' in final_state['pending_check_result']:
                print(f"\nSORTIE: Workflow terminé. Plan mis à jour: {args.plan_file}")
                print("Lancer 'assemble_report' pour générer le DOCX.")
            elif final_state.get('iterations', 0) > args.max_iterations: print(f"\nSORTIE: Limite {args.max_iterations} itérations atteinte.")
            else: print("\nSORTIE: Arrêt inattendu.")

        except Exception as e_invoke:
             log.critical(f"Erreur fatale invocation graphe: {e_invoke}", exc_info=True)
             sys.exit(1)

    else:
        log.error(f"Commande inconnue: {args.command}")
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOpération interrompue.")
        log.info("Interruption par utilisateur.")
        sys.exit(130)
    except SystemExit as e:
         if e.code != 0: log.warning(f"Sortie programme avec code: {e.code}")
         else: log.info("Sortie normale.")
         sys.exit(e.code)
    except Exception as e:
        log.critical(f"Erreur fatale non interceptée: {e}", exc_info=True)
        print(f"\nERREUR FATALE: {e}")
        sys.exit(1)