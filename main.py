# main.py (Version AVEC mémoire réactivée pour Ollama)

import argparse
import logging
import os
import sys
import json
import time
import ast

# --- Import des modules du projet ---
import config
from document_processor import process_all_journals, chunk_text, extract_text_from_pdf
from vector_database import VectorDBManager
from llm_interface import get_configured_llm
from data_models import JournalEntry, ReportPlan, ReportSection
from agent_tools import (
    SearchJournalEntriesTool, SearchGuidelinesTool, GetReportPlanStructureTool,
    GetPendingSectionsTool, UpdateSectionStatusTool, DraftSingleSectionTool
)
from tag_generator import TagGenerator
from competency_mapper import CompetencyMapper
from content_analyzer import ContentAnalyzer
from report_planner import ReportPlanner
from report_generator import ReportGenerator, assemble_report_from_plan
from quality_checker import QualityChecker
from visualization import Visualizer
from reference_manager import ReferenceManager
from progress_tracker import ProgressTracker
from memory_manager import MemoryManager
from typing import List, Dict, Any, Optional

# --- Imports LangChain ---
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
# --- MODIFICATION : Réactiver l'import de la mémoire ---
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.tools import BaseTool

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')
log = logging.getLogger(__name__)

# --- Fonction Principale ---
def main():
    # --- Configuration du Parseur d'Arguments ---
    parser = argparse.ArgumentParser(description="Agent IA (Embeddings Locaux, LLM Gemini ou Ollama via LangChain)")
    parser.add_argument("--llm", choices=["google", "ollama"], default=None,
                        help="Surcharge le fournisseur LLM défini dans config.py (google ou ollama).")
    subparsers = parser.add_subparsers(dest="command", help="Commandes disponibles")

    # (Définitions des subparsers - inchangées)
    parser_process = subparsers.add_parser("process_journals", help="Traite les journaux, extrait tags/compétences (LLM), stocke avec embeddings locaux.")
    parser_process.add_argument("--journal_dir", default=config.JOURNAL_DIR); parser_process.add_argument("--reprocess_all", action="store_true")
    parser_guidelines = subparsers.add_parser("process_guidelines", help="Traite le PDF des guidelines, stocke avec embeddings locaux.")
    parser_guidelines.add_argument("--pdf_path", default=config.GUIDELINES_PDF_PATH); parser_guidelines.add_argument("--reprocess", action="store_true")
    parser_plan = subparsers.add_parser("create_plan", help="Génère le plan du rapport (JSON) avec IDs uniques."); parser_plan.add_argument("--requirements_file", default=None); parser_plan.add_argument("--output_plan_file", default=config.DEFAULT_PLAN_FILE)
    parser_generate = subparsers.add_parser("generate_report", help="Génère un brouillon complet du rapport (non-agentique)."); parser_generate.add_argument("--plan_file", default=config.DEFAULT_PLAN_FILE); parser_generate.add_argument("--output_file", default=config.DEFAULT_REPORT_OUTPUT)
    parser_assemble = subparsers.add_parser("assemble_report", help="Assemble le rapport DOCX final depuis un plan JSON complété.")
    parser_assemble.add_argument("--plan_file", default=config.DEFAULT_PLAN_FILE, help="Chemin vers le fichier plan JSON (avec contenu).")
    parser_assemble.add_argument("--output_file", default=config.DEFAULT_REPORT_OUTPUT, help="Chemin pour sauvegarder le rapport DOCX assemblé.")
    parser_quality = subparsers.add_parser("check_quality", help="Vérifie la qualité du brouillon."); parser_quality.add_argument("--report_file", default=config.DEFAULT_REPORT_OUTPUT); parser_quality.add_argument("--plan_file", default=config.DEFAULT_PLAN_FILE); parser_quality.add_argument("--skip_journal_load", action="store_true")
    parser_visuals = subparsers.add_parser("create_visuals", help="Génère des visualisations."); parser_visuals.add_argument("--skip_journal_load", action="store_true")
    parser_refs = subparsers.add_parser("manage_refs", help="Gère la bibliographie."); ref_subparsers = parser_refs.add_subparsers(dest="ref_command", required=True); parser_add_ref = ref_subparsers.add_parser("add"); parser_add_ref.add_argument("--key", required=True); parser_add_ref.add_argument("--type", required=True, choices=['book', 'article', 'web', 'report', 'other']); parser_add_ref.add_argument("--author", required=True); parser_add_ref.add_argument("--year", required=True, type=int); parser_add_ref.add_argument("--title", required=True); parser_add_ref.add_argument("--data", default="{}"); parser_list_ref = ref_subparsers.add_parser("list")
    parser_agent = subparsers.add_parser("run_agent", help="Lance le workflow agentique avec LangChain."); parser_agent.add_argument("--max_iterations", type=int, default=50);
    parser_agent.add_argument(
        "--objective",
        default="Objectif Principal: Rédiger TOUTES les sections du rapport de Mission Professionnelle Digi5 conformément au plan. "
                "Processus: 1. Identifier la prochaine section 'pending' ou 'failed'. 2. Rédiger son contenu en français en respectant les guidelines Epitech et le contexte fourni. "
                "3. Mettre à jour son statut à 'drafted'. 4. Répéter jusqu'à ce qu'aucune section ne soit 'pending' ou 'failed'. "
                "Ne PAS s'arrêter avant que toutes les sections soient traitées ou qu'une erreur irrécupérable survienne."
    )
    parser_full = subparsers.add_parser("run_all", help="Lance le pipeline principal (proc_journ, proc_guide, plan, gen_report)."); parser_full.add_argument("--journal_dir", default=config.JOURNAL_DIR); parser_full.add_argument("--output_file", default=config.DEFAULT_REPORT_OUTPUT); parser_full.add_argument("--reprocess", action="store_true"); parser_full.add_argument("--skip_guidelines", action="store_true")

    args, remaining_argv = parser.parse_known_args()
    if args.command is None and remaining_argv:
        args = parser.parse_args(remaining_argv, namespace=args)
    if not args.command:
         parser.error("Aucune commande spécifiée. Choisissez parmi {process_journals, ...}")


    llm_provider = args.llm if args.llm else config.LLM_PROVIDER
    log.info(f"Fournisseur LLM sélectionné : {llm_provider}")

    log.info("Initialisation des composants partagés (MemoryManager, Planner)...")
    try: memory_manager = MemoryManager(); planner = ReportPlanner(); ref_manager = ReferenceManager(); log.info("Composants partagés initialisés.")
    except Exception as e: log.critical(f"ERREUR FATALE Init (Partagés): {e}", exc_info=True); sys.exit(1)

    try:
        # (Les blocs if/elif pour les commandes restent inchangés)
        if args.command == "process_journals":
             log.info("Initialisation des composants pour process_journals...");
             try:
                 llm = get_configured_llm(provider=llm_provider)
                 if not llm: raise RuntimeError(f"Impossible d'initialiser le LLM pour le fournisseur '{llm_provider}'")
                 vector_db = VectorDBManager(); tag_gen = TagGenerator(llm); comp_mapper = CompetencyMapper(llm)
             except Exception as e: log.critical(f"Erreur Init: {e}", exc_info=True); sys.exit(1)
             log.info("--- Commande: process_journals ---");
             pass # Placeholder

        elif args.command == "process_guidelines":
             log.info("--- Commande: process_guidelines ---")
             try: vector_db = VectorDBManager()
             except Exception as e: log.critical(f"Erreur Init: {e}", exc_info=True); sys.exit(1)
             pass # Placeholder

        elif args.command == "create_plan":
             log.info("--- Commande: create_plan ---")
             if args.requirements_file: log.warning("Chargement structure depuis fichier NI.")
             report_plan = planner.create_base_plan(); memory_manager.save_report_plan(report_plan, args.output_plan_file)
             log.info(f"Plan sauvegardé : {args.output_plan_file}"); print(f"\nPlan sauvegardé : {args.output_plan_file}")

        elif args.command == "assemble_report":
            log.info("--- Commande: assemble_report ---")
            log.info(f"Assemblage du rapport depuis le plan : {args.plan_file}")
            log.info(f"Le DOCX de sortie sera sauvegardé ici : {args.output_file}")
            assemble_report_from_plan(args.plan_file, args.output_file)
            log.info("--- Assemblage du rapport terminé ---")
            print(f"\nRapport assemblé et sauvegardé : {args.output_file}")

        elif args.command == "run_agent":
            log.info("--- Commande: run_agent (LangChain) ---")
            objective = args.objective; max_iterations = args.max_iterations
            log.info(f"Lancement de l'agent LangChain : objectif='{objective}', max_iterations={max_iterations}")

            try:
                log.info("Initialisation des composants pour l'exécution de l'agent...")
                vector_db_agent = VectorDBManager()
                memory_agent = MemoryManager()
                llm_agent = get_configured_llm(provider=llm_provider, temperature=0.2, max_tokens=2048)
                llm_tools = get_configured_llm(provider=llm_provider, temperature=0.6)
                if not llm_agent or not llm_tools:
                     raise RuntimeError(f"Impossible d'initialiser le LLM pour le fournisseur '{llm_provider}'")

                agent_tool_list: List[BaseTool] = [
                    SearchJournalEntriesTool(vector_db=vector_db_agent),
                    SearchGuidelinesTool(vector_db=vector_db_agent),
                    GetReportPlanStructureTool(memory_manager=memory_agent),
                    GetPendingSectionsTool(memory_manager=memory_agent),
                    UpdateSectionStatusTool(memory_manager=memory_agent),
                    DraftSingleSectionTool(vector_db=vector_db_agent, memory_manager=memory_agent, llm=llm_tools)
                ]
                log.info(f"Initialisé {len(agent_tool_list)} outils pour l'agent.")

                # --- MODIFICATION : Réactiver la mémoire ---
                agent_memory = ConversationSummaryBufferMemory(llm=llm_agent, max_token_limit=2500, memory_key="chat_history", return_messages=True)
                log.info("Mémoire de l'agent (ConversationSummaryBufferMemory) réactivée.")
                # --- FIN MODIFICATION ---

                prompt = hub.pull("hwchase17/react-chat"); log.info("Prompt ReAct chat chargé.")
                agent = create_react_agent(llm_agent, agent_tool_list, prompt); log.info("Agent ReAct créé.")

                # --- MODIFICATION : Repasser la mémoire à l'Executor ---
                agent_executor = AgentExecutor(
                    agent=agent,
                    tools=agent_tool_list,
                    memory=agent_memory, # Argument memory réactivé
                    verbose=True,
                    max_iterations=max_iterations,
                    handle_parsing_errors=True
                )
                log.info("Exécuteur d'agent créé AVEC mémoire.")
                # --- FIN MODIFICATION ---

            except Exception as e_agent_init: log.critical(f"Échec de l'initialisation des composants de l'agent : {e_agent_init}", exc_info=True); sys.exit(1)

            log.info(f"Invocation de l'agent : {objective}")
            print("\n--- Démarrage de l'exécution de l'agent LangChain (avec mémoire) ---") # Message mis à jour
            try:
                # L'appel invoke standard devrait fonctionner maintenant que la mémoire est fournie
                input_data = {"input": objective}
                result = agent_executor.invoke(input_data)
                print("\n--- Exécution de l'agent terminée ---");
                log.info(f"Sortie de l'agent : {result.get('output')}");
                print(f"\nSortie de l'agent :\n{result.get('output')}")
            except Exception as e_invoke: log.error(f"Erreur d'exécution de l'agent : {e_invoke}", exc_info=True); print(f"\nERREUR: {e_invoke}")

        else: log.error(f"Commande inconnue : {args.command}"); parser.print_help(); sys.exit(1)

    except Exception as e_main: log.critical(f"Erreur non gérée : {e_main}", exc_info=True); print(f"\nERREUR CRITIQUE : {e_main}."); sys.exit(1)

# --- Point d'Entrée Principal ---
if __name__ == "__main__":
    try: main()
    except SystemExit as e: exit_code = e.code if isinstance(e.code, int) else 1; sys.exit(exit_code)
    except KeyboardInterrupt: print("\nOpération annulée."); sys.exit(130)
    except Exception as e_global: log.critical(f"Erreur globale : {e_global}", exc_info=True); print(f"\nERREUR CRITIQUE : {e_global}."); sys.exit(1)