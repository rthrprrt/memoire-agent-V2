# main.py (Version LangChain Agent + Injection Dépendances Outils + Commande Assemble)

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
from llm_interface import get_langchain_llm
from data_models import JournalEntry, ReportPlan, ReportSection
# !! MODIFICATION IMPORT OUTILS !!
# Importer les CLASSES d'outils, pas la liste pré-instanciée
from agent_tools import (
    SearchJournalEntriesTool, SearchGuidelinesTool, GetReportPlanStructureTool,
    GetPendingSectionsTool, UpdateSectionStatusTool, DraftSingleSectionTool
)
# !! FIN MODIFICATION !!
from tag_generator import TagGenerator
from competency_mapper import CompetencyMapper
from content_analyzer import ContentAnalyzer
from report_planner import ReportPlanner
# <--- AJOUT : Importer la nouvelle fonction d'assemblage
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
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.tools import BaseTool # Pour type hinting liste outils

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')
log = logging.getLogger(__name__)

# --- Fonction Principale ---
def main():
    # --- Configuration du Parseur d'Arguments ---
    parser = argparse.ArgumentParser(description="AI Agent (Local Embeddings, Gemini LLM via LangChain)")
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

    # --- Définitions des subparsers ---
    parser_process = subparsers.add_parser("process_journals", help="Process journals, get tags/competencies (LLM), store with local embeddings.")
    parser_process.add_argument("--journal_dir", default=config.JOURNAL_DIR); parser_process.add_argument("--reprocess_all", action="store_true")

    parser_guidelines = subparsers.add_parser("process_guidelines", help="Process guidelines PDF, store with local embeddings.")
    parser_guidelines.add_argument("--pdf_path", default=config.GUIDELINES_PDF_PATH); parser_guidelines.add_argument("--reprocess", action="store_true")

    parser_plan = subparsers.add_parser("create_plan", help="Generate report plan JSON with unique IDs."); parser_plan.add_argument("--requirements_file", default=None); parser_plan.add_argument("--output_plan_file", default=config.DEFAULT_PLAN_FILE)

    parser_generate = subparsers.add_parser("generate_report", help="Generate full report draft (non-agentic)."); parser_generate.add_argument("--plan_file", default=config.DEFAULT_PLAN_FILE); parser_generate.add_argument("--output_file", default=config.DEFAULT_REPORT_OUTPUT)

    # <--- AJOUT : Subparser pour assemble_report ---
    parser_assemble = subparsers.add_parser("assemble_report", help="Assemble final DOCX report from a completed plan JSON file.")
    parser_assemble.add_argument("--plan_file", default=config.DEFAULT_PLAN_FILE, help="Path to the JSON report plan file with content.")
    parser_assemble.add_argument("--output_file", default=config.DEFAULT_REPORT_OUTPUT, help="Path to save the final assembled DOCX report.")
    # <--- FIN AJOUT ---

    parser_quality = subparsers.add_parser("check_quality", help="Run quality checks on draft."); parser_quality.add_argument("--report_file", default=config.DEFAULT_REPORT_OUTPUT); parser_quality.add_argument("--plan_file", default=config.DEFAULT_PLAN_FILE); parser_quality.add_argument("--skip_journal_load", action="store_true")

    parser_visuals = subparsers.add_parser("create_visuals", help="Generate visualizations."); parser_visuals.add_argument("--skip_journal_load", action="store_true")

    parser_refs = subparsers.add_parser("manage_refs", help="Manage bibliography."); ref_subparsers = parser_refs.add_subparsers(dest="ref_command", required=True); parser_add_ref = ref_subparsers.add_parser("add"); parser_add_ref.add_argument("--key", required=True); parser_add_ref.add_argument("--type", required=True, choices=['book', 'article', 'web', 'report', 'other']); parser_add_ref.add_argument("--author", required=True); parser_add_ref.add_argument("--year", required=True, type=int); parser_add_ref.add_argument("--title", required=True); parser_add_ref.add_argument("--data", default="{}"); parser_list_ref = ref_subparsers.add_parser("list")

    parser_agent = subparsers.add_parser("run_agent", help="Run agentic workflow using LangChain."); parser_agent.add_argument("--max_iterations", type=int, default=50); parser_agent.add_argument("--objective", default="Generate the full apprenticeship report section by section according to the plan.")

    parser_full = subparsers.add_parser("run_all", help="Run core pipeline (proc_journ, proc_guide, plan, gen_report)."); parser_full.add_argument("--journal_dir", default=config.JOURNAL_DIR); parser_full.add_argument("--output_file", default=config.DEFAULT_REPORT_OUTPUT); parser_full.add_argument("--reprocess", action="store_true"); parser_full.add_argument("--skip_guidelines", action="store_true")

    args = parser.parse_args()

    # --- Initialisation Différée des Composants (dans chaque commande si besoin) ---
    # Initialiser seulement memory_manager et planner ici car create_plan en a besoin
    log.info("Initializing shared components (MemoryManager, Planner)...")
    try: memory_manager = MemoryManager(); planner = ReportPlanner(); ref_manager = ReferenceManager(); log.info("Shared components initialized.")
    except Exception as e: log.critical(f"FATAL Init Error (Shared): {e}", exc_info=True); sys.exit(1)

    # --- Exécution de la Commande Sélectionnée ---
    try:
        if args.command == "process_journals":
             log.info("Initializing components for process_journals...");
             try: vector_db = VectorDBManager(); llm = get_langchain_llm(); tag_gen = TagGenerator(llm); comp_mapper = CompetencyMapper(llm)
             except Exception as e: log.critical(f"Init Error: {e}", exc_info=True); sys.exit(1)
             log.info("--- Command: process_journals ---"); # ... (Logique process_journals avec instances injectées) ...
             pass # Placeholder

        elif args.command == "process_guidelines":
             log.info("--- Command: process_guidelines ---")
             try: vector_db = VectorDBManager()
             except Exception as e: log.critical(f"Init Error: {e}", exc_info=True); sys.exit(1)
             # ... (Logique process_guidelines) ...
             pass # Placeholder

        elif args.command == "create_plan":
             log.info("--- Command: create_plan ---")
             # Utilise planner et memory_manager initialisés plus haut
             if args.requirements_file: log.warning("Loading structure from file NI.")
             report_plan = planner.create_base_plan(); memory_manager.save_report_plan(report_plan, args.output_plan_file)
             log.info(f"Plan saved to {args.output_plan_file}"); print(f"\nPlan saved: {args.output_plan_file}")

        # <--- AJOUT : Bloc d'exécution pour assemble_report ---
        elif args.command == "assemble_report":
            log.info("--- Command: assemble_report ---")
            log.info(f"Assembling report from plan: {args.plan_file}")
            log.info(f"Output DOCX will be saved to: {args.output_file}")
            # Appeler la fonction d'assemblage
            assemble_report_from_plan(args.plan_file, args.output_file)
            log.info("--- Report assembly finished ---")
            print(f"\nReport assembled and saved to: {args.output_file}")
        # <--- FIN AJOUT ---

        # ... (Autres commandes non-agentiques - nécessitent initialisation de leurs dépendances) ...

        elif args.command == "run_agent":
            log.info("--- Command: run_agent (LangChain) ---")
            objective = args.objective; max_iterations = args.max_iterations
            log.info(f"Starting LangChain agent: objective='{objective}', max_iterations={max_iterations}")

            # Initialiser TOUTES les dépendances nécessaires pour l'agent et ses outils ICI
            try:
                log.info("Initializing components for agent execution...")
                vector_db_agent = VectorDBManager()
                memory_agent = MemoryManager() # Instance mémoire pour les outils
                llm_agent = get_langchain_llm(temperature=0.2, max_tokens=2048) # LLM pour l'agent lui-même
                llm_tools = get_langchain_llm(temperature=0.6) # LLM séparé pour l'outil draft (peut être le même)
                if not all([vector_db_agent, memory_agent, llm_agent, llm_tools]):
                     raise RuntimeError("One or more core components failed to initialize.")

                # Créer les instances d'outils en injectant les dépendances
                agent_tool_list: List[BaseTool] = [
                    SearchJournalEntriesTool(vector_db=vector_db_agent),
                    SearchGuidelinesTool(vector_db=vector_db_agent),
                    GetReportPlanStructureTool(memory_manager=memory_agent),
                    GetPendingSectionsTool(memory_manager=memory_agent),
                    UpdateSectionStatusTool(memory_manager=memory_agent),
                    DraftSingleSectionTool(vector_db=vector_db_agent, memory_manager=memory_agent, llm=llm_tools)
                ]
                log.info(f"Initialized {len(agent_tool_list)} tools for agent.")

                # Initialiser la mémoire de l'agent LangChain
                agent_memory = ConversationSummaryBufferMemory(llm=llm_agent, max_token_limit=2500, memory_key="chat_history", return_messages=True)

                # Charger le prompt
                prompt = hub.pull("hwchase17/react-chat"); log.info("Loaded ReAct chat prompt.")

                # Créer l'agent et l'exécuteur
                agent = create_react_agent(llm_agent, agent_tool_list, prompt); log.info("ReAct agent created.")
                agent_executor = AgentExecutor(agent=agent, tools=agent_tool_list, memory=agent_memory, verbose=True, max_iterations=max_iterations, handle_parsing_errors=True); log.info("Agent Executor created.")

            except Exception as e_agent_init: log.critical(f"Failed to initialize agent components: {e_agent_init}", exc_info=True); sys.exit(1)

            # Lancer l'Agent
            log.info(f"Invoking agent: {objective}")
            print("\n--- Starting LangChain Agent Execution ---")
            try: result = agent_executor.invoke({ "input": objective }); print("\n--- Agent Execution Finished ---"); log.info(f"Agent output: {result.get('output')}"); print(f"\nAgent Output:\n{result.get('output')}")
            except Exception as e_invoke: log.error(f"Agent execution error: {e_invoke}", exc_info=True); print(f"\nERROR: {e_invoke}")

        else: log.error(f"Unknown command: {args.command}"); parser.print_help(); sys.exit(1)

    except Exception as e_main: log.critical(f"Unhandled error: {e_main}", exc_info=True); print(f"\nCRITICAL ERROR: {e_main}."); sys.exit(1)

# --- Point d'Entrée Principal ---
if __name__ == "__main__":
    try: main()
    except SystemExit as e: exit_code = e.code if isinstance(e.code, int) else 1; sys.exit(exit_code)
    except KeyboardInterrupt: print("\nOperation cancelled."); sys.exit(130)
    except Exception as e_global: log.critical(f"Global error: {e_global}", exc_info=True); print(f"\nCRITICAL ERROR: {e_global}."); sys.exit(1)