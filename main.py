# main.py (Version finale propre utilisant GeminiLLM + Fix historique + Fix parse args)

import argparse
import logging
import os
import sys
import json
import time # Utilisé par agent_tools via vector_database
import ast  # Pour ast.literal_eval

# --- Import des modules du projet ---
import config
from document_processor import process_all_journals, chunk_text, extract_text_from_pdf
from vector_database import VectorDBManager # Utilise embeddings locaux
from llm_interface import GeminiLLM # Utilise la classe pour Google AI
from data_models import JournalEntry, ReportPlan # Pour type hints
import agent_tools # Pour les outils de l'agent
from tag_generator import TagGenerator
from competency_mapper import CompetencyMapper
from content_analyzer import ContentAnalyzer
from report_planner import ReportPlanner
from report_generator import ReportGenerator
from quality_checker import QualityChecker
from visualization import Visualizer
from reference_manager import ReferenceManager
from progress_tracker import ProgressTracker
from memory_manager import MemoryManager
from typing import List, Dict, Any, Optional # Ajout de Any/Optional

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')
log = logging.getLogger(__name__) # Utiliser __name__ pour le logger du module

# --- Fonction Principale ---
def main():
    # --- Configuration du Parseur d'Arguments ---
    parser = argparse.ArgumentParser(description="AI Agent for Apprenticeship Report Generation (Embeddings: Local, LLM: Google Gemini)")
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

    # (Définitions des subparsers - identiques à avant)
    # -- Commande: process_journals --
    parser_process = subparsers.add_parser("process_journals", help="Process journals, generate embeddings locally, call LLM API for tags/competencies, store in DB.")
    parser_process.add_argument("--journal_dir", default=config.JOURNAL_DIR, help="Directory with journal DOCX files (YYYY-MM-DD.docx).")
    parser_process.add_argument("--reprocess_all", action="store_true", help="Clear existing journal vector DB collection first.")
    # -- Commande: process_guidelines --
    parser_guidelines = subparsers.add_parser("process_guidelines", help="Process guidelines PDF, generate embeddings locally, store in reference DB.")
    parser_guidelines.add_argument("--pdf_path", default=config.GUIDELINES_PDF_PATH, help="Path to the guidelines PDF file.")
    parser_guidelines.add_argument("--reprocess", action="store_true", help="Clear existing reference vector DB collection first.")
    # -- Commande: create_plan --
    parser_plan = subparsers.add_parser("create_plan", help="Generate the report structure plan (JSON).")
    parser_plan.add_argument("--requirements_file", default=None, help="(Optional) Path to custom structure file.")
    parser_plan.add_argument("--output_plan_file", default=config.DEFAULT_PLAN_FILE, help="Path to save the plan JSON.")
    # -- Commande: generate_report --
    parser_generate = subparsers.add_parser("generate_report", help="Generate report draft using LLM API based on plan and local context.")
    parser_generate.add_argument("--plan_file", default=config.DEFAULT_PLAN_FILE, help="Path to the report plan JSON.")
    parser_generate.add_argument("--output_file", default=config.DEFAULT_REPORT_OUTPUT, help="Path to save the DOCX draft.")
    # -- Commande: check_quality --
    parser_quality = subparsers.add_parser("check_quality", help="Run quality checks on a generated report draft using LLM API.")
    parser_quality.add_argument("--report_file", default=config.DEFAULT_REPORT_OUTPUT, help="Path to the report DOCX to check.")
    parser_quality.add_argument("--plan_file", default=config.DEFAULT_PLAN_FILE, help="Path to the report plan JSON.")
    parser_quality.add_argument("--skip_journal_load", action="store_true", help="Skip reloading journals for plagiarism check.")
    # -- Commande: create_visuals --
    parser_visuals = subparsers.add_parser("create_visuals", help="Generate visualizations (requires processed journals).")
    parser_visuals.add_argument("--skip_journal_load", action="store_true", help="Skip reloading journals.")
    # -- Commande: manage_refs --
    parser_refs = subparsers.add_parser("manage_refs", help="Manage bibliography references.")
    ref_subparsers = parser_refs.add_subparsers(dest="ref_command", help="Reference actions", required=True)
    parser_add_ref = ref_subparsers.add_parser("add", help="Add a new reference.")
    parser_add_ref.add_argument("--key", required=True); parser_add_ref.add_argument("--type", required=True, choices=['book', 'article', 'web', 'report', 'other'])
    parser_add_ref.add_argument("--author", required=True); parser_add_ref.add_argument("--year", required=True, type=int); parser_add_ref.add_argument("--title", required=True)
    parser_add_ref.add_argument("--data", default="{}", help="JSON string with additional type-specific fields.")
    parser_list_ref = ref_subparsers.add_parser("list", help="List stored references.")
    # -- Commande: run_agent --
    parser_agent = subparsers.add_parser("run_agent", help="Run the experimental agent loop using LLM API.")
    parser_agent.add_argument("--max_turns", type=int, default=10, help="Maximum interaction turns.")
    parser_agent.add_argument("--objective", default="Draft the 'Introduction' section.", help="Initial agent objective.")
    # -- Commande: run_all --
    parser_full = subparsers.add_parser("run_all", help="Run full workflow (process journals & guidelines, plan, generate).")
    parser_full.add_argument("--journal_dir", default=config.JOURNAL_DIR); parser_full.add_argument("--output_file", default=config.DEFAULT_REPORT_OUTPUT)
    parser_full.add_argument("--reprocess", action="store_true", help="Force reprocessing of journals AND guidelines.")
    parser_full.add_argument("--skip_guidelines", action="store_true", help="Skip processing guidelines.")

    args = parser.parse_args()

    # --- Initialisation des Composants Clés ---
    log.info("Initializing core components...")
    try:
        memory = MemoryManager(); vector_db = VectorDBManager(); llm = GeminiLLM()
        log.info(f"LLM Interface initialized using Google AI model: {config.GEMINI_CHAT_MODEL_NAME}")
        tag_gen = TagGenerator(); comp_mapper = CompetencyMapper(); analyzer = ContentAnalyzer()
        planner = ReportPlanner(); generator = ReportGenerator(vector_db); quality_checker = QualityChecker(vector_db)
        visualizer = Visualizer(); ref_manager = ReferenceManager(); progress_tracker = ProgressTracker()
        log.info("Core components initialized successfully.")
    except Exception as e: log.critical(f"FATAL: Failed to initialize core components: {e}", exc_info=True); sys.exit(1)

    # --- Exécution de la Commande Sélectionnée ---
    try:
        if args.command == "process_journals":
            log.info("--- Command: process_journals ---")
            if args.reprocess_all: log.warning("Clearing *journal* collection..."); vector_db.clear_journal_collection()
            log.info(f"Processing journals from: {args.journal_dir}")
            journal_entries: List[JournalEntry] = process_all_journals(args.journal_dir)
            if not journal_entries: log.error("No valid journals found. Exiting."); sys.exit(1)
            log.info(f"Found {len(journal_entries)} entries.")
            log.info("Generating tags (using Gemini)..."); journal_entries = tag_gen.process_entries(journal_entries)
            log.info("Mapping competencies (using Gemini)..."); journal_entries = comp_mapper.process_entries(journal_entries)
            log.info("Adding journal entries to vector DB (local embeddings)...")
            processed_count, error_count_db = 0, 0
            for entry in journal_entries:
                try:
                    chunks = chunk_text(entry.raw_text)
                    if chunks: entry_data_for_db = {"entry_id": entry.entry_id, "chunks": chunks, "date_iso": entry.date.isoformat(),"source_file": entry.source_file, "tags_str": ",".join(entry.tags or [])}; vector_db.add_entry_chunks(entry_data_for_db); processed_count += 1
                    else: log.warning(f"Entry {entry.entry_id} had no chunks.")
                except Exception as e_add: log.error(f"Failed to add chunks for {entry.entry_id}: {e_add}", exc_info=True); error_count_db += 1
            log.info(f"--- Finished Journal Processing: {processed_count} entries added. {error_count_db} errors. ---")

        elif args.command == "process_guidelines":
            log.info("--- Command: process_guidelines ---")
            pdf_path = args.pdf_path
            if not os.path.exists(pdf_path): log.error(f"Guidelines PDF not found: {pdf_path}"); sys.exit(1)
            if args.reprocess: log.warning("Clearing *reference* collection..."); vector_db.clear_reference_collection()
            log.info(f"Processing PDF: {pdf_path}"); pdf_text = extract_text_from_pdf(pdf_path)
            if not pdf_text: log.error(f"No text extracted from PDF '{pdf_path}'."); sys.exit(1)
            pdf_chunks = chunk_text(pdf_text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
            if not pdf_chunks: log.error("Failed to chunk PDF text."); sys.exit(1)
            log.info(f"Adding {len(pdf_chunks)} guideline chunks to DB...")
            doc_name = os.path.basename(pdf_path); vector_db.add_reference_chunks(doc_name=doc_name, chunks=pdf_chunks)
            log.info(f"--- Finished Guidelines Processing: Added guidelines from '{doc_name}' ---")

        elif args.command == "create_plan":
            log.info("--- Command: create_plan ---")
            structure_def = None # TODO: Implement loading from --requirements_file
            if args.requirements_file: log.warning("Loading structure from file NI.")
            report_plan: ReportPlan = planner.create_base_plan(structure_definition=structure_def)
            memory.save_report_plan(report_plan, args.output_plan_file)
            log.info(f"Report plan saved to {args.output_plan_file}"); print(f"\nPlan saved to: {args.output_plan_file}")

        elif args.command == "generate_report":
            log.info("--- Command: generate_report ---")
            report_plan = memory.load_report_plan(args.plan_file)
            if not report_plan: log.error(f"Plan file not found: {args.plan_file}"); sys.exit(1)
            log.info(f"Starting report generation (LLM: Gemini). Output: {args.output_file}")
            updated_plan = generator.generate_full_report(report_plan, args.output_file)
            memory.save_report_plan(updated_plan, args.plan_file)
            log.info(f"Report generation finished. Draft: '{args.output_file}'. Plan: '{args.plan_file}'.")
            print(f"\nReport draft generated: {args.output_file}")

        elif args.command == "check_quality":
            log.info("--- Command: check_quality ---")
            report_plan = memory.load_report_plan(args.plan_file)
            if not os.path.exists(args.report_file): log.error(f"Report file not found: {args.report_file}"); sys.exit(1)
            if not args.skip_journal_load:
                log.info("Loading journals for plagiarism check...")
                temp_journal_entries = process_all_journals(config.JOURNAL_DIR)
                if temp_journal_entries: quality_checker.load_journal_texts(temp_journal_entries)
                else: log.warning("Could not load journals for plagiarism check.")
            log.info(f"Checking quality of: {args.report_file} (LLM: Gemini)...")
            gap_issues, consistency_issues = [], []
            if report_plan:
                gap_issues = quality_checker.identify_content_gaps(report_plan)
                consistency_issues = quality_checker.check_consistency_across_sections(report_plan)
            plagiarism_issues, copied_percentage = quality_checker.check_plagiarism_against_journals(args.report_file)
            # Affichage...
            print("\n--- Quality Check Results ---"); print(f"File: {args.report_file}")
            if report_plan: completed, total, percent = progress_tracker.calculate_progress(report_plan); print(f"\nProgress: {completed}/{total} sections ({percent:.1f}%)")
            if gap_issues: print("\n[!] Gaps:"); [print(f"  - {i}") for i in gap_issues]
            else: print("\n[*] No gaps detected.")
            if consistency_issues: print("\n[!] Consistency Issues:"); [print(f"  - {i}") for i in consistency_issues]
            else: print("\n[*] No consistency issues detected.")
            if quality_checker.original_journal_texts: print(f"\n[*] Plagiarism Check:"); print(f"  - Est. Copied: {copied_percentage:.1f}%"); [print(f"    - {i}") for i in plagiarism_issues[:5]]; print(f"      ... ({len(plagiarism_issues)-5} more)" if len(plagiarism_issues)>5 else "")
            else: print("\n[!] Plagiarism check skipped.")
            print("-----------------------------\n")

        elif args.command == "create_visuals":
            log.info("--- Command: create_visuals ---")
            if args.skip_journal_load: log.error("--skip_journal_load NI."); sys.exit(1)
            journal_entries = process_all_journals(config.JOURNAL_DIR)
            if not journal_entries: log.error("No journals found."); sys.exit(1)
            log.info("Mapping data for visuals (LLM: Gemini)...")
            journal_entries = tag_gen.process_entries(journal_entries)
            journal_entries = comp_mapper.process_entries(journal_entries)
            log.info("Generating plots...")
            competency_timeline = comp_mapper.get_competency_timeline(journal_entries)
            visualizer.plot_competency_timeline(competency_timeline)
            project_mentions = analyzer.identify_mentioned_projects(journal_entries)
            visualizer.plot_project_activity(project_mentions)
            log.info(f"Visualizations saved in '{config.OUTPUT_DIR}'."); print(f"\nVisualizations saved in: {config.OUTPUT_DIR}")

        elif args.command == "manage_refs":
            log.info("--- Command: manage_refs ---")
            if args.ref_command == "add":
                log.info(f"Adding reference: {args.key}")
                try: extra_data = json.loads(args.data); assert isinstance(extra_data, dict)
                except: log.error(f"Invalid JSON: {args.data}", exc_info=True); print("Error: Invalid JSON in --data."); sys.exit(1)
                citation_data = {"author": args.author, "year": args.year, "title": args.title, **extra_data}
                ref_manager.add_citation(args.key, args.type, citation_data); print(f"Ref '{args.key}' added.")
            elif args.ref_command == "list":
                log.info("Listing references..."); citations = ref_manager._load_citations()
                if not citations: print("No references found.")
                else: print(f"\n--- Stored References ---"); print(ref_manager.generate_bibliography_text()); print("--------------------")

        elif args.command == "run_agent":
            # Définir les outils DANS le scope de cette commande
            AVAILABLE_TOOLS = {
                "search_journal_entries": agent_tools.search_journal_entries,
                "search_guidelines": agent_tools.search_guidelines,
                "get_report_plan_structure": agent_tools.get_report_plan_structure,
                "get_pending_sections": agent_tools.get_pending_sections,
                "update_section_status": agent_tools.update_section_status
            }
            log.info("--- Command: run_agent ---")
            objective = args.objective; max_turns = args.max_turns
            log.info(f"Starting agent: objective='{objective}', max_turns={max_turns}, LLM=Gemini")

            # --- Prompt Système ---
            system_prompt = f"""
You are an AI assistant tasked with writing an apprenticeship report based on journal entries and guidelines.
Your goal is to fulfill the user's objective: "{objective}".
You have access to the following tools:

1.  **search_journal_entries(query: str, k: int = 5)**:
    - Finds relevant passages from **Year 2 journal entries**. Use for details on projects, tasks, skills, challenges.
    - Example: `>>>TOOL_CALL\nsearch_journal_entries(query="challenges with Copilot access control", k=3)\n>>>END_TOOL_CALL`

2.  **search_guidelines(topic: str, k: int = 3)**:
    - Consults the official report **guidelines PDF** for requirements on a topic/section.
    - Example: `>>>TOOL_CALL\nsearch_guidelines(topic="Conclusion requirements", k=2)\n>>>END_TOOL_CALL`

3.  **get_report_plan_structure()**:
    - Retrieves the current **outline of the report**, including section titles, hierarchy, IDs, and generation status.
    - Use this to understand the overall structure or check section statuses. Takes no arguments.
    - Example: `>>>TOOL_CALL\nget_report_plan_structure()\n>>>END_TOOL_CALL`

4.  **get_pending_sections()**:
    - Returns a comma-separated list of **section IDs** that are marked as 'pending' or 'failed', indicating they need work.
    - Use this to find out which sections to work on next. Takes no arguments.
    - Example: `>>>TOOL_CALL\nget_pending_sections()\n>>>END_TOOL_CALL`

5.  **update_section_status(section_id: str, new_status: str)**:
    - Updates the status of a specific section in the plan (identified by its ID).
    - Allowed statuses: 'pending', 'drafting', 'drafted', 'failed', 'reviewing', 'approved'.
    - Use this *after* you have successfully drafted or attempted to draft a section.
    - Example: `>>>TOOL_CALL\nupdate_section_status(section_id="sec-intro-123", new_status="drafted")\n>>>END_TOOL_CALL`

**Workflow:**
1. **Understand Objective:** Analyze the goal (e.g., "{objective}").
2. **Plan Steps (Optional but helpful):** If needed, check the report plan (`get_report_plan_structure`) and identify pending sections (`get_pending_sections`). Decide which section to work on.
3. **Gather Info:** Use `search_guidelines` and `search_journal_entries` to get necessary context for the chosen section.
4. **Draft/Act:** Generate the required text or perform the action.
5. **Update Status:** If you drafted a section, update its status using `update_section_status`.
6. **Repeat:** Continue until the overall objective is met.

**Important:**
- Call ONE tool per turn using the EXACT format `>>>TOOL_CALL...>>>END_TOOL_CALL`.
- Use the results of tools to inform your next actions.
- Keep track of the main objective.
"""
            conversation_history = [{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Start: {objective}"}]

            # --- Boucle Agentique ---
            for turn in range(max_turns):
                log.info(f"--- Agent Turn {turn + 1}/{max_turns} ---"); print(f"\n--- Turn {turn + 1}/{max_turns} ---")

                # !! AJOUT CORRIGÉ : Préparer l'historique pour l'API Gemini !!
                current_history_for_api = conversation_history[:]
                if current_history_for_api[-1]['role'] == 'assistant':
                    log.warning("Last message was from assistant. Appending placeholder user message for Gemini API.")
                    placeholder_msg = "Okay, what is the next logical step to achieve the objective, based on our conversation?"
                    current_history_for_api.append({"role": "user", "content": placeholder_msg})
                # !! FIN AJOUT CORRIGÉ !!

                # 1. Appel LLM (Gemini)
                log.debug(f"Sending context to LLM (last role for API: {current_history_for_api[-1]['role']})")
                # Utiliser l'historique potentiellement modifié
                llm_response_content = llm._make_request(messages=current_history_for_api, max_tokens=1500, temperature=0.5)

                if not llm_response_content: log.error("LLM failed."); break
                log.info(f"LLM Raw Response: {llm_response_content[:200]}..."); print(f"\nAgent:\n{llm_response_content}\n")
                # Ajouter la VRAIE réponse au VRAI historique
                conversation_history.append({"role": "assistant", "content": llm_response_content})

                # --- Détection/Exécution Outil (Bloc Corrigé) ---
                tool_call_prefix = ">>>TOOL_CALL"; tool_call_suffix = ">>>END_TOOL_CALL"; tool_result = None; tool_executed = False
                if tool_call_prefix in llm_response_content:
                    try: # Bloc try englobant
                        call_start = llm_response_content.find(tool_call_prefix) + len(tool_call_prefix)
                        call_end = llm_response_content.find(tool_call_suffix, call_start)
                        if call_end != -1:
                            tool_call_str = llm_response_content[call_start:call_end].strip(); log.info(f"Detected: {tool_call_str}"); tool_executed = True
                            tool_name = tool_call_str.split("(", 1)[0].strip() if "(" in tool_call_str else tool_call_str
                            if tool_name in AVAILABLE_TOOLS:
                                # --- Bloc de Parsing d'Arguments Corrigé ---
                                tool_args = {}
                                args_str = ""
                                try: # Pour extraire la chaîne d'args
                                    args_start_index = tool_call_str.index('(') + 1
                                    args_end_index = tool_call_str.rindex(')')
                                    args_str = tool_call_str[args_start_index:args_end_index].strip()
                                except ValueError: args_str = "" # Pas d'args

                                if args_str: # Parser si non vide
                                    try: # Pour parser les paires
                                        arg_pairs = args_str.split(',')
                                        for pair in arg_pairs:
                                            pair = pair.strip()
                                            if '=' in pair:
                                                key, val_str = map(str.strip, pair.split('=', 1))
                                                try: # Pour literal_eval
                                                    val = ast.literal_eval(val_str)
                                                except (ValueError, SyntaxError): val = val_str.strip('\'"') # Fallback string
                                                tool_args[key] = val
                                            elif pair: log.warning(f"Arg part '{pair}' ignored.")
                                    except Exception as e_parse: log.error(f"Failed to parse args '{args_str}': {e_parse}"); tool_result = f"Error parsing args: {e_parse}"
                                # --- Fin Bloc de Parsing ---

                                # Exécuter si pas d'erreur de parsing
                                if tool_result is None:
                                    try: # Pour l'exécution
                                        log.info(f"Executing tool '{tool_name}' with args: {tool_args}")
                                        tool_function = AVAILABLE_TOOLS[tool_name]; tool_result = tool_function(**tool_args)
                                        log.info(f"Tool '{tool_name}' OK. Result len: {len(tool_result or '')}")
                                    except Exception as e_exec: log.error(f"Failed to execute tool '{tool_name}': {e_exec}", exc_info=True); tool_result = f"Error executing tool: {e_exec}"
                            else: log.warning(f"Unknown tool: '{tool_name}'"); tool_result = f"Error: Unknown tool '{tool_name}'."
                        else: log.warning("Invalid tool call format."); tool_result = "Error: Invalid format."
                    except Exception as e_t: log.error(f"Error processing tool call: {e_t}", exc_info=True); tool_result = f"Error: {e_t}"

                    # Ajouter résultat/erreur à l'historique ORIGINAL
                    tool_result_for_llm = tool_result if tool_result is not None else "Error: Tool exec failed."
                    msg = f">>>TOOL_RESULT\n{tool_result_for_llm}\n>>>END_TOOL_RESULT"
                    conversation_history.append({"role": "user", "content": msg}); print(f"\nTool Result:\n{msg}\n")
                # --- Fin Détection/Exécution Outil ---

                # 3. Condition d'Arrêt
                if not tool_executed and ("objective completed" in llm_response_content.lower() or "task finished" in llm_response_content.lower()):
                    log.info("Agent suggests completion."); print("\nAgent suggests completion."); break
                if turn == max_turns - 1: log.info("Max turns reached."); print("\nMaximum turns reached."); break
            # Fin de la boucle for turn

        elif args.command == "run_all":
            log.info("--- Command: run_all (Executing full workflow) ---")
            def run_subcommand(step_name: str, command: str): # Fonction interne
                log.info(f"Executing Step: {step_name}"); log.debug(f"Running: {command}")
                exit_code = os.system(command)
                if exit_code != 0: log.error(f"Step '{step_name}' failed (Code: {exit_code}). Aborting."); sys.exit(exit_code)
                log.info(f"Step '{step_name}' completed.")
            steps = 6; current_step = 1
            reprocess_j = "--reprocess_all" if args.reprocess else ""; reprocess_g = "--reprocess" if args.reprocess else ""
            print(f"\n>>> [{current_step}/{steps}] Proc Journals..."); cmd1 = f'python "{sys.argv[0]}" process_journals --journal_dir "{args.journal_dir}" {reprocess_j}'; run_subcommand("Proc Journals", cmd1); current_step+=1
            if not args.skip_guidelines: print(f"\n>>> [{current_step}/{steps}] Proc Guidelines..."); cmd2 = f'python "{sys.argv[0]}" process_guidelines {reprocess_g}'; run_subcommand("Proc Guidelines", cmd2)
            else: print(f"\n>>> [{current_step}/{steps}] Skipping Guidelines..."); current_step+=1
            print(f"\n>>> [{current_step}/{steps}] Create Plan..."); cmd3 = f'python "{sys.argv[0]}" create_plan'; run_subcommand("Create Plan", cmd3); current_step+=1
            print(f"\n>>> [{current_step}/{steps}] Generate Report..."); cmd4 = f'python "{sys.argv[0]}" generate_report --output_file "{args.output_file}"'; run_subcommand("Generate Report", cmd4); current_step+=1
            print(f"\n>>> [{current_step}/{steps}] Quality Checks..."); cmd5 = f'python "{sys.argv[0]}" check_quality --report_file "{args.output_file}"'; os.system(cmd5); log.info("Quality Checks finished."); current_step+=1
            print(f"\n>>> [{current_step}/{steps}] Create Visuals..."); cmd6 = f'python "{sys.argv[0]}" create_visuals'; os.system(cmd6); log.info("Visualizations finished."); current_step+=1
            log.info("--- Full Workflow Attempt Finished ---"); print(f"\nWorkflow finished. Draft: {args.output_file}")

        else: # Commande inconnue
            log.error(f"Unknown command provided: {args.command}")
            parser.print_help()
            sys.exit(1)

    # Gestion globale des exceptions pour la fonction main
    except Exception as e_main:
        log.critical(f"An unhandled error occurred in the main execution block: {e_main}", exc_info=True)
        print(f"\nUNHANDLED CRITICAL ERROR: {e_main}. Please check the logs.")
        sys.exit(1)


# --- Point d'Entrée Principal ---
if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        exit_code = e.code if isinstance(e.code, int) else 1
        if exit_code != 0: log.warning(f"Script exited with code {exit_code}.")
        # else: log.info("Script finished successfully.") # Loggé à la fin de main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        log.warning("Script interrupted by user (Ctrl+C).")
        print("\nOperation cancelled by user.")
        sys.exit(130)
    except Exception as e_global:
        log.critical(f"An unexpected critical error occurred outside the main function block: {e_global}", exc_info=True)
        print(f"\nCRITICAL ERROR: {e_global}. Check logs.")
        sys.exit(1)