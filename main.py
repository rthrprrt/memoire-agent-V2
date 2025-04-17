# main.py (Version propre et corrigée utilisant GeminiLLM)

import argparse
import ast
import logging
import os
import sys
import json
import time # Importé pour clear_collection dans vector_database (via agent_tools)

# --- Import des modules du projet ---
import config
from document_processor import process_all_journals, chunk_text, extract_text_from_pdf
from vector_database import VectorDBManager # Utilise embeddings locaux
# Utilisation de l'interface pour Google Gemini
from llm_interface import GeminiLLM
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
        memory = MemoryManager()
        vector_db = VectorDBManager()
        llm = GeminiLLM() # Utilise la classe pour Google AI
        log.info(f"LLM Interface initialized using Google AI model: {config.GEMINI_CHAT_MODEL_NAME}")
        # Injecter l'instance llm si nécessaire (design actuel suppose accès global ou __init__)
        # tag_gen = TagGenerator(llm) # Design Explicite (préférable à terme)
        tag_gen = TagGenerator()     # Design Actuel
        comp_mapper = CompetencyMapper() # Idem
        analyzer = ContentAnalyzer()    # Idem
        planner = ReportPlanner()
        generator = ReportGenerator(vector_db) # A besoin de vector_db ; utilise llm implicitement
        quality_checker = QualityChecker(vector_db) # A besoin de vector_db ; utilise llm implicitement
        visualizer = Visualizer()
        ref_manager = ReferenceManager()
        progress_tracker = ProgressTracker()
        log.info("Core components initialized successfully.")
    except Exception as e:
        log.critical(f"FATAL: Failed to initialize core components: {e}", exc_info=True)
        sys.exit(1)

    # --- Exécution de la Commande Sélectionnée ---
    try:
        if args.command == "process_journals":
            log.info("--- Command: process_journals ---")
            if args.reprocess_all:
                log.warning("Option --reprocess_all selected. Clearing *journal* collection...")
                vector_db.clear_journal_collection()

            log.info(f"Processing journals from: {args.journal_dir}")
            journal_entries: List[JournalEntry] = process_all_journals(args.journal_dir)
            if not journal_entries: log.error("No valid journals found. Exiting."); sys.exit(1)
            log.info(f"Found {len(journal_entries)} entries.")

            # Utilise l'instance 'llm' (GeminiLLM) initialisée plus haut
            log.info("Generating tags (using Gemini)...")
            journal_entries = tag_gen.process_entries(journal_entries)
            log.info("Mapping competencies (using Gemini)...")
            journal_entries = comp_mapper.process_entries(journal_entries)

            log.info("Adding journal entries to vector DB (local embeddings)...")
            processed_count, error_count_db = 0, 0
            for entry in journal_entries:
                # Bloc try/except pour l'ajout d'une seule entrée
                try:
                    chunks = chunk_text(entry.raw_text)
                    if chunks:
                        entry_data_for_db = {
                            "entry_id": entry.entry_id, "chunks": chunks, "date_iso": entry.date.isoformat(),
                            "source_file": entry.source_file, "tags_str": ",".join(entry.tags or [])
                        }
                        vector_db.add_entry_chunks(entry_data_for_db)
                        processed_count += 1
                    else: log.warning(f"Entry {entry.entry_id} had no chunks.")
                except Exception as e_add:
                    log.error(f"Failed to add chunks for {entry.entry_id}: {e_add}", exc_info=True)
                    error_count_db += 1
            log.info(f"--- Finished Journal Processing: {processed_count} entries added. {error_count_db} errors. ---")

        elif args.command == "process_guidelines":
            log.info("--- Command: process_guidelines ---")
            pdf_path = args.pdf_path
            if not os.path.exists(pdf_path): log.error(f"Guidelines PDF not found: {pdf_path}"); sys.exit(1)
            if args.reprocess: log.warning("Clearing *reference* collection..."); vector_db.clear_reference_collection()
            log.info(f"Processing PDF: {pdf_path}")
            pdf_text = extract_text_from_pdf(pdf_path)
            if not pdf_text: log.error(f"No text extracted from PDF '{pdf_path}'."); sys.exit(1)
            pdf_chunks = chunk_text(pdf_text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
            if not pdf_chunks: log.error("Failed to chunk PDF text."); sys.exit(1)
            log.info(f"Adding {len(pdf_chunks)} guideline chunks to DB...")
            doc_name = os.path.basename(pdf_path)
            vector_db.add_reference_chunks(doc_name=doc_name, chunks=pdf_chunks) # Le try/except est dans la méthode
            log.info(f"--- Finished Guidelines Processing: Added guidelines from '{doc_name}' ---")

        elif args.command == "create_plan":
            log.info("--- Command: create_plan ---")
            structure_def = None
            if args.requirements_file: log.warning("Loading structure from file NI.")
            report_plan: ReportPlan = planner.create_base_plan(structure_definition=structure_def)
            memory.save_report_plan(report_plan, args.output_plan_file)
            log.info(f"Report plan saved to {args.output_plan_file}")
            print(f"\nPlan saved to: {args.output_plan_file}")

        elif args.command == "generate_report":
            log.info("--- Command: generate_report ---")
            report_plan = memory.load_report_plan(args.plan_file)
            if not report_plan: log.error(f"Plan file not found: {args.plan_file}"); sys.exit(1)
            log.info(f"Starting report generation (LLM: Gemini). Output: {args.output_file}")
            # generator utilise l'instance llm (GeminiLLM)
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
            if report_plan: completed, total, percent = progress_tracker.calculate_progress(report_plan); print(f"\nProgress: {completed}/{total} sections complete ({percent:.1f}%)")
            if gap_issues: print("\n[!] Gaps Found:"); [print(f"  - {i}") for i in gap_issues]
            else: print("\n[*] No major gaps detected.")
            if consistency_issues: print("\n[!] Consistency Issues Found:"); [print(f"  - {i}") for i in consistency_issues]
            else: print("\n[*] No major consistency issues detected.")
            if quality_checker.original_journal_texts: print(f"\n[*] Plagiarism Check:"); print(f"  - Est. Copied: {copied_percentage:.1f}%"); [print(f"    - {i}") for i in plagiarism_issues[:5]]; print(f"      ... ({len(plagiarism_issues)-5} more)" if len(plagiarism_issues)>5 else "")
            else: print("\n[!] Plagiarism check skipped (no journals loaded).")
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
            log.info(f"Visualizations saved in '{config.OUTPUT_DIR}'.")
            print(f"\nVisualizations saved in: {config.OUTPUT_DIR}")

        elif args.command == "manage_refs":
            log.info("--- Command: manage_refs ---")
            if args.ref_command == "add":
                log.info(f"Adding reference: {args.key}")
                try: extra_data = json.loads(args.data); assert isinstance(extra_data, dict)
                except: log.error(f"Invalid JSON in --data: {args.data}", exc_info=True); print("Error: Invalid JSON in --data."); sys.exit(1)
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
            }
            log.info("--- Command: run_agent ---")
            objective = args.objective; max_turns = args.max_turns
            log.info(f"Starting agent: objective='{objective}', max_turns={max_turns}, LLM=Gemini")

            # --- Prompt Système ---
            system_prompt = f"""...(Prompt système identique)...""" # Garder le prompt système détaillé
            conversation_history = [{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Start: {objective}"}]

            # --- Boucle Agentique ---
            for turn in range(max_turns):
                log.info(f"--- Agent Turn {turn + 1}/{max_turns} ---"); print(f"\n--- Turn {turn + 1}/{max_turns} ---")
                # 1. Appel LLM (Gemini)
                llm_response_content = llm._make_request(messages=conversation_history, max_tokens=1500, temperature=0.5)
                if not llm_response_content: log.error("LLM failed."); break
                log.info(f"LLM Raw Response: {llm_response_content[:200]}..."); print(f"\nAgent:\n{llm_response_content}\n")
                conversation_history.append({"role": "assistant", "content": llm_response_content})

            # 2. Détecter et Exécuter Appel d'Outil
            tool_call_prefix = ">>>TOOL_CALL"
            tool_call_suffix = ">>>END_TOOL_CALL"
            tool_result = None
            tool_executed = False # Flag pour savoir si un outil a été appelé

            if tool_call_prefix in llm_response_content:
                try:
                    call_start = llm_response_content.find(tool_call_prefix) + len(tool_call_prefix)
                    call_end = llm_response_content.find(tool_call_suffix, call_start)

                    if call_end != -1:
                        tool_call_str = llm_response_content[call_start:call_end].strip()
                        log.info(f"Detected Tool Call: {tool_call_str}")
                        tool_executed = True # Marquer qu'un appel a été détecté

                        # Parser Nom de l'Outil
                        tool_name = ""
                        if "(" in tool_call_str:
                             tool_name = tool_call_str.split("(", 1)[0].strip()
                        else:
                             # Gérer le cas où le LLM oublie les parenthèses ?
                             log.warning(f"Potential tool call format issue: Missing '('. Assuming '{tool_call_str}' is the tool name.")
                             tool_name = tool_call_str # Tenter avec la chaîne entière comme nom

                        # Vérifier si l'outil est connu
                        if tool_name in AVAILABLE_TOOLS:
                            # Parser les Arguments (fonction interne simple)
                            def parse_kwargs(tool_call_str):
                                kwargs = {}
                                try:
                                    # Extract the argument string between parentheses
                                    start = tool_call_str.find('(')
                                    end = tool_call_str.rfind(')')
                                    if start == -1 or end == -1:
                                        return {}  # No parentheses, no arguments
                                    args_str = tool_call_str[start + 1:end].strip()
                                    if not args_str:
                                        return {}  # Empty argument string

                                    # Parse key-value pairs
                                    parts = args_str.split(',')
                                    for part in parts:
                                        part = part.strip()
                                        if not part: continue  # Skip empty parts
                                        if '=' not in part:
                                            raise ValueError(f"Invalid argument format: '{part}' (missing '=').")

                                        key, value_str = part.split('=', 1)
                                        key = key.strip()
                                        value_str = value_str.strip()

                                        # Attempt to parse the value using ast.literal_eval for various types
                                        try:
                                            value = ast.literal_eval(value_str)
                                        except (ValueError, SyntaxError):
                                            # If not a literal, treat it as a string (ensure quotes if needed)
                                            if not (value_str.startswith("'") and value_str.endswith("'")) and \
                                                not (value_str.startswith('"') and value_str.endswith('"')):
                                                value = value_str.strip("'\"")  # Remove potential quotes
                                            else:
                                                value = value_str  # Keep as is if already quoted

                                        kwargs[key] = value
                                except (ValueError, SyntaxError) as e_parse:
                                    log.error(f"Error parsing arguments for tool call '{tool_call_str}': {e_parse}", exc_info=True)
                                    raise ValueError(f"Could not parse tool call arguments: {e_parse}") from e_parse
                                return kwargs

                            try:
                                tool_args = parse_kwargs(tool_call_str)
                                log.info(f"Parsed Tool Args: {tool_args}")

                                # Exécuter l'outil
                                tool_function = AVAILABLE_TOOLS[tool_name]
                                tool_result = tool_function(**tool_args)
                                log.info(f"Tool '{tool_name}' executed. Result length: {len(tool_result or '')}")

                            except Exception as e_parse_exec:
                                log.error(f"Failed to parse arguments or execute tool '{tool_call_str}': {e_parse_exec}", exc_info=True)
                                tool_result = f"Error: Could not parse arguments or execute tool {tool_name}. Details: {e_parse_exec}"

                        else:
                            log.warning(f"LLM tried to call unknown tool: '{tool_name}'")
                            tool_result = f"Error: Tool '{tool_name}' is not available."
                    else:
                        log.warning("Detected TOOL_CALL prefix but no valid suffix or format in LLM response.")
                        tool_result = "Error: Invalid tool call format received."

                except Exception as e_tool_outer:
                    log.error(f"Error processing potential tool call block: {e_tool_outer}", exc_info=True)
                    tool_result = f"Error interpreting or executing tool call: {e_tool_outer}"

                # 3. Ajouter le Résultat de l'Outil à l'Historique (si un appel a été tenté)
                tool_result_for_llm = tool_result if tool_result is not None else "Error: Tool execution failed without specific result."
                tool_result_message = f">>>TOOL_RESULT\n{tool_result_for_llm}\n>>>END_TOOL_RESULT"
                # On ajoute toujours un message de résultat si un appel a été détecté, même en cas d'erreur
                conversation_history.append({"role": "user", "content": tool_result_message})
                print(f"\nTool Result Provided to Agent:\n{tool_result_message}\n")


            # 4. Condition d'Arrêt
            # Si l'agent n'a PAS appelé d'outil et semble répondre à l'objectif final
            if not tool_executed:
                 # Heuristique simple: si la réponse contient des mots clés indiquant la fin
                 if "objective completed" in llm_response_content.lower() or \
                    "task finished" in llm_response_content.lower() or \
                    "here are the guidelines" in llm_response_content.lower() and "Find guidelines" in objective : # Adapter la condition
                     log.info("Agent response suggests objective might be completed. Ending loop.")
                     print("\nAgent response suggests objective completion.")
                     break

            if turn == max_turns - 1:
                log.info("Maximum turns reached. Ending agent loop.")
                print("\nMaximum turns reached.")
                break
        # Fin de la boucle for turn

    # Gérer le cas où aucune commande n'est reconnue (ne devrait pas arriver avec required=True)
    else :
        log.error(f"Unknown command received: {args.command}")
        parser.print_help() # Afficher l'aide
        sys.exit(1)


# --- Point d'Entrée Principal ---
if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        # Gérer sys.exit() proprement
        exit_code = e.code if isinstance(e.code, int) else 1
        if exit_code != 0: log.warning(f"Script exited with code {exit_code}.")
        # else: log.info("Script finished successfully.") # Peut être redondant si loggé avant exit(0)
        sys.exit(exit_code)
    except KeyboardInterrupt:
        log.warning("Script interrupted by user (Ctrl+C).")
        print("\nOperation cancelled by user.")
        sys.exit(130)
    except Exception as e_global:
        # Attraper toute autre exception non prévue au niveau le plus haut
        log.critical(f"An unexpected critical error occurred outside the main function block: {e_global}", exc_info=True)
        print(f"\nCRITICAL ERROR: {e_global}. Check logs.")
        sys.exit(1)