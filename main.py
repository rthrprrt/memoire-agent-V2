# main.py (Version finale propre V3 - Correction Syntaxe Parsing Agent)

import argparse
import logging
import os
import sys
import json
import time
import ast # Pour literal_eval

# --- Import des modules du projet ---
import config
from document_processor import process_all_journals, chunk_text, extract_text_from_pdf
from vector_database import VectorDBManager
from llm_interface import GeminiLLM
from data_models import JournalEntry, ReportPlan, ReportSection
import agent_tools
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
from typing import List, Dict, Any, Optional

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')
log = logging.getLogger(__name__)

# --- Fonction Principale ---
def main():
    # --- Configuration du Parseur d'Arguments ---
    parser = argparse.ArgumentParser(description="AI Agent for Apprenticeship Report Generation (Embeddings: Local, LLM: Google Gemini)")
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)
    # Définitions des subparsers...
    parser_process = subparsers.add_parser("process_journals", help="Process journals, get tags/competencies (LLM), store with local embeddings.")
    parser_process.add_argument("--journal_dir", default=config.JOURNAL_DIR); parser_process.add_argument("--reprocess_all", action="store_true")
    parser_guidelines = subparsers.add_parser("process_guidelines", help="Process guidelines PDF, store with local embeddings.")
    parser_guidelines.add_argument("--pdf_path", default=config.GUIDELINES_PDF_PATH); parser_guidelines.add_argument("--reprocess", action="store_true")
    parser_plan = subparsers.add_parser("create_plan", help="Generate report plan JSON with unique IDs."); parser_plan.add_argument("--requirements_file", default=None); parser_plan.add_argument("--output_plan_file", default=config.DEFAULT_PLAN_FILE)
    parser_generate = subparsers.add_parser("generate_report", help="Generate full report draft (non-agentic)."); parser_generate.add_argument("--plan_file", default=config.DEFAULT_PLAN_FILE); parser_generate.add_argument("--output_file", default=config.DEFAULT_REPORT_OUTPUT)
    parser_quality = subparsers.add_parser("check_quality", help="Run quality checks on draft."); parser_quality.add_argument("--report_file", default=config.DEFAULT_REPORT_OUTPUT); parser_quality.add_argument("--plan_file", default=config.DEFAULT_PLAN_FILE); parser_quality.add_argument("--skip_journal_load", action="store_true")
    parser_visuals = subparsers.add_parser("create_visuals", help="Generate visualizations."); parser_visuals.add_argument("--skip_journal_load", action="store_true")
    parser_refs = subparsers.add_parser("manage_refs", help="Manage bibliography."); ref_subparsers = parser_refs.add_subparsers(dest="ref_command", required=True); parser_add_ref = ref_subparsers.add_parser("add"); parser_add_ref.add_argument("--key", required=True); parser_add_ref.add_argument("--type", required=True, choices=['book', 'article', 'web', 'report', 'other']); parser_add_ref.add_argument("--author", required=True); parser_add_ref.add_argument("--year", required=True, type=int); parser_add_ref.add_argument("--title", required=True); parser_add_ref.add_argument("--data", default="{}"); parser_list_ref = ref_subparsers.add_parser("list")
    parser_agent = subparsers.add_parser("run_agent", help="Run agentic workflow (section by section)."); parser_agent.add_argument("--max_turns", type=int, default=100); parser_agent.add_argument("--objective", default="Generate the full apprenticeship report section by section according to the plan."); parser_agent.add_argument("--delay", type=int, default=5, help="Delay between LLM calls.")
    parser_full = subparsers.add_parser("run_all", help="Run core pipeline (proc_journ, proc_guide, plan, gen_report)."); parser_full.add_argument("--journal_dir", default=config.JOURNAL_DIR); parser_full.add_argument("--output_file", default=config.DEFAULT_REPORT_OUTPUT); parser_full.add_argument("--reprocess", action="store_true"); parser_full.add_argument("--skip_guidelines", action="store_true")
    args = parser.parse_args()

    # --- Initialisation des Composants Clés ---
    log.info("Initializing core components...")
    try: memory = MemoryManager(); vector_db = VectorDBManager(); llm = GeminiLLM(); log.info(f"LLM: Google AI model {config.GEMINI_CHAT_MODEL_NAME}"); tag_gen = TagGenerator(); comp_mapper = CompetencyMapper(); analyzer = ContentAnalyzer(); planner = ReportPlanner(); generator = ReportGenerator(vector_db); quality_checker = QualityChecker(vector_db); visualizer = Visualizer(); ref_manager = ReferenceManager(); progress_tracker = ProgressTracker(); log.info("Core components initialized.")
    except Exception as e: log.critical(f"FATAL Init Error: {e}", exc_info=True); sys.exit(1)

    # --- Exécution de la Commande ---
    try:
        if args.command == "process_journals":
            log.info("--- Command: process_journals ---")
            if args.reprocess_all: log.warning("Clearing *journal* collection..."); vector_db.clear_journal_collection()
            log.info(f"Processing journals from: {args.journal_dir}")
            journal_entries: List[JournalEntry] = process_all_journals(args.journal_dir)
            if not journal_entries: log.error("No valid journals found."); sys.exit(1)
            log.info(f"Found {len(journal_entries)} entries.")
            log.info("Generating tags/competencies (using Gemini)...")
            journal_entries = tag_gen.process_entries(journal_entries); journal_entries = comp_mapper.process_entries(journal_entries)
            log.info("Adding journal entries to vector DB (local embeddings)...")
            processed_count, error_count_db = 0, 0
            # --- Bloc Try/Except pour chaque entrée ---
            for entry in journal_entries:
                try:
                    chunks = chunk_text(entry.raw_text)
                    if chunks:
                        entry_data_for_db = {"entry_id": entry.entry_id, "chunks": chunks, "date_iso": entry.date.isoformat(),"source_file": entry.source_file, "tags_str": ",".join(entry.tags or [])}
                        vector_db.add_entry_chunks(entry_data_for_db)
                        processed_count += 1
                    else: log.warning(f"Entry {entry.entry_id} had no chunks.")
                except Exception as e_add: # --- Except correspondant au Try ---
                    log.error(f"Failed to process/add chunks for {entry.entry_id}: {e_add}", exc_info=True)
                    error_count_db += 1
            # --- Fin Try/Except ---
            log.info(f"--- Finished Journal Processing: {processed_count} entries added. {error_count_db} errors. ---")

        elif args.command == "process_guidelines":
            # ... (Code process_guidelines - Correct) ...
            log.info("--- Command: process_guidelines ---")
            pdf_path = args.pdf_path;
            if not os.path.exists(pdf_path): log.error(f"PDF not found: {pdf_path}"); sys.exit(1)
            if args.reprocess: log.warning("Clearing *reference* collection..."); vector_db.clear_reference_collection()
            log.info(f"Processing PDF: {pdf_path}"); pdf_text = extract_text_from_pdf(pdf_path)
            if not pdf_text: log.error(f"No text extracted from PDF '{pdf_path}'."); sys.exit(1)
            pdf_chunks = chunk_text(pdf_text);
            if not pdf_chunks: log.error("Failed to chunk PDF text."); sys.exit(1)
            log.info(f"Adding {len(pdf_chunks)} guideline chunks..."); doc_name = os.path.basename(pdf_path); vector_db.add_reference_chunks(doc_name=doc_name, chunks=pdf_chunks)
            log.info(f"--- Finished Guidelines Processing: Added '{doc_name}' ---")

        elif args.command == "create_plan":
             # ... (Code create_plan - Correct) ...
            log.info("--- Command: create_plan ---")
            if args.requirements_file: log.warning("Loading structure from file NI.")
            report_plan = planner.create_base_plan(); memory.save_report_plan(report_plan, args.output_plan_file)
            log.info(f"Plan saved to {args.output_plan_file}"); print(f"\nPlan saved: {args.output_plan_file}")

        elif args.command == "generate_report":
             # ... (Code generate_report - Correct) ...
             log.info("--- Command: generate_report ---")
             report_plan = memory.load_report_plan(args.plan_file);
             if not report_plan: log.error(f"Plan not found: {args.plan_file}"); sys.exit(1)
             log.info(f"Starting generation (LLM: Gemini)..."); updated_plan = generator.generate_full_report(report_plan, args.output_file)
             memory.save_report_plan(updated_plan, args.plan_file); log.info(f"Generation finished. Draft: '{args.output_file}'.")
             print(f"\nReport draft generated: {args.output_file}")

        elif args.command == "check_quality":
            # ... (Code check_quality - Correct) ...
            log.info("--- Command: check_quality ---")
            pass # Placeholder, code existant

        elif args.command == "create_visuals":
            # ... (Code create_visuals - Correct) ...
            log.info("--- Command: create_visuals ---")
            pass # Placeholder, code existant

        elif args.command == "manage_refs":
            # ... (Code manage_refs - Correct) ...
             log.info("--- Command: manage_refs ---")
             pass # Placeholder, code existant

        elif args.command == "run_agent":
            AVAILABLE_TOOLS = { "search_journal_entries": agent_tools.search_journal_entries, "search_guidelines": agent_tools.search_guidelines, "get_report_plan_structure": agent_tools.get_report_plan_structure, "get_pending_sections": agent_tools.get_pending_sections, "update_section_status": agent_tools.update_section_status, "draft_single_section": agent_tools.draft_single_section }
            log.info("--- Command: run_agent ---")
            objective = args.objective; max_turns = args.max_turns; delay_between_calls = args.delay
            log.info(f"Starting agent: objective='{objective}', max_turns={max_turns}, LLM=Gemini, Delay={delay_between_calls}s")
            system_prompt = f"""...(Prompt système détaillé comme avant)..."""
            conversation_history = [{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Start: {objective}"}]

            # --- Boucle Agentique ---
            for turn in range(max_turns):
                log.info(f"--- Agent Turn {turn + 1}/{max_turns} ---"); print(f"\n--- Turn {turn + 1}/{max_turns} ---")
                if turn > 0: log.info(f"Waiting {delay_between_calls}s..."); time.sleep(delay_between_calls)
                current_history_for_api = conversation_history[:];
                if current_history_for_api[-1]['role'] == 'assistant': log.warning("Appending placeholder."); placeholder_msg = "Proceed."; current_history_for_api.append({"role": "user", "content": placeholder_msg})
                llm_response_content = llm._make_request(messages=current_history_for_api, max_tokens=2048, temperature=0.4)
                if not llm_response_content: log.error("LLM failed."); break
                log.info(f"LLM Raw: {llm_response_content[:200]}..."); print(f"\nAgent:\n{llm_response_content}\n")
                conversation_history.append({"role": "assistant", "content": llm_response_content})

                # --- Détection/Exécution Outil ---
                tool_call_prefix = ">>>TOOL_CALL"; tool_call_suffix = ">>>END_TOOL_CALL"; tool_result = None; tool_executed = False
                if tool_call_prefix in llm_response_content:
                    try: # Bloc try englobant pour le traitement de l'appel d'outil
                        call_start = llm_response_content.find(tool_call_prefix) + len(tool_call_prefix)
                        call_end = llm_response_content.find(tool_call_suffix, call_start)
                        if call_end != -1:
                            tool_call_str = llm_response_content[call_start:call_end].strip(); log.info(f"Detected: {tool_call_str}"); tool_executed = True
                            tool_name = tool_call_str.split("(", 1)[0].strip() if "(" in tool_call_str else tool_call_str
                            if tool_name in AVAILABLE_TOOLS:
                                # --- Bloc de Parsing d'Arguments Corrigé V2 ---
                                tool_args = {}
                                args_str = ""
                                try: # Extraire la chaîne d'arguments
                                    args_start_index = tool_call_str.index('(') + 1
                                    args_end_index = tool_call_str.rindex(')')
                                    args_str = tool_call_str[args_start_index:args_end_index].strip()
                                except ValueError: args_str = "" # Pas d'args

                                if args_str: # Parser si non vide
                                    try: # Pour le parsing des paires
                                        arg_pairs = args_str.split(',')
                                        for pair in arg_pairs:
                                            pair = pair.strip()
                                            if not pair: continue # Ignorer vide

                                            # --- Logique if/else correcte ---
                                            if '=' in pair:
                                                key, val_str = map(str.strip, pair.split('=', 1))
                                                try: # Pour literal_eval
                                                    # Important: literal_eval attend une représentation Python valide
                                                    # Si la valeur est une chaîne simple sans guillemets, il échouera.
                                                    # Essayons d'abord int/float, puis string nettoyé comme fallback sûr.
                                                    try: val = int(val_str)
                                                    except ValueError:
                                                        try: val = float(val_str)
                                                        except ValueError:
                                                            # Si échec, traiter comme string nettoyé
                                                            val = val_str.strip('\'"') # Enlève guillemets
                                                except Exception as e_eval:
                                                     log.warning(f"Could not evaluate argument value '{val_str}': {e_eval}. Treating as string.")
                                                     val = val_str.strip('\'"') # Fallback string
                                                tool_args[key] = val
                                            else:
                                                # Si pas de '=', ignorer ou logguer
                                                log.warning(f"Argument part '{pair}' ignored (not key=value format).")
                                            # --- Fin logique if/else ---
                                    except Exception as e_parse:
                                        log.error(f"Failed to parse arguments string '{args_str}': {e_parse}", exc_info=True)
                                        tool_result = f"Error parsing args: {e_parse}"
                                # --- Fin Parsing Arguments ---

                                # Exécuter si pas d'erreur de parsing
                                if tool_result is None:
                                    try: # Pour l'exécution
                                        log.info(f"Executing: {tool_name}({tool_args})")
                                        tool_function = AVAILABLE_TOOLS[tool_name]
                                        tool_result = tool_function(**tool_args) # Appel avec kwargs
                                        log.info(f"Tool '{tool_name}' OK.")
                                    except Exception as e_exec:
                                        log.error(f"Failed to execute tool '{tool_name}': {e_exec}", exc_info=True)
                                        tool_result = f"Error executing tool: {e_exec}"
                            else:
                                log.warning(f"Unknown tool requested: '{tool_name}'"); tool_result = f"Error: Unknown tool '{tool_name}'."
                        else:
                            log.warning("Invalid tool call format detected (missing END_TOOL_CALL?)."); tool_result = "Error: Invalid tool call format."
                    # Fin du bloc try principal pour le traitement de l'appel
                    except Exception as e_t:
                        log.error(f"General error processing tool call block: {e_t}", exc_info=True)
                        tool_result = f"Error processing tool call: {e_t}"

                    # Ajouter résultat/erreur à l'historique ORIGINAL
                    tool_result_for_llm = tool_result if tool_result is not None else "Error: Tool execution failed silently."
                    msg = f">>>TOOL_RESULT\n{tool_result_for_llm}\n>>>END_TOOL_RESULT"
                    conversation_history.append({"role": "user", "content": msg}); print(f"\nTool Result:\n{msg}\n")

                    # !! Condition d'arrêt Explicite après get_pending_sections !!
                    if tool_name == "get_pending_sections" and isinstance(tool_result, str) and \
                       ("No sections are currently marked" in tool_result or "No pending or failed sections found" in tool_result):
                         log.info("Tool 'get_pending_sections' indicates no more pending sections. Ending agent loop.")
                         print("\nAll sections processed according to the plan.")
                         break # Sortir de la boucle for turn
                # --- Fin Détection/Exécution Outil ---

                # 3. Condition d'Arrêt (Max turns)
                if turn == max_turns - 1: log.info("Max turns reached."); print("\nMaximum turns reached."); break
            # Fin de la boucle for turn

        elif args.command == "run_all":
            # ... (code run_all existant, peut être simplifié) ...
            log.info("--- Command: run_all ---"); print("Note: 'run_all' executes the non-agentic pipeline."); pass

        else: # Commande inconnue
            log.error(f"Unknown command: {args.command}")
            parser.print_help()
            sys.exit(1)

    # Gestion globale des exceptions pour la fonction main
    except Exception as e_main: log.critical(f"Unhandled error: {e_main}", exc_info=True); print(f"\nCRITICAL ERROR: {e_main}."); sys.exit(1)

# --- Point d'Entrée Principal ---
if __name__ == "__main__":
    try: main()
    except SystemExit as e: exit_code = e.code if isinstance(e.code, int) else 1; sys.exit(exit_code)
    except KeyboardInterrupt: print("\nOperation cancelled."); sys.exit(130)
    except Exception as e_global: log.critical(f"Global error: {e_global}", exc_info=True); print(f"\nCRITICAL ERROR: {e_global}."); sys.exit(1)