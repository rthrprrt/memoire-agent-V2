# main.py (Version adaptée pour embeddings locaux)

import argparse
import logging
import os
import sys
import json # Importé pour la gestion des références

# Import necessary modules from the project
import config
from document_processor import process_all_journals, chunk_text, extract_text_from_pdf
from vector_database import VectorDBManager
from llm_interface import DeepSeekLLM
from tag_generator import TagGenerator
from competency_mapper import CompetencyMapper
from content_analyzer import ContentAnalyzer
from report_planner import ReportPlanner
from report_generator import ReportGenerator
from quality_checker import QualityChecker
from visualization import Visualizer
from reference_manager import ReferenceManager
from progress_tracker import ProgressTracker
from memory_manager import MemoryManager # To hold state during execution
# Import seulement ce qui est nécessaire pour le type hinting
from typing import List
from data_models import JournalEntry, ReportPlan
import agent_tools

# Configure logging to include module names for better tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')

# Logger spécifique pour main.py
log = logging.getLogger(__name__) # Utiliser __name__ est une bonne pratique


def main():
    parser = argparse.ArgumentParser(description="AI Agent for Apprenticeship Report Generation")
    # Rendre la commande obligatoire pour éviter les erreurs si on lance juste 'python main.py'
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

    # --- Subparser for Processing Journals ---
    parser_process = subparsers.add_parser("process_journals",
                                         help="Process journal DOCX files, generate tags, map competencies, "
                                              "and store in the 'journal_entries' Vector DB collection.")
    parser_process.add_argument("--journal_dir", default=config.JOURNAL_DIR,
                                help="Directory containing journal DOCX files (expected format: YYYY-MM-DD.docx).")
    parser_process.add_argument("--reprocess_all", action="store_true",
                                help="Clear existing *journal* vector DB collection before processing.")
    parser_guidelines = subparsers.add_parser("process_guidelines",
                                            help="Process the guidelines PDF specified in config.py and store "
                                                 "it in the 'reference_docs' Vector DB collection.")
    parser_guidelines.add_argument("--pdf_path", default=config.GUIDELINES_PDF_PATH,
                                   help="Path to the guidelines PDF file.")
    parser_guidelines.add_argument("--reprocess", action="store_true",
                                   help="Clear existing *reference* vector DB collection before processing.")
    parser_agent = subparsers.add_parser("run_agent",
                                       help="Run the agent interactively to work on the report.")
    parser_agent.add_argument("--max_turns", type=int, default=10,
                              help="Maximum number of interaction turns with the LLM.")
    parser_agent.add_argument("--objective", default="Draft the 'Introduction' section of the report.",
                              help="Initial objective for the agent.")
    
    # --- Subparser for Creating Report Plan ---
    parser_plan = subparsers.add_parser("create_plan", help="Generate the report structure/plan.")
    parser_plan.add_argument("--requirements_file", default=None,
                             help="(Optional) Path to a file defining report structure (overrides config).")
    parser_plan.add_argument("--output_plan_file", default=config.DEFAULT_PLAN_FILE,
                             help="File path to save the generated plan JSON.")

    # --- Subparser for Generating Report ---
    parser_generate = subparsers.add_parser("generate_report",
                                          help="Generate the draft report content based on the plan using DeepSeek API.")
    parser_generate.add_argument("--plan_file", default=config.DEFAULT_PLAN_FILE,
                                 help="Path to the report plan JSON file.")
    parser_generate.add_argument("--output_file", default=config.DEFAULT_REPORT_OUTPUT,
                                 help="Path to save the generated DOCX report draft.")

    # --- Subparser for Quality Check ---
    parser_quality = subparsers.add_parser("check_quality", help="Run quality checks on a generated report draft.")
    parser_quality.add_argument("--report_file", default=config.DEFAULT_REPORT_OUTPUT,
                                help="Path to the report DOCX file to check.")
    parser_quality.add_argument("--plan_file", default=config.DEFAULT_PLAN_FILE,
                                help="Path to the report plan JSON (needed for consistency/gap checks).")
    parser_quality.add_argument("--skip_journal_load", action="store_true",
                                help="Skip reloading journal texts (faster if check follows generation, but less accurate for plagiarism).")

    # --- Subparser for Visualizations ---
    parser_visuals = subparsers.add_parser("create_visuals",
                                         help="Generate visualizations (timeline, etc.). Requires processed journals data.")
    parser_visuals.add_argument("--skip_journal_load", action="store_true",
                                help="Skip reloading/reprocessing journals (assumes data is available).")


    # --- Subparser for Reference Management ---
    parser_refs = subparsers.add_parser("manage_refs", help="Manage bibliography references (stored in references.json).")
    ref_subparsers = parser_refs.add_subparsers(dest="ref_command", help="Reference actions", required=True)
    parser_add_ref = ref_subparsers.add_parser("add", help="Add a new reference.")
    parser_add_ref.add_argument("--key", required=True, help="Unique citation key (e.g., Smith2023).")
    parser_add_ref.add_argument("--type", required=True, choices=['book', 'article', 'web', 'report', 'other'],
                                help="Type of reference.")
    parser_add_ref.add_argument("--author", required=True, help="Author(s), e.g., 'Smith, J. and Doe, A.'")
    parser_add_ref.add_argument("--year", required=True, type=int, help="Year of publication.")
    parser_add_ref.add_argument("--title", required=True, help="Title of the work.")
    # Utiliser --data pour les champs spécifiques au type
    parser_add_ref.add_argument("--data", default="{}",
                                help="JSON string with additional type-specific fields. "
                                     "Example for book: '{\"publisher\": \"PubCo\", \"place\": \"London\"}'. "
                                     "Example for web: '{\"url\": \"http://...\", \"accessed\": \"YYYY-MM-DD\", \"website_name\": \"Site Name\"}'. "
                                     "Example for article: '{\"journal\": \"Journal Name\", \"volume\": \"10\", \"issue\": \"2\", \"pages\": \"11-20\"}'.")

    parser_list_ref = ref_subparsers.add_parser("list", help="List all stored references.")


    # --- Subparser for Full Workflow (Convenience) ---
    parser_full = subparsers.add_parser("run_all", help="Run the full workflow: process, plan, generate, check, visualize.")
    parser_full.add_argument("--journal_dir", default=config.JOURNAL_DIR, help="Directory with journals.")
    parser_full.add_argument("--output_file", default=config.DEFAULT_REPORT_OUTPUT, help="Final report output file.")
    parser_full.add_argument("--reprocess", action="store_true", help="Force reprocessing of journals (clears DB).")


    args = parser.parse_args()

    # --- Initialize Core Components ---
    log.info("Initializing core components...")
    try:
        memory = MemoryManager()
        vector_db = VectorDBManager()
        tag_gen = TagGenerator()
        comp_mapper = CompetencyMapper()
        analyzer = ContentAnalyzer()
        planner = ReportPlanner()
        generator = ReportGenerator(vector_db)
        quality_checker = QualityChecker(vector_db)
        visualizer = Visualizer()
        ref_manager = ReferenceManager()
        progress_tracker = ProgressTracker()
        llm = DeepSeekLLM()
        log.info("Core components initialized successfully.")
    except Exception as e:
        log.error(f"FATAL: Failed to initialize core components: {e}", exc_info=True)
        sys.exit(1)

    AVAILABLE_TOOLS = {
        "search_journal_entries": agent_tools.search_journal_entries,
        "search_guidelines": agent_tools.search_guidelines,
        # Ajoutez d'autres outils ici si définis dans agent_tools.py
    }

    # --- Command Execution ---

    if args.command == "process_journals":
        log.info("--- Command: process_journals ---")
        if args.reprocess_all:
            log.warning("Option --reprocess_all selected. Clearing existing vector database collection...")
            try:
                vector_db.clear_collection()
            except Exception as e_clear:
                log.error(f"Failed to clear vector DB collection: {e_clear}", exc_info=True)
                # Décider si on continue ou on arrête ? Pour l'instant, on continue mais on log l'erreur.

        # 1. Lire les fichiers DOCX
        log.info(f"Processing journals from directory: {args.journal_dir}")
        journal_entries: List[JournalEntry] = process_all_journals(args.journal_dir)
        if not journal_entries:
            log.error("No valid journal entries found or processed. Please check the 'journals' directory and filename format (YYYY-MM-DD.docx). Exiting.")
            sys.exit(1)
        log.info(f"Found {len(journal_entries)} journal entries to process.")

        # 2. Générer les Tags (Appels API DeepSeek)
        log.info("Generating tags for entries...")
        journal_entries = tag_gen.process_entries(journal_entries)

        # 3. Mapper les Compétences (Appels API DeepSeek)
        log.info("Mapping competencies for entries...")
        journal_entries = comp_mapper.process_entries(journal_entries)

        # 4. Découper en Chunks et Ajouter à la Base Vectorielle (Embeddings Locaux)
        log.info("Chunking text and adding entries to vector database (using local embeddings)...")
        processed_count = 0
        error_count_db = 0
        for entry in journal_entries:
            try:
                chunks = chunk_text(entry.raw_text)
                if chunks:
                    # Préparer le dictionnaire pour add_entry_chunks
                    entry_data_for_db = {
                        "entry_id": entry.entry_id,
                        "chunks": chunks,
                        "date_iso": entry.date.isoformat(),
                        "source_file": entry.source_file,
                        "tags_str": ",".join(entry.tags) if entry.tags else ""
                    }
                    vector_db.add_entry_chunks(entry_data_for_db)
                    processed_count += 1
                else:
                    log.warning(f"Entry {entry.entry_id} produced no text chunks. Skipping vector DB add.")
            except Exception as e_add:
                log.error(f"Failed to add chunks for entry {entry.entry_id} to vector DB: {e_add}", exc_info=True)
                error_count_db += 1

        log.info(f"--- Finished Journal Processing ---")
        log.info(f"Entries added/updated in DB: {processed_count}")
        if error_count_db > 0:
             log.warning(f"Encountered {error_count_db} errors during vector DB addition.")

    elif args.command == "process_guidelines":
        log.info("--- Command: process_guidelines ---")
        pdf_path = args.pdf_path
        if not os.path.exists(pdf_path):
             log.error(f"Guidelines PDF file not found at path: {pdf_path}")
             log.error("Please check the path in config.py (GUIDELINES_PDF_PATH) or provide correct path with --pdf_path.")
             sys.exit(1)

        if args.reprocess:
             log.warning("Option --reprocess selected. Clearing existing *reference* vector database collection...")
             vector_db.clear_reference_collection() # Appelle la méthode spécifique

        log.info(f"Processing guidelines PDF: {pdf_path}")
        pdf_text = extract_text_from_pdf(pdf_path)

        if not pdf_text:
             log.error(f"Could not extract text from PDF '{pdf_path}'. Cannot process guidelines.")
             sys.exit(1)

        log.info("Chunking PDF text...")
        pdf_chunks = chunk_text(pdf_text, chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP) # Utiliser les mêmes params de chunking ? Adaptable.

        if not pdf_chunks:
             log.error("Failed to create chunks from the PDF text.")
             sys.exit(1)

        log.info(f"Adding {len(pdf_chunks)} guideline chunks to vector database...")
        try:
             doc_name = os.path.basename(pdf_path)
             vector_db.add_reference_chunks(doc_name=doc_name, chunks=pdf_chunks)
             log.info(f"Successfully added guidelines from '{doc_name}' to the reference collection.")
        except Exception as e_add_ref:
             log.error(f"Failed to add reference chunks to vector DB: {e_add_ref}", exc_info=True)
             sys.exit(1)

        log.info("--- Finished Guidelines Processing ---")

    elif args.command == "create_plan":
        log.info("--- Command: create_plan ---")
        structure_def = None
        if args.requirements_file:
            log.warning("Loading structure from requirements file is not yet implemented. Using default.")
            # TODO: Implémenter la lecture depuis args.requirements_file
        else:
            log.info("Using default report structure from config.")

        try:
            report_plan: ReportPlan = planner.create_base_plan(structure_definition=structure_def)
            memory.save_report_plan(report_plan, args.output_plan_file)
            log.info(f"Report plan saved successfully to {args.output_plan_file}")
            print(f"\nReport plan generated and saved to: {args.output_plan_file}")
            print("You can review and optionally edit this JSON file before generating the report.")
        except Exception as e_plan:
            log.error(f"Failed to create or save the report plan: {e_plan}", exc_info=True)
            sys.exit(1)


    elif args.command == "generate_report":
        log.info("--- Command: generate_report ---")
        log.info(f"Loading report plan from: {args.plan_file}")
        report_plan = memory.load_report_plan(args.plan_file)
        if not report_plan:
            log.error(f"Could not load report plan from {args.plan_file}. Cannot generate report. "
                      "Ensure you have run 'create_plan' first.")
            sys.exit(1)

        log.info(f"Starting report content generation. Output will be saved to: {args.output_file}")
        log.info("This process involves multiple API calls to DeepSeek and may take some time...")
        try:
            # La génération modifie l'objet report_plan en place (ajoute le contenu, change les statuts)
            updated_plan = generator.generate_full_report(report_plan, args.output_file)

            # Sauvegarder le plan mis à jour qui contient maintenant le contenu généré et les statuts
            memory.save_report_plan(updated_plan, args.plan_file)
            log.info(f"Report generation finished. Draft saved to '{args.output_file}'. "
                     f"Updated plan (with content) saved back to '{args.plan_file}'.")
            print(f"\nReport draft generated and saved to: {args.output_file}")
        except Exception as e_gen:
             log.error(f"An error occurred during report generation: {e_gen}", exc_info=True)
             # Sauvegarder l'état actuel du plan même en cas d'erreur partielle ?
             if report_plan: memory.save_report_plan(report_plan, args.plan_file + ".error")
             log.info(f"Attempted to save partial plan state to {args.plan_file}.error")
             sys.exit(1)


    elif args.command == "check_quality":
        log.info("--- Command: check_quality ---")
        log.info(f"Loading report plan from: {args.plan_file}")
        report_plan = memory.load_report_plan(args.plan_file)
        if not report_plan:
            log.warning(f"Could not load report plan from {args.plan_file}. "
                        "Consistency and gap checks based on plan structure will be skipped.")

        if not os.path.exists(args.report_file):
             log.error(f"Report file '{args.report_file}' not found. Cannot perform quality checks.")
             sys.exit(1)
        log.info(f"Checking quality of report file: {args.report_file}")

        # Chargement des textes originaux pour la comparaison
        if not args.skip_journal_load:
            log.info("Loading original journal texts for plagiarism comparison...")
            temp_journal_entries = process_all_journals(config.JOURNAL_DIR) # Recharger pour avoir les textes
            if temp_journal_entries:
                quality_checker.load_journal_texts(temp_journal_entries)
            else:
                log.warning("Could not load journal texts. Plagiarism check might be inaccurate.")
        else:
            log.info("Skipping journal text loading as per --skip_journal_load.")


        # Exécution des différentes vérifications
        gap_issues = []
        consistency_issues = []
        log.info("Running quality checks (this may involve API calls)...")
        try:
            if report_plan:
                log.info("Checking for content gaps based on plan...")
                gap_issues = quality_checker.identify_content_gaps(report_plan)
                log.info("Checking for consistency across sections...")
                consistency_issues = quality_checker.check_consistency_across_sections(report_plan)
            else:
                log.warning("Skipping content gap and consistency checks as report plan was not loaded.")

            log.info("Checking for potential over-copying from journals...")
            plagiarism_issues, copied_percentage = quality_checker.check_plagiarism_against_journals(args.report_file)

            # Affichage des résultats de manière structurée
            print("\n--- Quality Check Results ---")
            print(f"Report File: {args.report_file}")

            if report_plan:
                completed, total, percent = progress_tracker.calculate_progress(report_plan)
                print(f"\nReport Progress: {completed}/{total} sections marked as complete ({percent:.1f}%)")

                if gap_issues:
                    print("\n[!] Potential Content Gaps Found:")
                    for issue in gap_issues: print(f"  - {issue}")
                else: print("\n[*] No major content gaps detected based on plan.")

                if consistency_issues:
                    print("\n[!] Potential Consistency Issues Found:")
                    for issue in consistency_issues: print(f"  - {issue}")
                else: print("\n[*] No major consistency issues detected between checked sections.")
            else:
                 print("\n[!] Report plan not loaded, skipping gap and consistency checks.")


            if quality_checker.original_journal_texts: # Vérifier si les textes étaient chargés
                 print(f"\n[*] Over-Copying Check (Similarity vs Journals - Threshold ~85%):")
                 print(f"  - Estimated potentially copied text: {copied_percentage:.1f}%")
                 if plagiarism_issues:
                     print(f"  - Found {len(plagiarism_issues)} sentences with high similarity:")
                     for issue in plagiarism_issues[:5]: print(f"    - {issue}") # Montrer les 5 premiers
                     if len(plagiarism_issues) > 5: print(f"      ... and {len(plagiarism_issues)-5} more.")
                 else:
                     print("  - No sentences found with very high similarity to original journals.")
            else:
                 print("\n[!] Original journal texts not loaded, skipping plagiarism check.")

            print("-----------------------------\n")

        except Exception as e_qual:
             log.error(f"An error occurred during quality checks: {e_qual}", exc_info=True)
             sys.exit(1)


    elif args.command == "create_visuals":
        log.info("--- Command: create_visuals ---")
        journal_entries = None
        # Essayer de charger depuis la mémoire si une commande précédente les a chargés ?
        # Ou recharger systématiquement pour garantir les données ? Rechargeons pour l'instant.
        if not args.skip_journal_load:
            log.info("Loading and processing journals for visualization data...")
            try:
                journal_entries = process_all_journals(config.JOURNAL_DIR)
                if journal_entries:
                    log.info("Ensuring tags and competencies are mapped...")
                    journal_entries = tag_gen.process_entries(journal_entries)
                    journal_entries = comp_mapper.process_entries(journal_entries)
                else:
                     log.error("No valid journal entries found. Cannot generate visualizations.")
                     sys.exit(1)
            except Exception as e_load_viz:
                 log.error(f"Failed to load journal data for visualization: {e_load_viz}", exc_info=True)
                 sys.exit(1)
        else:
             log.warning("Skipping journal loading for visuals. Ensure data was processed previously.")
             # Note: This branch needs a way to access previously processed data, e.g., via MemoryManager persistence.
             print("Error: --skip_journal_load for visuals requires persistent state (not implemented). Please run without the flag.")
             sys.exit(1) # Ou essayer de continuer sans données ? Non.

        if journal_entries:
             try:
                 log.info("Generating competency timeline visualization...")
                 competency_timeline = comp_mapper.get_competency_timeline(journal_entries)
                 visualizer.plot_competency_timeline(competency_timeline)
                 log.info("Generating project activity visualization...")
                 project_mentions = analyzer.identify_mentioned_projects(journal_entries)
                 visualizer.plot_project_activity(project_mentions)
                 log.info(f"Visualizations saved in '{config.OUTPUT_DIR}' directory.")
                 print(f"\nVisualizations generated and saved in: {config.OUTPUT_DIR}")
             except Exception as e_viz:
                 log.error(f"An error occurred during visualization generation: {e_viz}", exc_info=True)
                 sys.exit(1)


        if journal_entries:
             try:
                 log.info("Generating competency timeline visualization...")
                 competency_timeline = comp_mapper.get_competency_timeline(journal_entries)
                 visualizer.plot_competency_timeline(competency_timeline)

                 log.info("Generating project activity visualization...")
                 # Assurer que l'analyzer peut utiliser les tags chargés
                 project_mentions = analyzer.identify_mentioned_projects(journal_entries)
                 visualizer.plot_project_activity(project_mentions)

                 log.info(f"Visualizations saved in '{config.OUTPUT_DIR}' directory.")
                 print(f"\nVisualizations generated and saved in: {config.OUTPUT_DIR}")
             except Exception as e_viz:
                 log.error(f"An error occurred during visualization generation: {e_viz}", exc_info=True)
                 sys.exit(1)


    elif args.command == "manage_refs":
         log.info("--- Command: manage_refs ---")
         if args.ref_command == "add":
             log.info(f"Adding reference with key: {args.key}")
             try:
                 # Essayer de parser le JSON pour les données supplémentaires
                 extra_data = json.loads(args.data)
                 if not isinstance(extra_data, dict):
                      raise ValueError("--data argument must be a valid JSON dictionary string.")

                 # Créer le dictionnaire de données complet
                 citation_data = {
                     "author": args.author,
                     "year": args.year,
                     "title": args.title,
                     **extra_data # Fusionner avec les données supplémentaires
                 }
                 ref_manager.add_citation(args.key, args.type, citation_data)
                 print(f"Reference '{args.key}' added successfully to {ref_manager.filepath}")
             except json.JSONDecodeError:
                 log.error(f"Invalid JSON string provided for --data: {args.data}", exc_info=True)
                 print(f"Error: Invalid JSON provided for --data argument: '{args.data}'")
                 print("Example format: '{\"publisher\": \"PubCo\", \"place\": \"London\"}'")
             except Exception as e_ref_add:
                 log.error(f"Error adding reference '{args.key}': {e_ref_add}", exc_info=True)
                 print(f"Error adding reference: {e_ref_add}")

         elif args.ref_command == "list":
             log.info("Listing all stored references...")
             citations = ref_manager._load_citations() # Charger les citations actuelles
             if not citations:
                 print("No references found in storage.")
             else:
                 print(f"\n--- Stored References (from {ref_manager.filepath}) ---")
                 # Utiliser generate_bibliography pour un affichage formaté et trié
                 bib_text = ref_manager.generate_bibliography_text() # Prend toutes les citations par défaut
                 print(bib_text)
                 print(f"------------------------ ({len(citations)} total)")


    elif args.command == "run_agent":
        # ++ Définition des outils disponibles DANS le scope de cette commande ++
        AVAILABLE_TOOLS = {
            "search_journal_entries": agent_tools.search_journal_entries,
            "search_guidelines": agent_tools.search_guidelines,
            # Ajoutez ici d'autres outils depuis agent_tools si nécessaire
        }
        # ++ Fin Définition ++

        log.info("--- Command: run_agent ---")
        objective = args.objective
        max_turns = args.max_turns
        log.info(f"Starting agent with objective: '{objective}' (max_turns={max_turns})")

        # --- Définition du Prompt Système Initial ---
        # (Prompt système identique à la version précédente)
        system_prompt = f"""
You are an AI assistant tasked with writing an apprenticeship report based on journal entries and guidelines.
Your goal is to fulfill the user's objective: "{objective}".
You have access to the following tools:

1.  **search_journal_entries(query: str, k: int = 5)**:
    - Use this to find relevant passages from the **year 2** journal entries about projects, tasks, skills, or challenges.
    - Provide a specific query related to the information you need.
    - Example Usage (if you need info on Copilot security challenges):
      `>>>TOOL_CALL
      search_journal_entries(query="challenges related to Copilot security and access control", k=3)
      >>>END_TOOL_CALL`

2.  **search_guidelines(topic: str, k: int = 3)**:
    - Use this to consult the official report guidelines (from the PDF) about requirements for a specific topic or section title.
    - Provide the topic or section title as the query.
    - Example Usage (if you need guidelines for the Introduction):
      `>>>TOOL_CALL
      search_guidelines(topic="Introduction section requirements", k=2)
      >>>END_TOOL_CALL`

**Workflow:**
1.  **Think:** Analyze the objective and the current conversation history. Decide if you need more information.
2.  **Act:** If you need information, choose ONE tool and format your call EXACTLY between `>>>TOOL_CALL` and `>>>END_TOOL_CALL`. Output ONLY the tool call in this format.
3.  **Wait:** The system will execute the tool and provide the result prefixed with `>>>TOOL_RESULT`.
4.  **Respond:** If you don't need a tool, or after receiving a tool result, provide your response, analysis, or the generated text fulfilling the objective.

**Important:**
- Only call one tool per turn.
- Format tool calls precisely as shown.
- If you have enough information, proceed to generate the required text or answer.
- Keep track of the objective: "{objective}".
"""

        # --- Initialisation de la Mémoire Conversationnelle ---
        conversation_history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Please start working on the objective: {objective}"}
        ]

        # --- Boucle Agentique ---
        for turn in range(max_turns):
            log.info(f"--- Agent Turn {turn + 1}/{max_turns} ---")
            print(f"\n--- Turn {turn + 1}/{max_turns} ---")

            # 1. Appeler le LLM
            log.debug(f"Sending context to LLM (last message role: {conversation_history[-1]['role']})")
            llm_response_content = llm._make_request(messages=conversation_history, max_tokens=1500, temperature=0.5)

            if not llm_response_content:
                log.error("LLM failed to respond. Ending agent loop.")
                print("Error: LLM did not respond.")
                break

            log.info(f"LLM Raw Response: {llm_response_content[:200]}...")
            print(f"\nAgent Thought/Response:\n{llm_response_content}\n")
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
                            def parse_kwargs(s):
                                kwargs = {}
                                try:
                                    # Isoler les arguments entre parenthèses
                                    args_content = s[s.find('(') + 1 : s.rfind(')')].strip()
                                    if not args_content: return {} # Pas d'arguments

                                    # Split basique (ne gère pas les virgules/égales dans les valeurs string)
                                    parts = args_content.split(',')
                                    for part in parts:
                                        key_val = part.split('=', 1)
                                        if len(key_val) == 2:
                                            key = key_val[0].strip()
                                            val_str = key_val[1].strip()
                                            try: val = int(val_str)
                                            except ValueError:
                                                try: val = float(val_str)
                                                except ValueError: val = val_str.strip('\'"') # Enlève guillemets
                                            kwargs[key] = val
                                except Exception as e_parse_inner:
                                     log.error(f"Inner parsing error in parse_kwargs for '{s}': {e_parse_inner}")
                                     raise ValueError(f"Could not parse arguments: {s}") from e_parse_inner
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
    else:
        log.error(f"Unknown command received: {args.command}")
        parser.print_help() # Afficher l'aide
        sys.exit(1)


# --- Point d'entrée ---
if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
         # Gérer les sorties normales et d'erreur de sys.exit()
         exit_code = e.code if isinstance(e.code, int) else 1 # Default to 1 if code is not int
         if exit_code != 0: log.warning(f"Script exited with code {exit_code}.")
         else: log.info("Script finished successfully.")
         sys.exit(exit_code) # Propager le code de sortie
    except KeyboardInterrupt:
         log.warning("Script interrupted by user (Ctrl+C).")
         print("\nOperation cancelled by user.")
         sys.exit(130) # Code de sortie standard pour Ctrl+C
    except Exception as e:
         # Attraper toute autre exception non prévue
         log.critical(f"An unexpected critical error occurred in main: {e}", exc_info=True)
         print(f"\nCRITICAL ERROR: {e}. Please check the logs for full details.")
         sys.exit(1) # Quitter avec un code d'erreur générique