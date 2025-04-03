# main.py (Version adaptée pour embeddings locaux)

import argparse
import logging
import os
import sys
import json # Importé pour la gestion des références

# Import necessary modules from the project
import config
from document_processor import process_all_journals, chunk_text
from vector_database import VectorDBManager
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
                                              "and store in Vector DB using local embeddings.")
    parser_process.add_argument("--journal_dir", default=config.JOURNAL_DIR,
                                help="Directory containing journal DOCX files (expected format: YYYY-MM-DD.docx).")
    parser_process.add_argument("--reprocess_all", action="store_true",
                                help="Clear existing vector DB collection before processing.")

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
        # Utiliser Memory Manager pour conserver l'état pendant l'exécution du script
        memory = MemoryManager()
        vector_db = VectorDBManager() # Initialise la connexion et le modèle d'embedding local
        tag_gen = TagGenerator()      # Initialise le générateur de tags (utilise DeepSeek)
        comp_mapper = CompetencyMapper() # Initialise le mappeur de compétences (utilise DeepSeek)
        analyzer = ContentAnalyzer()    # Initialise l'analyseur (utilise DeepSeek)
        planner = ReportPlanner()
        # Le générateur utilise DeepSeek pour la rédaction mais la DB locale pour le contexte
        generator = ReportGenerator(vector_db)
        # Le quality checker utilise la DB locale et DeepSeek pour l'analyse
        quality_checker = QualityChecker(vector_db)
        visualizer = Visualizer()
        ref_manager = ReferenceManager() # Gère references.json
        progress_tracker = ProgressTracker()
        log.info("Core components initialized successfully.")
    except Exception as e:
        log.error(f"FATAL: Failed to initialize core components: {e}", exc_info=True)
        sys.exit(1) # Quitter si les composants essentiels ne peuvent être initialisés

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
            log.info("Loading and processing journals to generate visualization data...")
            try:
                journal_entries = process_all_journals(config.JOURNAL_DIR)
                # Assurer que les tags et compétences sont présents pour les visualisations
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
             log.warning("Skipping journal loading as per --skip_journal_load. Assumes previous steps populated necessary data.")
             # Ici, il faudrait un moyen de récupérer les journal_entries depuis la mémoire ou un état sauvegardé.
             # Pour l'instant, cette option est limitée sans MemoryManager persistant.
             print("Error: --skip_journal_load for visuals is not fully supported without persistent state.")
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


    elif args.command == "run_all":
        log.info("--- Command: run_all (Executing full workflow) ---")
        # Enchaînement simple des commandes via os.system (moins robuste qu'appeler les fonctions)
        # Gérer les erreurs entre étapes serait mieux avec des appels de fonction Python directs.
        reprocess_flag = "--reprocess_all" if args.reprocess else ""

        print(f"\n>>> [1/5] Processing Journals ({'Reprocessing' if args.reprocess else 'Standard'})...")
        cmd1 = f'python "{sys.argv[0]}" process_journals --journal_dir "{args.journal_dir}" {reprocess_flag}'
        log.debug(f"Executing: {cmd1}")
        exit_code1 = os.system(cmd1)
        if exit_code1 != 0:
            log.error("Journal processing failed. Aborting workflow.")
            sys.exit(exit_code1)

        print("\n>>> [2/5] Creating Report Plan...")
        cmd2 = f'python "{sys.argv[0]}" create_plan'
        log.debug(f"Executing: {cmd2}")
        exit_code2 = os.system(cmd2)
        if exit_code2 != 0:
            log.error("Report plan creation failed. Aborting workflow.")
            sys.exit(exit_code2)

        print("\n>>> [3/5] Generating Report Draft...")
        cmd3 = f'python "{sys.argv[0]}" generate_report --output_file "{args.output_file}"'
        log.debug(f"Executing: {cmd3}")
        exit_code3 = os.system(cmd3)
        if exit_code3 != 0:
            log.error("Report generation failed. Aborting workflow.")
            sys.exit(exit_code3)

        print("\n>>> [4/5] Running Quality Checks...")
        cmd4 = f'python "{sys.argv[0]}" check_quality --report_file "{args.output_file}"'
        log.debug(f"Executing: {cmd4}")
        exit_code4 = os.system(cmd4)
        # On n'arrête pas forcément le workflow si les quality checks échouent, juste on log

        print("\n>>> [5/5] Creating Visualizations...")
        cmd5 = f'python "{sys.argv[0]}" create_visuals'
        log.debug(f"Executing: {cmd5}")
        exit_code5 = os.system(cmd5)
        # On n'arrête pas forcément non plus ici

        log.info("--- Full Workflow Attempt Finished ---")
        print(f"\nFull workflow finished. Check logs for details. Final draft expected at: {args.output_file}")


if __name__ == "__main__":
    # Encapsuler l'appel principal dans un try/except pour attraper les erreurs non gérées
    try:
        main()
    except SystemExit as e:
         # Permet aux sys.exit() de fonctionner normalement sans afficher de traceback complet
         if e.code != 0:
              log.info(f"Script exited with code {e.code}.")
         else:
              log.info("Script finished successfully.")
    except Exception as e:
         # Attrape toute autre exception non prévue au niveau supérieur
         log.critical(f"An unexpected critical error occurred: {e}", exc_info=True)
         print(f"\nCRITICAL ERROR: {e}. Please check the logs for details.")
         sys.exit(1) # Quitter avec un code d'erreur