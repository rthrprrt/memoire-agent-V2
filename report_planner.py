import config
from data_models import ReportPlan, ReportSection
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ReportPlanner:
    """Creates the structure (plan) for the apprenticeship report."""

    def __init__(self):
        # Potentially load thesis requirements from a file or use config default
        self.default_structure = config.DEFAULT_REPORT_STRUCTURE
        pass # No complex state needed for basic planning

    def create_base_plan(self, structure_definition: Optional[List[str]] = None) -> ReportPlan:
        """Creates a ReportPlan object based on a list of section titles."""
        if structure_definition is None:
            structure_definition = self.default_structure
            logging.info("Using default report structure from config.")

        report_sections = []
        current_level_1 = None
        current_level_2 = None

        for title_raw in structure_definition:
            title = title_raw.strip()
            level = 0
            if title.startswith("   "): # Assuming 3 spaces for Level 3
                level = 3
                title = title.lstrip()
            elif title.startswith(" "): # Assuming 1 space (or more) for Level 2
                 level = 2
                 title = title.lstrip()
            else:
                level = 1

            section = ReportSection(title=title, level=level, status="pending")

            if level == 1:
                report_sections.append(section)
                current_level_1 = section
                current_level_2 = None # Reset level 2 when a new level 1 starts
            elif level == 2:
                if current_level_1:
                    current_level_1.subsections.append(section)
                    current_level_2 = section
                else:
                    logging.warning(f"Level 2 section '{title}' found without preceding Level 1. Adding to root.")
                    report_sections.append(section) # Add as a top-level section as fallback
            elif level == 3:
                if current_level_2:
                    current_level_2.subsections.append(section)
                elif current_level_1:
                     logging.warning(f"Level 3 section '{title}' found without preceding Level 2. Adding to Level 1.")
                     current_level_1.subsections.append(section) # Add to level 1 as fallback
                else:
                     logging.warning(f"Level 3 section '{title}' found without preceding Level 1 or 2. Adding to root.")
                     report_sections.append(section) # Add as a top-level section as fallback


        plan = ReportPlan(structure=report_sections)
        logging.info("Generated basic report plan.")
        return plan

    # --- Optional Enhancements ---
    # def refine_plan_with_data(self, plan: ReportPlan, entries: List[JournalEntry]) -> ReportPlan:
    #     """
    #     (Future Enhancement) Analyzes journal entries to suggest specific subsections
    #     (e.g., naming projects automatically) or identify areas with rich data.
    #     """
    #     logging.info("Refining plan with journal data (placeholder)...")
    #     # Example: Find project names and add them as subsections under "Projects Undertaken"
    #     # Requires content analysis results
    #     # analyzer = ContentAnalyzer()
    #     # projects = analyzer.identify_mentioned_projects(entries)
    #     # for section in plan.structure:
    #     #     if "Projects Undertaken" in section.title:
    #     #          section.subsections = [] # Clear potential defaults
    #     #          for proj_name in projects:
    #     #              section.subsections.append(ReportSection(title=f"Project: {proj_name}", level=section.level + 1))
    #     #         break
    #     return plan