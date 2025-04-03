import config
from data_models import ReportPlan, ReportSection
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ProgressTracker:
    """Tracks the completion status of the report sections."""

    def __init__(self):
        pass # No persistent state needed here, operates on the plan

    def calculate_progress(self, report_plan: ReportPlan) -> Tuple[int, int, float]:
        """
        Calculates the number of completed sections and the overall percentage.
        Considers 'drafted', 'reviewed', 'final' as complete for percentage calculation.
        Returns (completed_count, total_count, percentage).
        """
        total_sections = 0
        completed_sections = 0
        # Define what statuses count as 'completed' for the percentage
        completed_statuses = {"drafted", "reviewed", "final"}

        def count_sections(sections: List[ReportSection]):
            nonlocal total_sections, completed_sections
            for section in sections:
                 # Skip bibliography/appendices? Or count them? Count for now.
                 # if section.title.lower() not in ["bibliography", "appendices"]:
                 total_sections += 1
                 if section.status in completed_statuses:
                     completed_sections += 1
                 count_sections(section.subsections)

        if not report_plan or not report_plan.structure:
             return 0, 0, 0.0

        count_sections(report_plan.structure)

        percentage = (completed_sections / total_sections * 100) if total_sections > 0 else 0.0
        logging.info(f"Progress: {completed_sections}/{total_sections} sections completed ({percentage:.2f}%).")
        return completed_sections, total_sections, round(percentage, 2)

    def get_pending_sections(self, report_plan: ReportPlan) -> List[str]:
        """Returns a list of section titles that are still pending."""
        pending = []
        pending_statuses = {"pending", "failed"} # Include failed sections as needing attention

        def find_pending(sections: List[ReportSection]):
            for section in sections:
                 if section.status in pending_statuses:
                      pending.append(section.title)
                 # Only check subsections if parent is not pending/failed? Optional logic.
                 find_pending(section.subsections)

        if report_plan and report_plan.structure:
            find_pending(report_plan.structure)

        logging.info(f"Found {len(pending)} pending/failed sections.")
        return pending