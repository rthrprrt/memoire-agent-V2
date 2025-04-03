from typing import List, Optional, Dict
from data_models import JournalEntry, ReportPlan
import json
import os
import config
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MemoryManager:
    """
    Simple state manager for the agent during a single execution.
    Primarily holds loaded data like journal entries and the report plan.
    Persistence between script runs relies on saving/loading files (e.g., plan JSON).
    """

    def __init__(self):
        self.journal_entries: List[JournalEntry] = []
        self.report_plan: Optional[ReportPlan] = None
        self.citations: Dict[str, Dict] = {} # Store citation data {key: data}

    def load_journal_entries(self, entries: List[JournalEntry]):
        """Loads processed journal entries into memory."""
        self.journal_entries = entries
        logging.info(f"Loaded {len(entries)} journal entries into memory.")

    def get_journal_entries(self) -> List[JournalEntry]:
        """Returns the loaded journal entries."""
        return self.journal_entries

    def get_entry_by_id(self, entry_id: str) -> Optional[JournalEntry]:
        """Finds a specific journal entry by its ID."""
        for entry in self.journal_entries:
            if entry.entry_id == entry_id:
                return entry
        return None

    def save_report_plan(self, plan: ReportPlan, filepath: str = config.DEFAULT_PLAN_FILE):
        """Saves the report plan to a JSON file."""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(plan.model_dump(mode='json'), f, indent=4) # Use model_dump for Pydantic v2
            logging.info(f"Report plan saved to {filepath}")
        except Exception as e:
            logging.error(f"Error saving report plan to {filepath}: {e}")

    def load_report_plan(self, filepath: str = config.DEFAULT_PLAN_FILE) -> Optional[ReportPlan]:
        """Loads the report plan from a JSON file."""
        if not os.path.exists(filepath):
            logging.warning(f"Report plan file not found: {filepath}")
            return None
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                plan_data = json.load(f)
            self.report_plan = ReportPlan(**plan_data)
            logging.info(f"Report plan loaded successfully from {filepath}")
            return self.report_plan
        except Exception as e:
            logging.error(f"Error loading report plan from {filepath}: {e}")
            self.report_plan = None
            return None

    def get_report_plan(self) -> Optional[ReportPlan]:
        """Returns the loaded report plan."""
        return self.report_plan

    def update_section_status(self, section_title: str, new_status: str):
        """Updates the status of a section in the loaded report plan."""
        if not self.report_plan:
            logging.warning("Cannot update section status: No report plan loaded.")
            return False

        def find_and_update(sections: List):
            for section in sections:
                if section.title == section_title:
                    section.status = new_status
                    logging.debug(f"Updated status for section '{section_title}' to '{new_status}'.")
                    return True
                if section.subsections:
                    if find_and_update(section.subsections):
                        return True
            return False

        if find_and_update(self.report_plan.structure):
             return True
        else:
             logging.warning(f"Section '{section_title}' not found in the report plan.")
             return False

    # Add methods for managing citations if needed
    def add_citation(self, key: str, data: Dict):
         self.citations[key] = data
         logging.info(f"Added citation '{key}'")

    def get_citations(self) -> Dict[str, Dict]:
         return self.citations