import config
from llm_interface import DeepSeekLLM
from data_models import JournalEntry
from typing import List, Dict, Tuple
import logging
import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CompetencyMapper:
    """Maps journal entry content to predefined competencies."""

    def __init__(self):
        self.llm = DeepSeekLLM()
        self.competencies_list = config.COMPETENCIES_TO_TRACK

    def map_competencies_for_entry(self, entry: JournalEntry) -> List[str]:
        """Identifies competencies demonstrated in a single journal entry using LLM."""
        logging.info(f"Mapping competencies for entry {entry.entry_id} ({entry.date})...")
        if not entry.raw_text:
            return []

        text_sample = entry.raw_text[:2000] # Limit context

        prompt = f"""
        Analyze the following journal entry text from an AI Project Officer apprentice.
        Identify which of the predefined competencies are demonstrated or discussed in this entry.
        Consider skills applied, tools used, situations described, and learning experiences.

        Predefined Competencies:
        {', '.join(self.competencies_list)}

        Journal Entry Text:
        ---
        {text_sample}
        ---

        Competencies Demonstrated (Output ONLY a comma-separated list from the predefined list above):
        """
        response = self.llm.analyze_content(text_sample, prompt) # Using generic analyze method

        identified = []
        if response:
            found_competencies = [c.strip() for c in response.split(',') if c.strip()]
            # Validate against the predefined list
            for comp in found_competencies:
                 # Simple matching (case-insensitive)
                 for predefined_comp in self.competencies_list:
                     if comp.lower() == predefined_comp.lower():
                         identified.append(predefined_comp) # Use the canonical name
                         break
                 # Add fuzzy matching here if needed (e.g., using libraries like fuzzywuzzy)
        identified = list(set(identified)) # Ensure uniqueness
        entry.competencies_identified = identified # Store in the entry object
        logging.info(f"Identified competencies for {entry.entry_id}: {identified}")
        return identified

    def process_entries(self, entries: List[JournalEntry]) -> List[JournalEntry]:
        """Maps competencies for a list of journal entries."""
        for entry in entries:
            self.map_competencies_for_entry(entry)
            # Optional delay
            # import time
            # time.sleep(1)
        return entries

    def get_competency_timeline(self, entries: List[JournalEntry]) -> Dict[str, List[datetime.date]]:
        """Generates a timeline showing when each competency was identified."""
        timeline = {comp: [] for comp in self.competencies_list}
        for entry in entries:
            for comp in entry.competencies_identified:
                if comp in timeline: # Ensure it's a tracked competency
                     # Avoid duplicate dates for the same competency on the same day if multiple entries exist
                    if not timeline[comp] or timeline[comp][-1] != entry.date:
                         timeline[comp].append(entry.date)

        # Sort dates for each competency
        for comp in timeline:
            timeline[comp].sort()

        logging.info("Generated competency timeline data.")
        return timeline