import config
from llm_interface import DeepSeekLLM
from data_models import JournalEntry
from typing import List
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TagGenerator:
    """Generates tags for journal entries using the LLM."""

    def __init__(self):
        self.llm = DeepSeekLLM()

    def generate_tags_for_entry(self, entry: JournalEntry, max_tags: int = 10) -> List[str]:
        """Generates and assigns tags to a single JournalEntry object."""
        if not entry.raw_text:
            logging.warning(f"Entry {entry.entry_id} has no raw text. Skipping tag generation.")
            return []

        logging.info(f"Generating tags for entry {entry.entry_id} ({entry.date})...")
        # Use a portion of the text if it's very long, to save tokens/time
        text_sample = entry.raw_text[:2000] # Use first 2000 chars

        tags = self.llm.generate_tags(text_sample, max_tags=max_tags)

        if tags:
            logging.info(f"Generated tags for {entry.entry_id}: {tags}")
            entry.tags = list(set(entry.tags + tags)) # Add new tags, ensure uniqueness
            return tags
        else:
            logging.warning(f"Could not generate tags for entry {entry.entry_id}.")
            return []

    def process_entries(self, entries: List[JournalEntry]) -> List[JournalEntry]:
        """Generates tags for a list of journal entries."""
        for entry in entries:
            self.generate_tags_for_entry(entry)
            # Optional: Add a small delay between API calls if needed
            # import time
            # time.sleep(1)
        return entries