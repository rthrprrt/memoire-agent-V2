import config
from data_models import Citation
from typing import Dict, List, Optional
import logging
import json
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Basic Harvard formatting rules (can be significantly expanded)
# This is a simplified implementation. Consider using libraries like 'citeproc-py'
# or 'pybtex' for more robust citation formatting if needed.

class ReferenceManager:
    """Manages citations and generates a Harvard-style bibliography."""

    def __init__(self, filepath: str = os.path.join(config.OUTPUT_DIR, "references.json")):
        self.filepath = filepath
        self.citations: Dict[str, Citation] = self._load_citations()

    def _load_citations(self) -> Dict[str, Citation]:
        """Loads citations from a JSON file."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Validate and parse into Citation objects
                    loaded_citations = {}
                    for key, item_data in data.items():
                        try:
                            # Ensure 'data' is treated as a dict even if loaded as str
                            if isinstance(item_data.get('data'), str):
                                item_data['data'] = json.loads(item_data['data'])
                            loaded_citations[key] = Citation(**item_data)
                        except Exception as e:
                            logging.error(f"Error parsing citation data for key '{key}': {e}. Skipping.")
                    logging.info(f"Loaded {len(loaded_citations)} citations from {self.filepath}")
                    return loaded_citations
            except Exception as e:
                logging.error(f"Error loading citations from {self.filepath}: {e}")
        return {}

    def _save_citations(self):
        """Saves the current citations to the JSON file."""
        try:
            # Prepare data for JSON serialization using model_dump
            data_to_save = {key: citation.model_dump(mode='json') for key, citation in self.citations.items()}
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=4)
            logging.info(f"Saved {len(self.citations)} citations to {self.filepath}")
        except Exception as e:
            logging.error(f"Error saving citations to {self.filepath}: {e}")


    def add_citation(self, key: str, citation_type: str, data: Dict):
        """Adds a new citation."""
        if key in self.citations:
            logging.warning(f"Citation key '{key}' already exists. Overwriting.")
        citation = Citation(key=key, citation_type=citation_type, data=data)
        citation.formatted_harvard = self._format_harvard(citation) # Format on add
        self.citations[key] = citation
        self._save_citations()
        logging.info(f"Added and formatted citation: {key}")

    def get_citation(self, key: str) -> Optional[Citation]:
        """Retrieves a citation by its key."""
        return self.citations.get(key)

    def _format_harvard(self, citation: Citation) -> str:
        """Formats a single citation in a simplified Harvard style."""
        # --- Simplified Harvard Formatting ---
        data = citation.data
        try:
            authors = data.get('author', 'Anon')
            year = data.get('year', 'n.d.') # No date
            title = data.get('title', 'Untitled')
            # Add italics for book/journal titles potentially using markdown/HTML later
            # For plain text:
            title_formatted = f"'{title}'" # Use single quotes for article titles
            if citation.citation_type.lower() in ['book', 'report']:
                 title_formatted = f"*{title}*" # Use asterisks for italics placeholder for books

            if citation.citation_type.lower() == 'book':
                publisher = data.get('publisher', '')
                place = data.get('place', '') # Place of publication
                location = f"{place}: {publisher}" if place and publisher else publisher or place
                return f"{authors} ({year}) {title_formatted}. {location}."

            elif citation.citation_type.lower() == 'article':
                journal = data.get('journal', 'Unknown Journal')
                volume = data.get('volume', '')
                issue = data.get('issue', '')
                pages = data.get('pages', '')
                vol_issue = f"{volume}({issue})" if volume and issue else volume or issue
                location = f"*{journal}*, {vol_issue}, pp. {pages}" if vol_issue and pages else f"*{journal}*"
                return f"{authors} ({year}) {title_formatted}. {location}."

            elif citation.citation_type.lower() == 'web':
                url = data.get('url', '')
                accessed_date = data.get('accessed', 'n.d.')
                website_name = data.get('website_name', '') # e.g., BBC News
                name_part = f"*{website_name}*." if website_name else ""
                return f"{authors} ({year}) {title_formatted}. {name_part} [Online]. Available at: {url} (Accessed: {accessed_date})."

            else: # Default basic format
                return f"{authors} ({year}) {title_formatted}."

        except Exception as e:
            logging.error(f"Error formatting citation {citation.key}: {e}")
            # Return basic info on error
            return f"{data.get('author', 'Anon')} ({data.get('year', 'n.d.')}) {data.get('title', '[Formatting Error]')}"


    def generate_bibliography_text(self, used_keys: Optional[List[str]] = None) -> str:
        """
        Generates a sorted, formatted bibliography string.
        If used_keys is provided, only includes those citations. Otherwise, includes all.
        """
        logging.info("Generating bibliography...")
        bib_items = []

        target_citations = {}
        if used_keys is None:
            target_citations = self.citations
            logging.info("Including all stored citations in bibliography.")
        else:
            logging.info(f"Including {len(used_keys)} specified citations in bibliography.")
            used_keys_set = set(used_keys)
            target_citations = {k: v for k, v in self.citations.items() if k in used_keys_set}
            # Log missing keys
            missing_keys = used_keys_set - set(target_citations.keys())
            if missing_keys:
                logging.warning(f"The following citation keys were requested but not found: {', '.join(missing_keys)}")


        if not target_citations:
            return "No citations available or specified."

        # Sort citations alphabetically by author, then year
        sorted_keys = sorted(target_citations.keys(), key=lambda k: (
            target_citations[k].data.get('author', '').lower(),
            target_citations[k].data.get('year', 0)
        ))

        for key in sorted_keys:
            citation = target_citations[key]
            if not citation.formatted_harvard: # Format if not already done
                 citation.formatted_harvard = self._format_harvard(citation)
            bib_items.append(citation.formatted_harvard)

        return "\n".join(bib_items)

    # Placeholder for finding citation keys in text (complex NLP task)
    def find_citation_keys_in_text(self, text: str) -> List[str]:
         """(Placeholder) Finds citation keys like (Smith, 2023) in text."""
         logging.warning("find_citation_keys_in_text is a placeholder and needs implementation.")
         # Regex could find patterns like (Author, Year) or [Number]
         # E.g., import re; keys = re.findall(r'\(([A-Za-z]+),\s*(\d{4})\)', text)
         return []