# tag_generator.py (Version avec injection LLM LangChain)

import logging
from typing import List
# Importer le type de base LangChain pour l'annotation
from langchain_core.language_models.chat_models import BaseChatModel
# Importer JournalEntry si utilisé pour le type hinting (optionnel ici)
# from data_models import JournalEntry
from typing import List, Any # <-- AJOUTER Any

log = logging.getLogger(__name__)
if not log.handlers: logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')

class TagGenerator:
    """Generates tags for journal entries using an injected LangChain LLM instance."""

    def __init__(self, llm_instance: BaseChatModel):
        """Initializes with a LangChain LLM instance."""
        if llm_instance is None: raise ValueError("LLM instance is required")
        self.llm = llm_instance
        log.info("TagGenerator initialized with provided LLM instance.")

    def generate_tags(self, text: str, max_tags: int = 10) -> List[str]:
        """Generates tags using the provided LangChain LLM instance."""
        log.debug(f"Generating tags for text snippet (length {len(text)})...")
        prompt = f"""
        Analyze the following text and extract up to {max_tags} relevant keywords or tags.
        Focus on specific skills, tools, projects, concepts, or activities.
        Output ONLY a comma-separated list of tags. No explanations or intro.

        Text:
        ---
        {text[:2500]} # Limit context
        ---

        Tags:
        """
        try:
            # Utilise l'interface LangChain invoke
            response = self.llm.invoke(prompt)
            if response and hasattr(response, 'content') and isinstance(response.content, str):
                content = response.content.strip()
                tags = [tag.strip() for tag in content.split(',') if tag.strip() and len(tag) > 1]
                log.info(f"Generated tags: {tags}")
                return tags[:max_tags] # Respecter max_tags
            else:
                log.warning(f"Tag generation returned unexpected response: {response}")
                return []
        except Exception as e:
            log.error(f"Error during tag generation LLM call: {e}", exc_info=True)
            return []

    def process_entries(self, entries: List[Any]) -> List[Any]: # Utiliser Any si JournalEntry non importé
        """Generates tags for a list of journal entries."""
        log.info(f"Processing {len(entries)} entries for tag generation...")
        for i, entry in enumerate(entries):
            if hasattr(entry, 'raw_text') and entry.raw_text:
                 log.debug(f"Processing entry {i+1}/{len(entries)} (ID: {getattr(entry, 'entry_id', 'N/A')})")
                 generated_tags = self.generate_tags(entry.raw_text)
                 if not hasattr(entry, 'tags') or entry.tags is None: entry.tags = []
                 # Ajout sans doublons
                 existing_tags_lower = {t.lower() for t in entry.tags}
                 for t in generated_tags:
                      if t.lower() not in existing_tags_lower:
                           entry.tags.append(t)
            else:
                 log.warning(f"Skipping tag generation for entry {getattr(entry, 'entry_id', 'N/A')}: no raw_text.")
        return entries