# competency_mapper.py (Version avec injection LLM LangChain)

import config
import logging
from typing import List, Dict, Optional, Any 
# Importer le type LangChain
from langchain_core.language_models.chat_models import BaseChatModel
# from data_models import JournalEntry # Si besoin pour type hinting

log = logging.getLogger(__name__)
if not log.handlers: logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')

class CompetencyMapper:
    """Maps journal entry content to predefined competencies using an injected LangChain LLM."""

    def __init__(self, llm_instance: BaseChatModel):
        """Initializes with a LangChain LLM instance."""
        if llm_instance is None: raise ValueError("LLM instance is required")
        self.llm = llm_instance
        self.competencies_list = config.COMPETENCIES_TO_TRACK # Récupère depuis config
        self.competencies_lower_map = {c.lower(): c for c in self.competencies_list} # Pour validation insensible à la casse
        log.info("CompetencyMapper initialized with provided LLM instance.")

    def map_competencies_for_entry(self, entry: Any) -> List[str]: # Utiliser Any pour entry
        """Identifies competencies in a single journal entry using the LLM."""
        if not hasattr(entry, 'raw_text') or not entry.raw_text: return []
        entry_id = getattr(entry, 'entry_id', 'N/A')
        log.debug(f"Mapping competencies for entry {entry_id}...")
        text_sample = entry.raw_text[:2500] # Limiter contexte

        prompt = f"""
        Analyze the following journal entry text.
        Identify which of the predefined competencies are demonstrated or discussed.
        Consider skills, tools, situations, and learning experiences.

        Predefined Competencies:
        {', '.join(self.competencies_list)}

        Journal Entry Text:
        ---
        {text_sample}
        ---

        Competencies Demonstrated (Output ONLY a comma-separated list from the predefined list above. If none, output 'None'):
        """
        identified_canonical = []
        try:
            response = self.llm.invoke(prompt)
            if response and hasattr(response, 'content') and isinstance(response.content, str):
                 content = response.content.strip()
                 if content.lower() != 'none':
                     found_competencies = [c.strip() for c in content.split(',') if c.strip()]
                     for comp_found in found_competencies:
                         comp_lower = comp_found.lower()
                         if comp_lower in self.competencies_lower_map:
                              # Utiliser la casse canonique de la liste config
                              canonical_name = self.competencies_lower_map[comp_lower]
                              if canonical_name not in identified_canonical: # Eviter doublons
                                   identified_canonical.append(canonical_name)
                         else:
                              log.warning(f"LLM returned competency '{comp_found}' not in predefined list for entry {entry_id}.")
            else: log.warning(f"Competency mapping returned unexpected response: {response}")

        except Exception as e: log.error(f"Error during competency mapping LLM call for entry {entry_id}: {e}", exc_info=True)

        # Mettre à jour l'entrée directement (si l'objet est mutable et passé par référence)
        if hasattr(entry, 'competencies_identified'):
             entry.competencies_identified = identified_canonical
        log.info(f"Identified competencies for {entry_id}: {identified_canonical}")
        return identified_canonical # Retourner aussi la liste

    def process_entries(self, entries: List[Any]) -> List[Any]: # Utiliser Any pour entry
        """Maps competencies for a list of journal entries."""
        log.info(f"Processing {len(entries)} entries for competency mapping...")
        for i, entry in enumerate(entries):
            log.debug(f"Processing entry {i+1}/{len(entries)} (ID: {getattr(entry, 'entry_id', 'N/A')})")
            self.map_competencies_for_entry(entry)
            # Pas de sleep nécessaire ici, les appels LLM sont séquentiels
        return entries