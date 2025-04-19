# content_analyzer.py (Version avec injection LLM LangChain)

import config
import logging
from typing import List, Dict, Optional, Any
# Importer le type LangChain
from langchain_core.language_models.chat_models import BaseChatModel
from collections import Counter
import datetime
# from data_models import JournalEntry # Si besoin type hinting

log = logging.getLogger(__name__)
if not log.handlers: logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')

class ContentAnalyzer:
    """Analyzes journal content using an injected LangChain LLM instance."""

    def __init__(self, llm_instance: BaseChatModel):
        """Initializes with a LangChain LLM instance."""
        if llm_instance is None: raise ValueError("LLM instance is required")
        self.llm = llm_instance
        log.info("ContentAnalyzer initialized with provided LLM instance.")

    def identify_recurring_themes(self, entries: List[Any], top_n: int = 5) -> Optional[Dict[str, int]]:
        """Identifies recurring themes across multiple journal entries using LLM."""
        log.info("Identifying recurring themes...")
        # S'assurer que les objets entry ont bien l'attribut 'raw_text' et 'date'
        texts_with_dates = [f"Date: {getattr(e, 'date', 'Unknown')}\n{getattr(e, 'raw_text', '')}"
                           for e in entries if hasattr(e, 'raw_text')]
        if not texts_with_dates: log.warning("No entries with text found for theme analysis."); return None

        full_text = "\n\n---\n\n".join(texts_with_dates)
        max_len = 15000
        if len(full_text) > max_len: log.warning("Truncating text for theme analysis."); full_text = full_text[:max_len]

        prompt = f"""
        Analyze the following journal entries. Identify the top {top_n} main recurring themes, projects, or activities.
        Output ONLY a comma-separated list of these themes. No explanations.

        Journal Excerpts:
        ---
        {full_text}
        ---

        Top Recurring Themes:
        """
        try:
            response = self.llm.invoke(prompt) # Utiliser invoke
            if response and hasattr(response, 'content') and isinstance(response.content, str):
                 content = response.content.strip()
                 themes = [theme.strip() for theme in content.split(',') if theme.strip()]
                 theme_counts = Counter(themes)
                 log.info(f"Identified themes: {dict(theme_counts)}")
                 return dict(theme_counts.most_common(top_n))
            else: log.warning(f"Theme analysis returned unexpected response: {response}"); return None
        except Exception as e: log.error(f"Error during theme analysis LLM call: {e}", exc_info=True); return None


    def identify_mentioned_projects(self, entries: List[Any]) -> Dict[str, List[datetime.date]]:
        """Identifies potential project names mentioned and their dates (using tags as proxy)."""
        # Cette version utilise les tags, pas directement le LLM ici.
        log.info("Identifying mentioned projects (using tags)...")
        project_mentions: Dict[str, List[datetime.date]] = {}
        for entry in entries:
            entry_date = getattr(entry, 'date', None)
            if not entry_date or not hasattr(entry, 'tags') or not entry.tags: continue

            for tag in entry.tags:
                 is_potential_project = (tag.istitle() and len(tag)>3 or "Project" in tag or tag.isupper() and len(tag)>2 or any(char.isdigit() for char in tag))
                 generic_terms = {"ai", "python", "meeting", "update", "analysis", "report", "data", "skill"}
                 if is_potential_project and tag.lower() not in generic_terms:
                     if tag not in project_mentions: project_mentions[tag] = []
                     if entry_date not in project_mentions[tag]: project_mentions[tag].append(entry_date)
        for name in project_mentions: project_mentions[name].sort()
        log.info(f"Identified potential projects via tags: {project_mentions}")
        return project_mentions

    def find_potential_gaps(self, entries: List[Any], expected_topics: List[str]) -> List[str]:
        """Identifies topics from a list that seem underrepresented using LLM."""
        log.info("Checking for potential content gaps using LLM...")
        all_text = " ".join([getattr(e, 'raw_text', '') for e in entries])
        if not all_text.strip() or not expected_topics: return []

        max_len = 15000; text_for_llm = all_text[:max_len]
        prompt = f"""
        Review the following text. Which of the expected topics below are significantly discussed?
        Output ONLY a comma-separated list of the topics found in the text.

        Expected Topics: {', '.join(expected_topics)}

        Text Summary:
        ---
        {text_for_llm}
        ---

        Present Topics Found:
        """
        present_topics_found = set()
        try:
            response = self.llm.invoke(prompt) # Utiliser invoke
            if response and hasattr(response, 'content') and isinstance(response.content, str):
                content = response.content.strip()
                present_topics_found.update([topic.strip().lower() for topic in content.split(',') if topic.strip()])
        except Exception as e: log.error(f"Error during gap analysis LLM call: {e}", exc_info=True)

        expected_topics_lower = {t.lower() for t in expected_topics}
        missing_topics_lower = expected_topics_lower - present_topics_found
        # Retrouver la casse originale
        missing_topics = [t for t in expected_topics if t.lower() in missing_topics_lower]
        log.info(f"Potential gaps (topics not found by LLM): {missing_topics}")
        return missing_topics