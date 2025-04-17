# content_analyzer.py

import config
from llm_interface import GeminiLLM
from data_models import JournalEntry
# Import necessary modules for type hints and functionality
from typing import List, Dict, Optional
import logging
from collections import Counter
import datetime # <-- Add this import

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ContentAnalyzer:
    """Analyzes journal content for themes, patterns, etc."""

    def __init__(self):
        self.llm = GeminiLLM()

    def identify_recurring_themes(self, entries: List[JournalEntry], top_n: int = 5) -> Optional[Dict[str, int]]:
        """Identifies recurring themes across multiple journal entries using LLM."""
        logging.info("Identifying recurring themes...")
        full_text = "\n\n---\n\n".join([f"Date: {e.date}\n{e.raw_text}" for e in entries])

        # Limit text length if necessary
        max_len = 15000 # Adjust based on DeepSeek context limits and performance
        if len(full_text) > max_len:
             logging.warning(f"Full text exceeds {max_len} chars, truncating for theme analysis.")
             full_text = full_text[:max_len] # Or implement smarter sampling/summarization

        prompt = f"""
        Analyze the following collection of journal entries from an AI Project Officer apprentice.
        Identify the main recurring themes, projects, or activities discussed.
        List the top {top_n} most prominent themes. Output ONLY a comma-separated list of themes.

        Journal Entries Excerpts:
        ---
        {full_text}
        ---

        Top Recurring Themes:
        """
        response = self.llm.analyze_content(full_text, prompt) # Using generic analyze method

        if response:
            themes = [theme.strip() for theme in response.split(',') if theme.strip()]
            # Could refine this by using tag counts or more specific LLM analysis
            # For now, assume LLM gives a decent list. Count occurrences if needed.
            theme_counts = Counter(themes) # Simple count if LLM repeats themes
            logging.info(f"Identified themes: {theme_counts}")
            return dict(theme_counts.most_common(top_n))
        else:
            logging.error("Failed to identify recurring themes via LLM.")
            return None

    # This function signature now has 'datetime' defined
    def identify_mentioned_projects(self, entries: List[JournalEntry]) -> Dict[str, List[datetime.date]]:
        """Identifies potential project names mentioned and the dates they appear."""
        logging.info("Identifying mentioned projects...")
        project_mentions: Dict[str, List[datetime.date]] = {} # project_name: [date1, date2]

        # Simple approach: Use tags that seem like project names (e.g., contain "Project", specific acronyms)
        # More robust: Use LLM to explicitly extract project names per entry
        for entry in entries:
            # Example LLM-based extraction per entry (might be slow/costly)
            # prompt = f"Extract specific project names or codenames mentioned in this journal entry. Output ONLY a comma-separated list. Entry:\n---\n{entry.raw_text[:1000]}\n---\nProject Names:"
            # names_str = self.llm.analyze_content(entry.raw_text[:1000], prompt)
            # if names_str:
            #     names = [n.strip() for n in names_str.split(',') if n.strip()]
            #     for name in names:
            #         if name not in project_mentions:
            #             project_mentions[name] = []
            #         if entry.date not in project_mentions[name]: # Check before appending date
            #              project_mentions[name].append(entry.date)

            # Using tags as a proxy (faster, less accurate) - Ensure tags are generated first!
            if hasattr(entry, 'tags') and entry.tags: # Check if tags exist
                for tag in entry.tags:
                    # Heuristics: Tags with capitals, containing 'Project', specific keywords
                    # Refine these heuristics based on typical project naming conventions
                    is_potential_project = (
                        tag.istitle() and len(tag) > 3 or # Catch "Project Alpha"
                        "Project" in tag or
                        tag.isupper() and len(tag) > 2 or # Catch ACRONYMS
                        any(char.isdigit() for char in tag) # Catch names with numbers
                    )
                    # Avoid generic tags like 'AI', 'Python', 'Meeting'
                    generic_terms = {"ai", "python", "meeting", "update", "analysis", "report", "data", "skill"}
                    if is_potential_project and tag.lower() not in generic_terms:
                        if tag not in project_mentions:
                            project_mentions[tag] = []
                        # Only add the date if it's not already there for this project
                        if entry.date not in project_mentions[tag]:
                            project_mentions[tag].append(entry.date)


        # Clean up dates (sort)
        for name in project_mentions:
             project_mentions[name].sort()

        logging.info(f"Identified potential projects and mention dates: {project_mentions}")
        return project_mentions

    def find_potential_gaps(self, entries: List[JournalEntry], expected_topics: List[str]) -> List[str]:
        """Identifies topics from a list that seem underrepresented in the journals."""
        logging.info("Checking for potential content gaps...")
        all_text = " ".join([e.raw_text for e in entries])
        present_topics = set()

        # Simple keyword check (fast, less accurate)
        # for topic in expected_topics:
        #     if topic.lower() in all_text.lower():
        #         present_topics.add(topic)

        # LLM based check (slower, potentially better understanding)
        # Limit context for LLM call
        max_len = 4000 # Or adjust based on model limits
        text_for_llm = all_text[:max_len] if len(all_text) > max_len else all_text

        prompt = f"""
        Review the following combined journal text. Determine which of the following expected topics are significantly discussed or mentioned.
        Output ONLY a comma-separated list of the topics that ARE present in the text. Be precise and only list topics explicitly found.

        Expected Topics: {', '.join(expected_topics)}

        Journal Text Summary:
        ---
        {text_for_llm}
        ---

        Present Topics:
        """
        response = self.llm.analyze_content(text_for_llm, prompt)
        if response:
             # Clean up the response, handle variations
             found_topics_raw = [topic.strip().lower() for topic in response.split(',') if topic.strip()]
             expected_topics_lower = {t.lower(): t for t in expected_topics} # Map lower to original case

             for found_topic_lower in found_topics_raw:
                 # Check if the found topic (lowercase) exists in our expected topics (lowercase keys)
                 if found_topic_lower in expected_topics_lower:
                     present_topics.add(expected_topics_lower[found_topic_lower]) # Add the original cased topic


        missing_topics = [topic for topic in expected_topics if topic not in present_topics]
        logging.info(f"Potential gaps identified (topics not clearly found): {missing_topics}")
        return missing_topics