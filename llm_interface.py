import time
from openai import OpenAI # Use OpenAI's library structure for DeepSeek API
import config
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DeepSeekLLM:
    """Interface for interacting with the DeepSeek API."""

    def __init__(self):
        if not config.DEEPSEEK_API_KEY:
            raise ValueError("DeepSeek API key not configured.")
        self.client = OpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_API_BASE,
        )
        self.chat_model = config.DEEPSEEK_CHAT_MODEL
        # self.embedding_model = config.DEEPSEEK_EMBEDDING_MODEL # If needed

    def _make_request(self, messages: List[Dict[str, str]], max_retries: int = 3, delay: int = 5, **params) -> Optional[str]:
        """Makes a request to the Chat API with retry logic."""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.chat_model,
                    messages=messages,
                    **params
                )
                # Check if response and choices are valid
                if response and response.choices and response.choices[0].message:
                     # Ensure content is not None before returning
                     content = response.choices[0].message.content
                     if content is not None:
                         logging.info(f"LLM request successful. Tokens used: {response.usage}")
                         return content.strip()
                     else:
                        logging.warning("LLM response content is None.")
                        return None # Or raise an error, or return empty string
                else:
                    logging.warning(f"Invalid response structure received: {response}")

            except Exception as e:
                logging.error(f"API Error: {e}. Attempt {attempt + 1} of {max_retries}.")
                if attempt < max_retries - 1:
                    time.sleep(delay)
                else:
                    logging.error("Max retries reached. Failing request.")
                    return None # Or raise the exception: raise e
        return None

    # --- Specific Task Methods ---

    def generate_tags(self, text: str, max_tags: int = 10) -> List[str]:
        """Generates relevant tags for a given text."""
        prompt = f"""
        Analyze the following journal entry text and extract the most relevant keywords or tags (up to {max_tags}).
        Focus on specific skills, tools, projects, company names, concepts, or significant activities mentioned.
        Output ONLY a comma-separated list of tags. Do not include explanations or introductory text.

        Text:
        ---
        {text[:1500]}  # Limit context for tagging if needed
        ---

        Tags:
        """
        messages = [{"role": "user", "content": prompt}]
        response = self._make_request(messages, max_tokens=100, temperature=0.2)
        if response:
            tags = [tag.strip() for tag in response.split(',') if tag.strip()]
            return tags
        return []

    def summarize_text(self, text: str, max_length: int = 150) -> Optional[str]:
        """Summarizes a piece of text."""
        prompt = f"""
        Provide a concise summary (around {max_length} words) of the following text:
        ---
        {text}
        ---
        Summary:
        """
        messages = [{"role": "user", "content": prompt}]
        return self._make_request(messages, max_tokens=200, temperature=0.5)


    def draft_report_section(self, section_title: str, context_chunks: List[str], report_structure: Optional[List[str]] = None, instructions: Optional[str] = None) -> Optional[str]:
        """Drafts a report section using provided context."""
        context = "\n---\n".join(context_chunks)
        structure_info = f"Consider the overall report structure: {', '.join(report_structure)}" if report_structure else ""
        extra_instructions = f"Follow these specific instructions: {instructions}" if instructions else ""

        prompt = f"""
        You are writing a section titled "{section_title}" for an MSc Apprenticeship Report.
        Your persona is a Master's student (Business Transformation & AI) reporting on their experience as an AI Project Officer at Gecina.
        Use the following relevant excerpts from the student's journal as primary context. Synthesize the information, maintain a professional and academic tone.
        Do NOT simply list the excerpts. Integrate the information smoothly into a coherent narrative for the section.
        Avoid overly casual language from the journal. Focus on achievements, learnings, challenges, and activities relevant to the section title.
        {structure_info}
        {extra_instructions}

        Journal Context:
        ---
        {context}
        ---

        Draft for section "{section_title}":
        """
        messages = [{"role": "user", "content": prompt}]
        # Adjust max_tokens based on expected section length
        return self._make_request(messages, max_tokens=1024, temperature=0.6)

    def analyze_content(self, text: str, analysis_prompt: str) -> Optional[str]:
        """Performs a specific analysis task based on a prompt."""
        messages = [
            {"role": "system", "content": "You are an analytical assistant helping to understand journal entries."},
            {"role": "user", "content": f"{analysis_prompt}\n\nText to Analyze:\n---\n{text}\n---\n\nAnalysis:"}
        ]
        return self._make_request(messages, max_tokens=500, temperature=0.3)

    def check_consistency(self, text_segment1: str, text_segment2: str, aspect: str) -> Optional[str]:
        """Checks consistency between two text segments regarding a specific aspect."""
        prompt = f"""
        Analyze the following two text segments from a report. Assess their consistency regarding '{aspect}'.
        Point out any discrepancies or confirm consistency. Be specific.

        Segment 1:
        ---
        {text_segment1}
        ---

        Segment 2:
        ---
        {text_segment2}
        ---

        Consistency Check regarding '{aspect}':
        """
        messages = [{"role": "user", "content": prompt}]
        return self._make_request(messages, max_tokens=300, temperature=0.1)