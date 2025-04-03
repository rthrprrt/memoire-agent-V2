import config
from llm_interface import DeepSeekLLM
from data_models import ReportPlan, ReportSection, JournalEntry
from document_processor import extract_text_from_docx # To read the generated report
from vector_database import VectorDBManager # To compare against original chunks
from typing import List, Dict, Optional, Tuple
import logging
import difflib # For basic text comparison

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class QualityChecker:
    """Performs various quality checks on the generated report."""

    def __init__(self, vector_db: VectorDBManager):
        self.llm = DeepSeekLLM()
        self.vector_db = vector_db
        # Store original journal text for plagiarism check
        # This might consume significant memory for large journals.
        # Consider loading on demand or using the vector DB more cleverly.
        self.original_journal_texts: Dict[str, str] = {} # entry_id: raw_text

    def load_journal_texts(self, entries: List[JournalEntry]):
        """Loads original texts for comparison."""
        self.original_journal_texts = {entry.entry_id: entry.raw_text for entry in entries}
        logging.info(f"Loaded raw text for {len(self.original_journal_texts)} entries for quality checks.")

    def check_consistency_across_sections(self, report_plan: ReportPlan) -> List[str]:
        """
        Checks for consistency in key information (e.g., project names, timelines)
        across different sections of the report using LLM.
        """
        logging.info("Checking consistency across report sections...")
        issues = []
        sections_with_content = []

        def collect_content(sections: List[ReportSection]):
            for sec in sections:
                if sec.content and len(sec.content) > 50: # Only check sections with substantial content
                    sections_with_content.append(sec)
                collect_content(sec.subsections)

        collect_content(report_plan.structure)

        if len(sections_with_content) < 2:
            logging.info("Not enough content-filled sections to perform consistency checks.")
            return issues

        # Compare pairs of sections (can be computationally intensive)
        # Focus on potentially overlapping areas like project descriptions
        # For demonstration, compare first project description with challenges section (example)
        project_sections = [s for s in sections_with_content if "project" in s.title.lower()]
        challenge_section = next((s for s in sections_with_content if "challenges" in s.title.lower()), None)

        if project_sections and challenge_section:
             for proj_sec in project_sections:
                 logging.debug(f"Checking consistency between '{proj_sec.title}' and '{challenge_section.title}'")
                 # Define aspect to check (e.g., project timeline, key challenge mentioned)
                 aspect = f"details and timeline of {proj_sec.title}"
                 consistency_result = self.llm.check_consistency(
                     proj_sec.content,
                     challenge_section.content,
                     aspect
                 )
                 if consistency_result and ("discrepancy" in consistency_result.lower() or "inconsistent" in consistency_result.lower()):
                      issue_desc = f"Potential inconsistency between '{proj_sec.title}' and '{challenge_section.title}' regarding {aspect}: {consistency_result}"
                      issues.append(issue_desc)
                      logging.warning(issue_desc)
                 elif consistency_result:
                      logging.info(f"Consistency check passed between '{proj_sec.title}' and '{challenge_section.title}'.")
                 else:
                      logging.warning(f"Could not perform consistency check between '{proj_sec.title}' and '{challenge_section.title}'.")


        logging.info(f"Consistency check finished. Found {len(issues)} potential issues.")
        return issues


    def check_plagiarism_against_journals(self, report_docx_path: str, similarity_threshold: float = 0.85) -> Tuple[List[str], float]:
        """
        Checks generated report sections for overly direct copying from original journals.
        Returns list of potential issues and estimated percentage of *potentially* copied content.
        NOTE: This is a simplified check and not a robust plagiarism detection system.
        """
        logging.info("Checking for potential 'plagiarism' (over-copying) against original journals...")
        potential_issues = []
        total_report_sentences = 0
        copied_sentences = 0

        try:
            report_text = extract_text_from_docx(report_docx_path)
            if not report_text:
                logging.error(f"Could not read report file: {report_docx_path}")
                return ["Error reading report file"], 0.0
        except Exception as e:
            logging.error(f"Error reading report file {report_docx_path}: {e}")
            return [f"Error reading report file: {e}"], 0.0

        # Simple sentence splitting (can be improved with NLTK/spaCy)
        report_sentences = [s.strip() for s in report_text.replace('\n', ' ').split('.') if len(s.strip()) > 15] # Basic filter
        total_report_sentences = len(report_sentences)

        if not self.original_journal_texts:
            logging.warning("Original journal texts not loaded. Cannot perform plagiarism check.")
            return ["Journal texts not loaded"], 0.0
        if total_report_sentences == 0:
             logging.warning("Report text seems empty or unparseable into sentences.")
             return ["Report text empty or unparseable"], 0.0


        # Compare each report sentence against all journal sentences (can be slow!)
        # Optimization: Use Vector DB search first to find candidate journal entries/chunks.
        all_journal_text = " ".join(self.original_journal_texts.values())
        journal_sentences = set(s.strip() for s in all_journal_text.replace('\n', ' ').split('.') if len(s.strip()) > 15)

        for rep_sent in report_sentences:
            # Use SequenceMatcher for similarity check
            best_match_ratio = 0.0
            best_match_journal_sent = ""
            # Comparing against every journal sentence is very slow.
            # A better approach would be to query the vector DB with the report sentence
            # and check similarity against the retrieved chunks/sentences.
            # Simplified check for demo:
            for jour_sent in journal_sentences:
                 # Quick length check to prune obvious non-matches
                 if abs(len(rep_sent) - len(jour_sent)) > len(rep_sent) * 0.5:
                     continue

                 similarity = difflib.SequenceMatcher(None, rep_sent.lower(), jour_sent.lower()).ratio()
                 if similarity > best_match_ratio:
                     best_match_ratio = similarity
                     best_match_journal_sent = jour_sent # Keep track if needed

                 if similarity >= similarity_threshold:
                     break # Found a close enough match

            if best_match_ratio >= similarity_threshold:
                issue_desc = f"High similarity ({best_match_ratio:.2f}) found for sentence: '{rep_sent[:100]}...' (Potential source: '{best_match_journal_sent[:100]}...')"
                potential_issues.append(issue_desc)
                copied_sentences += 1
                # Optimization: remove matched journal sentence if it shouldn't match multiple times?
                # if best_match_journal_sent in journal_sentences:
                #      journal_sentences.remove(best_match_journal_sent)


        ai_generated_percentage = 100.0 * (1 - (copied_sentences / total_report_sentences)) if total_report_sentences > 0 else 0.0
        logging.info(f"Plagiarism check finished. Found {len(potential_issues)} potentially copied sentences.")
        logging.info(f"Estimated percentage of potentially copied text: {100.0 - ai_generated_percentage:.2f}%")

        return potential_issues, round(100.0 - ai_generated_percentage, 2)


    def identify_content_gaps(self, report_plan: ReportPlan, required_keywords: Optional[List[str]] = None) -> List[str]:
        """Checks if sections are missing content or if required keywords are absent."""
        logging.info("Checking for content gaps...")
        gaps = []

        def check_section_content(sections: List[ReportSection]):
            for section in sections:
                # Skip sections that are inherently content-light (like Bibliography placeholder)
                if section.title.lower() in ["bibliography", "appendices", "table of contents"]:
                    pass
                elif not section.content or len(section.content.strip()) < 50: # Arbitrary minimum length
                    gaps.append(f"Section '{section.title}' appears to have missing or very short content.")
                    logging.warning(f"Gap found: Section '{section.title}' seems empty.")
                # Check subsections
                check_section_content(section.subsections)

        check_section_content(report_plan.structure)

        # Check for required keywords if provided
        if required_keywords:
             full_report_text = ""
             def collect_text(sections: List[ReportSection]):
                  nonlocal full_report_text
                  for section in sections:
                      if section.content:
                          full_report_text += section.content + "\n"
                      collect_text(section.subsections)
             collect_text(report_plan.structure)

             for keyword in required_keywords:
                 if keyword.lower() not in full_report_text.lower():
                     gaps.append(f"Required keyword '{keyword}' seems to be missing from the report content.")
                     logging.warning(f"Gap found: Required keyword '{keyword}' missing.")

        logging.info(f"Content gap check finished. Found {len(gaps)} potential gaps.")
        return gaps