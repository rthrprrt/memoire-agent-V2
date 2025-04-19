# quality_checker.py (Version confirmée avec injection LLM LangChain)

import config
from langchain_core.language_models.chat_models import BaseChatModel
from data_models import ReportPlan, ReportSection, JournalEntry
from document_processor import extract_text_from_docx
from vector_database import VectorDBManager
from typing import List, Dict, Optional, Tuple, Any
import logging
import difflib

log = logging.getLogger(__name__)
if not log.handlers: logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')

class QualityChecker:
    """Performs various quality checks on the generated report."""

    def __init__(self, vector_db: VectorDBManager, llm_instance: BaseChatModel):
        if not vector_db: raise ValueError("VectorDBManager instance is required.")
        if not llm_instance: raise ValueError("LLM instance is required.")
        self.vector_db = vector_db
        self.llm = llm_instance
        self.original_journal_texts: Dict[str, str] = {}
        log.info("QualityChecker initialized with provided VectorDB and LLM instances.")

    def load_journal_texts(self, entries: List[JournalEntry]):
        self.original_journal_texts = {entry.entry_id: entry.raw_text for entry in entries if hasattr(entry, 'entry_id') and hasattr(entry, 'raw_text')}
        log.info(f"Loaded raw text for {len(self.original_journal_texts)} entries.")

    def check_consistency_across_sections(self, report_plan: ReportPlan) -> List[str]:
        log.info("Checking consistency across sections (using LLM)...")
        issues = []; sections_with_content: List[ReportSection] = []
        def collect_content(sections: List[ReportSection]):
            if not sections: return
            for sec in sections:
                content = getattr(sec, 'content', None)
                if content and len(content) > 50: sections_with_content.append(sec)
                if hasattr(sec, 'subsections'): collect_content(sec.subsections)
        collect_content(report_plan.structure)
        if len(sections_with_content) < 2: log.info("Not enough content for consistency checks."); return issues
        # Example comparison (improve for production)
        project_sections = [s for s in sections_with_content if "project" in getattr(s, 'title', '').lower()]
        challenge_section = next((s for s in sections_with_content if "challenges" in getattr(s, 'title', '').lower()), None)
        if project_sections and challenge_section and hasattr(challenge_section, 'content'):
            for proj_sec in project_sections:
                 if not hasattr(proj_sec, 'content'): continue
                 aspect = f"details/timeline of {proj_sec.title}"
                 prompt = f"Analyze consistency between Seg1 and Seg2 regarding '{aspect}'. Point out discrepancies or confirm. Concise.\n\nSeg1:\n{proj_sec.content[:1000]}\n\nSeg2:\n{challenge_section.content[:1000]}"
                 try:
                     consistency_result_msg = self.llm.invoke(prompt)
                     consistency_result = getattr(consistency_result_msg, 'content', '')
                     if consistency_result and ("discrepancy" in consistency_result.lower() or "inconsistent" in consistency_result.lower()):
                          issue = f"Inconsistency '{proj_sec.title}' vs '{challenge_section.title}' ({aspect}): {consistency_result[:150]}..."
                          issues.append(issue); log.warning(issue)
                     elif consistency_result: log.info(f"Consistency OK: '{proj_sec.title}'.")
                     else: log.warning(f"Empty consistency check result for '{proj_sec.title}'.")
                 except Exception as e: log.error(f"LLM consistency check failed: {e}")
        log.info(f"Consistency check finished: {len(issues)} issues found."); return issues

    def check_plagiarism_against_journals(self, report_docx_path: str, similarity_threshold: float = 0.85) -> Tuple[List[str], float]:
        # ... (Logique difflib existante, pas de LLM ici) ...
        log.info("Checking plagiarism (difflib)..."); potential_issues = []; total_report_chars = 0; highly_similar_chars = 0
        try: report_text = extract_text_from_docx(report_docx_path)
        except Exception as e: log.error(f"Read error: {e}"); return [f"Read error: {e}"], 0.0
        if not report_text: log.warning("Report empty."); return [], 0.0
        total_report_chars = len(report_text)
        if not self.original_journal_texts: log.warning("Journals not loaded."); return [], 0.0
        # Basic sentence split/comparison (SLOW - needs optimization)
        report_sentences = [s.strip() for s in report_text.split('.') if len(s.strip()) > 20]
        journal_full_text = " ".join(self.original_journal_texts.values()); journal_sentences = {s.strip() for s in journal_full_text.split('.') if len(s.strip()) > 20}; matched_journal_indices = set()
        for i, rep_sent in enumerate(report_sentences):
            matcher = difflib.SequenceMatcher(None, rep_sent.lower())
            for j, jour_sent in enumerate(journal_sentences):
                 if j in matched_journal_indices: continue
                 matcher.set_seq2(jour_sent.lower())
                 if matcher.real_quick_ratio() > similarity_threshold and matcher.quick_ratio() > similarity_threshold and matcher.ratio() > similarity_threshold:
                    best_match_ratio = matcher.ratio(); issue = f"High similarity ({best_match_ratio:.2f}) report sent {i+1}: '{rep_sent[:80]}...'"; potential_issues.append(issue); highly_similar_chars += len(rep_sent); matched_journal_indices.add(j); break
        copied_percentage = (highly_similar_chars / total_report_chars * 100) if total_report_chars > 0 else 0.0
        log.info(f"Plagiarism check done: {len(potential_issues)} similar sentences. Est %: {copied_percentage:.1f}%"); return potential_issues, round(copied_percentage, 1)

    def identify_content_gaps(self, report_plan: ReportPlan, required_keywords: Optional[List[str]] = None) -> List[str]:
        log.info("Checking for content gaps..."); gaps = []; full_report_text = ""
        def check_section_content(sections: List[ReportSection]):
            nonlocal full_report_text; # ... (logique interne identique pour trouver sections vides et collecter texte) ...
        check_section_content(report_plan.structure)
        if required_keywords and full_report_text:
            log.info("Checking required keywords (LLM)..."); present_topics = set()
            prompt = f"""Review text. Which keywords are present? Output ONLY comma-separated list of PRESENT keywords.\nKeywords: {', '.join(required_keywords)}\nText:\n{full_report_text[:15000]}\nPresent Keywords:"""
            try:
                response_msg = self.llm.invoke(prompt) # Utilise l'instance injectée
                response = getattr(response_msg, 'content', '')
                if response: present_topics.update([kw.strip().lower() for kw in response.split(',') if kw.strip()])
            except Exception as e: log.error(f"LLM keyword check failed: {e}")
            missing_keywords_lower = {kw.lower() for kw in required_keywords} - present_topics
            if missing_keywords_lower:
                 original_case_missing = [kw for kw in required_keywords if kw.lower() in missing_keywords_lower]
                 for keyword in original_case_missing: gaps.append(f"Keyword missing: '{keyword}'."); log.warning(f"Gap: Keyword '{keyword}' missing.")
        log.info(f"Gap check finished: {len(gaps)} gaps."); return gaps