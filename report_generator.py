import config
from llm_interface import GeminiLLM
from vector_database import VectorDBManager
from data_models import ReportPlan, ReportSection, JournalEntry
from typing import List, Optional
import logging
from docx import Document # For creating the DOCX output
from docx.shared import Inches

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ReportGenerator:
    """Generates the report content section by section."""

    def __init__(self, vector_db: VectorDBManager):
        self.llm = GeminiLLM()
        self.vector_db = vector_db

    def _get_context_for_section(self, section_title: str, k: int = 7) -> List[str]:
        """Retrieves relevant text chunks from the vector database for a section title."""
        logging.debug(f"Searching vector DB for context related to: '{section_title}'")
        # Enhance query if needed (e.g., add keywords from parent sections)
        search_results = self.vector_db.search(query=section_title, k=k)
        context_chunks = [result['document'] for result in search_results]
        logging.debug(f"Retrieved {len(context_chunks)} context chunks.")
        return context_chunks

    def _generate_section_content(self, section: ReportSection, report_plan: ReportPlan) -> Optional[str]:
        """Generates content for a single section using LLM and context."""
        logging.info(f"Generating content for section: '{section.title}' (Level {section.level})")

        # --- Special Handling for certain sections ---
        if section.title.lower() == "bibliography":
             logging.info("Skipping generation for Bibliography section (handled separately).")
             return "(Bibliography content will be generated later)"
        if section.title.lower() == "introduction":
             # Might need broader context or specific instructions
             context_chunks = self._get_context_for_section(f"Overall apprenticeship summary, goals, {section.title}", k=10)
             instructions = "Write a compelling introduction outlining the report's purpose, the apprenticeship context (AI Project Officer at Gecina), and the report's structure."
        elif "project" in section.title.lower():
            # Be more specific in context search for projects
            context_chunks = self._get_context_for_section(f"Details about {section.title}, tasks, outcomes, AI usage", k=10)
            instructions = "Describe the project, focusing on objectives, methodology, your specific contributions (especially AI-related), challenges, and results. Use details from the journal context."
        elif "skills developed" in section.title.lower() or "competencies" in section.title.lower():
             context_chunks = self._get_context_for_section(f"Examples of {section.title}, learning experiences, technical skills, soft skills", k=10)
             instructions = f"Detail the key skills (like {', '.join(config.COMPETENCIES_TO_TRACK[:3])}...) developed during the apprenticeship, providing specific examples from the journal context."
        else:
            # Default context retrieval
            context_chunks = self._get_context_for_section(section.title, k=7)
            instructions = None # Use default prompt instructions

        if not context_chunks:
            logging.warning(f"No context found for section '{section.title}'. Generation might be poor.")
            # Optionally, try a broader search or skip generation
            # return None # Or attempt generation without specific context

        # Get overall structure for context
        all_section_titles = [s.title for s in report_plan.structure] # Top level only for brevity

        draft_content = self.llm.draft_report_section(
            section_title=section.title,
            context_chunks=context_chunks,
            report_structure=all_section_titles,
            instructions=instructions
        )

        if draft_content:
            logging.info(f"Successfully drafted content for '{section.title}'. Length: {len(draft_content)} chars.")
            # Basic post-processing (optional)
            # draft_content = draft_content.replace("...", " ") # Example cleanup
            return draft_content
        else:
            logging.error(f"Failed to generate content for section '{section.title}'.")
            return None


    def generate_full_report(self, report_plan: ReportPlan, output_docx_path: str = config.DEFAULT_REPORT_OUTPUT):
        """Generates content for all sections and saves to a DOCX file."""
        logging.info("Starting full report generation...")
        document = Document()
        document.add_heading(report_plan.title, level=0)

        # Store generated content back into the plan object temporarily
        generated_content_map = {} # title: content

        def process_section(section: ReportSection, current_level: int):
            nonlocal document
            content = self._generate_section_content(section, report_plan)
            if content:
                 generated_content_map[section.title] = content
                 section.content = content # Update the plan object directly if desired
                 section.status = "drafted"
                 try:
                     # Add heading (adjust level based on plan, max Word level is 9)
                     heading_level = min(section.level, 9)
                     if heading_level > 0: # Don't add heading for level 0 (report title)
                         document.add_heading(section.title, level=heading_level)
                     # Add content paragraph(s)
                     # Split content by newline to create paragraphs, but be careful with formatting
                     paragraphs = content.split('\n')
                     for para in paragraphs:
                         if para.strip(): # Avoid adding empty paragraphs
                             document.add_paragraph(para)
                     document.add_paragraph() # Add space after section
                 except Exception as e:
                      logging.error(f"Error adding section '{section.title}' to DOCX: {e}")
            else:
                section.status = "failed"
                logging.error(f"Skipping section '{section.title}' in DOCX due to generation failure.")


            for subsection in section.subsections:
                process_section(subsection, current_level + 1)

        # Iterate through the top-level sections
        for top_section in report_plan.structure:
            process_section(top_section, 1) # Start with level 1

        # TODO: Add Bibliography generation here using reference_manager
        # bib_content = ReferenceManager().generate_bibliography_text(citations)
        # document.add_heading("Bibliography", level=1)
        # document.add_paragraph(bib_content)

        # Save the document
        try:
            document.save(output_docx_path)
            logging.info(f"Report draft saved successfully to {output_docx_path}")
        except Exception as e:
            logging.error(f"Error saving DOCX file {output_docx_path}: {e}")

        return report_plan # Return the plan with updated statuses and content