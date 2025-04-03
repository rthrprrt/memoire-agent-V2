# Apprenticeship Report Agent

This AI agent automates the creation of an apprenticeship report by leveraging daily journal entries stored as DOCX files. It utilizes the DeepSeek API for natural language processing to analyze content, track competencies, generate a structured report, create visualizations, and ensure quality.

## Features

- **DOCX Processing:** Parses daily journal entries.
- **Content Analysis:** Extracts key information, themes, and skills.
- **Automatic Tagging:** Generates relevant tags for each entry.
- **Vector Search:** Enables semantic search over journal content using ChromaDB.
- **Competency Mapping:** Tracks the development of predefined skills over time.
- **Report Planning:** Generates a structured report outline based on requirements.
- **AI-Powered Generation:** Uses the DeepSeek API to draft report sections from journal data.
- **Quality Assurance:** Checks for consistency, detects potential plagiarism (against journal entries), and identifies content gaps.
- **Visualizations:** Creates timelines and competency charts.
- **Harvard Referencing:** Manages citations and generates a bibliography (requires input for external sources).
- **Progress Tracking:** Monitors the completion status of the report.
- **Export:** Outputs the final report in DOCX and PDF formats.

---

## Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd apprenticeship-report-agent
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure:**
   - Rename `.env.example` to `.env`.
   - Edit the `.env` file and add your DeepSeek API key:
     ```env
     DEEPSEEK_API_KEY="your_deepseek_api_key_here"
     ```
   - Adjust paths in `config.py` if necessary (the defaults should work with the standard structure).

5. **Place Journal Entries:**
   - Put your daily journal DOCX files into the `journals/` directory.
   - Ensure the filenames ideally represent the date (e.g., `YYYY-MM-DD.docx`) for easier processing (the processor can be adapted if needed).

---

## Usage

The agent is controlled via command-line arguments.

### Common Commands

1. **Import & Process Journals:**
   ```bash
   python main.py process_journals
   ```

2. **Generate Report Plan:**
   ```bash
   python main.py create_plan --requirements_file path/to/your/thesis_structure.txt
   ```
   *(Note: Implement the `--requirements_file` logic or define a default structure.)*

3. **Generate Report Draft:**
   ```bash
   python main.py generate_report --plan_file output/report_plan.json --output_file output/report_draft.docx
   ```

4. **Run Quality Checks:**
   ```bash
   python main.py check_quality --report_file output/report_draft.docx
   ```

5. **Create Visualizations:**
   ```bash
   python main.py create_visuals
   ```

6. **Manage References (Example - Add a Reference):**
   ```bash
   # Example conceptual command - implementation needed:
   # python main.py add_reference --key "Smith2023" --type "book" --author "Smith, J." --year 2023 --title "AI in Business" --publisher "PubCo"
   ```

7. **Export Final Report (including Bibliography and PDF conversion):**
   ```bash
   python main.py export_report --input_docx output/report_draft.docx --output_pdf output/report_final.pdf
   ```

---

## Full Workflow Example

```bash
# 1. Process initial journals
python main.py process_journals

# 2. Create the report plan (review and edit output/report_plan.json manually if needed)
python main.py create_plan

# 3. Generate the first draft
python main.py generate_report --plan_file output/report_plan.json --output_file output/report_draft_v1.docx

# 4. Check the quality of the draft
python main.py check_quality --report_file output/report_draft_v1.docx

# 5. Generate supporting visuals
python main.py create_visuals

# (Optional: Add more journal entries and re-process)
# python main.py process_journals

# (Optional: Refine the report based on quality checks or new entries – requires manual editing or advanced agent features)

# 6. Export the final version
python main.py export_report --input_docx output/report_draft_v1.docx --output_pdf output/report_final.pdf