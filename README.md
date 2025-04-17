# Apprenticeship Report Agent (memoire-agent-V2)

This AI agent aims to automate the creation of an MSc Apprenticeship Report ("Mission Professionnelle Digi5") using daily journal entries (DOCX) and official guidelines (PDF). It leverages local embeddings for context retrieval and Google's Gemini API for reasoning, task execution, and content generation, while prioritizing data privacy during vectorization.

## Current Status & Features (Checkpoint: Post-Initial Agent Loop Test)

*   ✅ **DOCX Processing:** Parses daily journal entries (expects `YYYY-MM-DD.docx` format, uses `python-docx`).
*   ✅ **PDF Guideline Processing:** Extracts text from the official report guidelines PDF (uses `pypdf2`).
*   ✅ **Local Embeddings:** Uses local Sentence Transformers (`paraphrase-multilingual-mpnet-base-v2` by default via `sentence-transformers` library) to generate text embeddings. **Journal/Guideline text does NOT leave the local machine during embedding.**
*   ✅ **Vector Storage:** Stores text chunks and their local embeddings in ChromaDB (`chromadb` library), maintaining separate collections for `journal_entries` and `reference_docs`. Persisted locally in the `vector_db/` directory.
*   ✅ **LLM Integration:** Uses **Google's Gemini API** (via `google-generativeai` library, configured with `GeminiLLM` class in `llm_interface.py`) for:
    *   Generating tags (`tag_generator.py`).
    *   Mapping competencies (`competency_mapper.py`).
    *   Analyzing content (`content_analyzer.py`).
    *   Generating report sections (`report_generator.py`).
    *   Powering the agent loop (`main.py`).
*   ✅ **Report Planning:** Generates an initial structured report plan (`report_plan.json`) with unique section IDs based on a predefined structure (`report_planner.py`, `data_models.py`).
*   ✅ **Basic Agentic Workflow (`run_agent` command):**
    *   An experimental loop where the Gemini LLM can interact with tools.
    *   Uses a text-based format (`>>>TOOL_CALL...`) for tool invocation.
    *   Successfully calls implemented tools (`search_guidelines`, `search_journal_entries`, `get_report_plan_structure`, `get_pending_sections`, `update_section_status`) via `agent_tools.py`.
    *   Improved argument parsing using `ast.literal_eval`.
    *   Basic handling of Gemini API conversation history requirements.
*   ✅ **Supporting Modules:** Includes components for visualization (`matplotlib`), reference management (`references.json`), progress tracking (basic), and memory management (in-memory for now).
*   ✅ **Data Privacy:** Embeddings are generated locally. LLM interactions send necessary context (journal snippets, guidelines, prompts) to Google AI API, which is the user's accepted configuration for this phase.
*   ✅ **Environment Setup:** Configured for local execution using Conda/venv and Nix (via `.idx/dev.nix` for Firebase Studio compatibility, although local execution is now prioritized).

## Phase 2 Roadmap: Autonomous Agentic Workflow

The next phase focuses on enhancing the agent's autonomy and implementing a more robust workflow driven by the LLM.

**Goal:** Enable the agent to independently manage the report writing process from planning to drafting, using its tools effectively based on the overall objective and the report plan.

**Key Development Steps:**

1.  **[ ] Implement Remaining Core Tools:**
    *   Ensure robustness of `search_journal_entries`.
    *   Implement tools for Year 1 data retrieval (requires data preparation first).
    *   Refine `update_section_status` (consider state management beyond simple file save/load).
2.  **[X] Robust Tool Argument Parsing:** Implemented using `ast.literal_eval`. *(Marked as done based on recent implementation)*.
3.  **[ ] Develop Planning & Execution Capabilities:**
    *   Refactor `run_agent` loop (or create `AgentRunner` class).
    *   **Objective:** Give the agent the high-level goal "Generate the full report".
    *   **Logic:**
        *   Agent uses `get_report_plan_structure` and `get_pending_sections` to identify the next section to work on.
        *   Agent uses `search_guidelines` and `search_journal_entries` (and potentially Year 1 search) to gather context for that section.
        *   Agent calls a dedicated tool/function `draft_single_section(section_id, context)` which uses the LLM (`GeminiLLM.draft_report_section`) to generate the text.
        *   Agent uses `update_section_status` to mark the section as `drafted` or `failed`.
        *   Agent loops until `get_pending_sections` returns no sections.
4.  **[ ] Improve Memory Management:**
    *   Implement context window management for `conversation_history` (e.g., sliding window, summarization) to handle long processes.
    *   Refactor `agent_tools` to avoid creating separate `MemoryManager` instances; pass the main instance or use a shared state mechanism.
5.  **[ ] Implement Basic Reflection/Correction Loop:**
    *   After drafting a section, add a step where the agent evaluates the draft against guidelines (`search_guidelines`) and potentially its own internal checklist (via LLM prompt).
    *   If issues are found, the agent attempts to redraft the section with corrective instructions.
6.  **[ ] Integrate Year 1 Data:**
    *   User prepares summary documents for Year 1.
    *   Implement `process_year1_docs` command (similar to guidelines/journals).
    *   Implement `search_year1_summary` tool.
    *   Adapt planning/drafting logic to incorporate Year 1 context where relevant.

## Setup (Local Conda Environment - Recommended)

1.  **Clone:** `git clone https://github.com/rthrprrt/memoire-agent-V2.git`
2.  **Conda Environment:**
    *   `conda create -n apprenticeship-agent-env python=3.10 -y` (or 3.11)
    *   `conda activate apprenticeship-agent-env`
3.  **Dependencies:** `pip install -r requirements.txt`
4.  **API Keys & Config:**
    *   Create `.env` file in the root directory.
    *   Add `GOOGLE_API_KEY="<your_google_ai_key_from_ai_studio>"`
    *   Verify paths and model names in `config.py` (especially `GUIDELINES_PDF_PATH`, `LOCAL_EMBEDDING_MODEL_NAME`, `GEMINI_CHAT_MODEL_NAME`).
5.  **Journals:** Place `.docx` files named `YYYY-MM-DD.docx` in the `journals/` directory (this directory is ignored by git).
6.  **Guidelines PDF:** Place the official guidelines PDF (e.g., `Mémoire_Alternance_Job.pdf`) at the location specified by `GUIDELINES_PDF_PATH` in `config.py`.
7.  **`.gitignore`:** Ensure `.gitignore` includes `/.venv/`, `/output/`, `/vector_db/`, `/journals/`, `.env`, etc.

## Usage

**Core Workflow Commands (Run in order initially):**

1.  **(Optional, Run Once if needed):** `python rename_journals.py` - Renames journals from French date format.
2.  `python main.py process_guidelines [--reprocess]` - Processes PDF guidelines into vector DB.
3.  `python main.py process_journals [--reprocess_all]` - Processes DOCX journals into vector DB, calls Gemini for tags/competencies.
4.  `python main.py create_plan` - Generates `output/report_plan.json` with section IDs.
5.  `python main.py generate_report` - Generates full draft DOCX using Gemini (less agentic).
6.  `python main.py run_agent [--objective "Your Goal"] [--max_turns N]` - Runs the experimental agent loop.

**Other Commands:**

*   `python main.py check_quality`: Runs basic quality checks.
*   `python main.py create_visuals`: Generates timeline plots.
*   `python main.py manage_refs [add|list] [options...]`: Manages references.
*   `python main.py run_all [--reprocess] [--skip_guidelines]`: Attempts the full pipeline sequentially.

## Future Expansion (Beyond Phase 2)

*   Knowledge Graph integration.
*   Web search tool integration.
*   Advanced memory techniques.
*   GUI/Web interface (Streamlit/Flask).
*   More sophisticated error handling and recovery.
*   Fine-tuning embedding/LLM models.