# Apprenticeship Report Agent (memoire-agent-V2)

This AI agent aims to automate the creation of an MSc Apprenticeship Report using daily journal entries (DOCX) and official guidelines (PDF). It leverages local embeddings for context retrieval and a hybrid AI approach for reasoning, task execution, and content generation, prioritizing data privacy for sensitive journal content.

## Current Status & Features (End of Phase 1 - Foundations)

*   ✅ **DOCX Processing:** Parses daily journal entries (requires `YYYY-MM-DD.docx` format).
*   ✅ **PDF Guideline Processing:** Extracts text from the official report guidelines PDF.
*   ✅ **Local Embeddings:** Uses local Sentence Transformers (`paraphrase-multilingual-mpnet-base-v2` by default) via ChromaDB for vectorizing journal entries and guidelines. **Journal/Guideline text does NOT leave the local machine during embedding.**
*   ✅ **Dual Vector Stores:** Maintains separate ChromaDB collections for `journal_entries` and `reference_docs`.
*   ✅ **Metadata Extraction (via LLM):** Can generate tags and map competencies by sending *portions* of journal text to an LLM API (currently configured for local Ollama/Mistral for privacy).
*   ✅ **Content Generation (via LLM):** Can generate report sections by retrieving relevant local context (embeddings) and sending it *along with prompts* to an LLM API (currently configured for local Ollama/Mistral).
*   ✅ **Basic Agent Loop:** Experimental `run_agent` command allows an LLM to interact with defined tools (`search_journal_entries`, `search_guidelines`) via text-based calls.
*   ✅ **Privacy-Focused LLM Calls:** Current LLM calls (tags, competencies, generation) are configured to use a **local Ollama model**, ensuring no journal content is sent to external APIs during these steps.
*   ✅ **Supporting Modules:** Includes components for planning, quality checks (basic), visualization (basic), reference management, etc.

## Phase 2 Roadmap: Towards an Autonomous Agentic Workflow (Hybrid Model)

The next phase focuses on transforming the current pipeline into a more autonomous agent using a sophisticated hybrid AI architecture:

**Goal:** Create an agent capable of planning, executing, and reflecting on the report writing process, leveraging the strengths of different LLMs while ensuring sensitive journal data remains secure.

**Proposed Hybrid Architecture:**

1.  **Orchestrator / Reasoner / Guidelines Guardian:** `deepseek-reasoner` (via API)
    *   **Role:** High-level planning, task decomposition, checking consistency against guidelines, evaluating logical flow. Utilizes its Chain-of-Thought (CoT) capabilities.
    *   **Data Access:** Receives **only** non-sensitive data: prompts, objectives, metadata, guidelines content, *summarized/anonymized* results from the worker LLM. **Never sees raw journal content.**
2.  **Worker / Redactor / Data Processor:** `gemini-pro` (via Google AI Free Tier API)
    *   **Role:** Executes specific tasks requiring access to detailed context: generating tags/competencies from raw text, searching journals, drafting report sections based on retrieved chunks, summarizing journal content.
    *   **Data Access:** Receives raw text chunks retrieved locally from ChromaDB, specific instructions from the Orchestrator (relayed by Python code). Sends potentially sensitive snippets to Google AI API.
3.  **Embeddings:** Local Sentence Transformers (via ChromaDB). Unchanged.
4.  **Python Intermediary (Agent Core Logic):** Acts as the bridge and filter.
    *   Receives high-level instructions/tool calls from DeepSeek-Reasoner.
    *   Determines if sensitive data is needed.
    *   If yes: retrieves data locally, calls Gemini API for the specific task.
    *   If no (or after Gemini call): potentially processes/summarizes/anonymizes Gemini's output before returning a result/status update to DeepSeek-Reasoner.
    *   Executes local tools (like guideline search) directly when requested by DeepSeek-Reasoner.

**Key Development Steps for Phase 2:**

1.  **[ ] Re-integrate DeepSeek API:**
    *   Update `config.py` and `.env` for `DEEPSEEK_API_KEY`.
    *   Update `llm_interface.py` to include a class/method for calling `deepseek-reasoner` specifically. Keep or adapt the `GeminiLLM` class for the worker role.
2.  **[ ] Refine Tool Calling Mechanism:**
    *   Solidify the text-based format (`>>>TOOL_CALL...`) or investigate alternative reliable methods for DeepSeek-Reasoner to trigger tools.
    *   Improve the argument parsing (`parse_kwargs` in `main.py` or a dedicated function).
3.  **[ ] Implement Core Agent Loop (Hybrid Logic):**
    *   Refactor the `run_agent` command in `main.py` (or move to `agent_runner.py`).
    *   The loop should be driven by `deepseek-reasoner`.
    *   Implement the logic where Python code intercepts DeepSeek's requests:
        *   Calls local tools directly (guideline search).
        *   Calls Gemini API via `GeminiLLM` when journal content is needed for a task requested by DeepSeek.
        *   Processes Gemini's output before potentially sending a summary/status back to DeepSeek.
4.  **[ ] Enhance `agent_tools.py`:**
    *   Implement more tools: `get_report_plan_structure`, `get_pending_sections`, `update_section_status`.
    *   Create the `gather_context_for(section_title)` tool that orchestrates local searches (journals, guidelines, year 1 summaries) before potentially calling Gemini for drafting.
5.  **[ ] Develop Planning & Execution Capabilities:**
    *   Enable the DeepSeek orchestrator to:
        *   Load/understand the report plan (`report_plan.json`).
        *   Iterate through sections (`get_pending_sections`).
        *   Request context gathering and drafting for each section via tools.
        *   Track progress (`update_section_status`).
6.  **[ ] Implement Basic Reflection/Correction Loop:**
    *   After a section is drafted (by Gemini), have DeepSeek evaluate it against guidelines (using `search_guidelines`) and its internal reasoning (CoT).
    *   If issues are found, DeepSeek requests a revision (triggering another call to Gemini with refined instructions).
7.  **[ ] Integrate Year 1 Data:**
    *   Prepare summary documents for Year 1.
    *   Create a new ChromaDB collection (`year1_summaries`).
    *   Implement `process_year1_docs` command.
    *   Implement `search_year1_summary` tool.
8.  **[ ] Improve Memory Management:**
    *   Address context window limits for DeepSeek and Gemini (summarization, windowing).
    *   Enhance `MemoryManager` for better short-term state tracking if needed.

## Setup

1.  **Clone:** `git clone <your-repo-url>`
2.  **Environment:** Create Conda env: `conda create -n apprenticeship-agent-env python=3.10 -y`, then `conda activate apprenticeship-agent-env`.
3.  **Dependencies:** `pip install -r requirements.txt` (Installs `pypdf2`, `google-generativeai`, `sentence-transformers`, `torch`, etc. Removes `ollama` if present).
4.  **API Keys & Config:**
    *   Create `.env` file.
    *   Add `GOOGLE_API_KEY="<your_google_ai_key>"`
    *   Add `DEEPSEEK_API_KEY="<your_deepseek_key>"` (Needed for Phase 2 Orchestrator).
    *   Verify paths and model names in `config.py` (especially `GUIDELINES_PDF_PATH`, `LOCAL_EMBEDDING_MODEL_NAME`, `GEMINI_CHAT_MODEL_NAME`).
5.  **Journals:** Place `.docx` files named `YYYY-MM-DD.docx` in the `journals/` directory.
6.  **Guidelines PDF:** Place the official guidelines PDF at the location specified by `GUIDELINES_PDF_PATH` in `config.py`.
7.  **(Phase 2 Prep):** Download Ollama (if using for local testing/comparison) and pull models (`ollama pull mistral`). *Note: Ollama is no longer the primary LLM in the Phase 2 plan.*

## Usage (Current & Planned)

**Current Core Commands:**

*   `python rename_journals.py`: (Run once) Renames journal files from French date format to YYYY-MM-DD.
*   `python main.py process_journals [--reprocess_all]`: Processes DOCX journals, generates embeddings locally, stores in DB.
*   `python main.py process_guidelines [--reprocess]`: Processes the guidelines PDF, generates embeddings locally, stores in reference DB.
*   `python main.py create_plan`: Generates `output/report_plan.json`.
*   `python main.py generate_report`: Generates `output/apprenticeship_report.docx` using *current* LLM config (Ollama/local). **Will be updated in Phase 2 to use Gemini worker.**
*   `python main.py run_agent [--objective ...] [--max_turns ...]`: Runs the *experimental* agent loop using *current* LLM config (Ollama/local). **Will be updated in Phase 2 for hybrid model.**
*   `python main.py check_quality`: Runs basic checks on the generated report.
*   `python main.py create_visuals`: Generates timeline plots.
*   `python main.py manage_refs [add|list] [options...]`: Manages references.

**Phase 2 Target Command:**

*   `python main.py run_agent --objective "Generate the full apprenticeship report according to guidelines"`: (Future Goal) Launches the autonomous workflow driven by DeepSeek-Reasoner and Gemini.

## Future Expansion (Beyond Phase 2)

*   Knowledge Graph integration for relationship analysis.
*   Web search tool for external references.
*   More sophisticated memory techniques (vectorized conversation history).
*   GUI/Web interface (Streamlit/Flask).
*   Robust error handling and retry logic for API calls/tools.