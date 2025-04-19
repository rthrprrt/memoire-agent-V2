# Apprenticeship Report Agent (memoire-agent-V2)

This AI agent aims to automate the creation of an MSc Apprenticeship Report ("Mission Professionnelle Digi5") using daily journal entries (DOCX) and official guidelines (PDF). It leverages local embeddings for context retrieval and Google's Gemini API (via LangChain) for reasoning, task execution, and content generation.

## Current Status & Features (Checkpoint: LangChain Agent Integrated - Debugging Workflow)

*   ✅ **DOCX/PDF Processing:** Parses journals (YYYY-MM-DD.docx format) and guidelines PDF.
*   ✅ **Local Embeddings:** Uses local Sentence Transformers (`paraphrase-multilingual-mpnet-base-v2`) via ChromaDB. Data remains local during vectorization.
*   ✅ **Vector Storage:** Separate ChromaDB collections (`journal_entries`, `reference_docs`) persisted locally (`vector_db/`).
*   ✅ **LLM:** Uses **Google Gemini API** (e.g., `gemini-1.5-flash`, via `langchain-google-genai`) for all LLM tasks (tags, skills, analysis, drafting, agent reasoning). Configured in `llm_interface.py`.
*   ✅ **Report Planning:** Generates `output/report_plan.json` with unique section IDs and status tracking (`report_planner.py`, `memory_manager.py`).
*   ✅ **LangChain Agent (`run_agent` command):**
    *   Uses `AgentExecutor` with a ReAct agent (`create_react_agent`).
    *   Includes `ConversationSummaryBufferMemory`.
    *   Tools defined as classes in `agent_tools.py` (search journals/guidelines, get/update plan, draft section). Tools use dependency injection.
    *   Agent can start, query plan status (`get_report_plan_structure`, `get_pending_sections`), and correctly identify when no sections are pending.
*   ⚠️ **Current Issue:** The agent fails when attempting to execute the `draft_single_section` tool. The tool internally reports "Error: Section ID '...' not found," even after the ID was correctly identified by `get_pending_sections`. This needs debugging within the `DraftSingleSectionTool._run` method in `agent_tools.py`.

## Phase 2 Roadmap: Autonomous Agentic Workflow

**Goal:** Achieve autonomous section-by-section report generation.

**Immediate Next Steps:**

1.  **[ ] Debug `DraftSingleSectionTool`:** Fix the "Section ID not found" error within the tool's `_run` method (likely in the internal `find_section` logic or plan loading).
2.  **[ ] Test Full Workflow:** After fixing the tool, run `run_agent --max_iterations <high_number>` with a reset plan (`create_plan`) to verify section-by-section drafting completion. Monitor Gemini API quota usage.
3.  **[ ] Implement Robust Memory:** If long runs hit context limits or the agent loses track, implement/configure memory summarization (e.g., refine `ConversationSummaryBufferMemory` settings or implement manual summarization).
4.  **[ ] Refine Agent Logic/Prompts:** Improve the ReAct prompt and potentially agent logic for better error handling, stopping conditions, and task sequencing.
5.  **[ ] Implement Reflection/Correction:** Add a step/tool for the agent to review drafted sections against guidelines before marking as "drafted".

**Later Steps:**

*   Integrate Year 1 data (summaries, specific tool/collection).
*   Implement user interaction tools (feedback, plan modification).
*   Address LangChain deprecation warnings (Pydantic v1/v2).

## Setup (Local Conda Environment)

1.  **Clone:** `git clone https://github.com/rthrprrt/memoire-agent-V2.git`
2.  **Conda Environment:** `conda create -n <name> python=3.10 -y`, `conda activate <name>`
3.  **Dependencies:** `pip install -r requirements.txt` (includes `langchain`, `google-generativeai`, `sentence-transformers`, etc.)
4.  **API Key:** Create `.env` file, add `GOOGLE_API_KEY="<your_google_ai_key>"`
5.  **Config:** Verify paths/models in `config.py`.
6.  **Data:** Place renamed journals (`YYYY-MM-DD.docx`) in `journals/`. Place guidelines PDF at path in `config.py`.
7.  **`.gitignore`:** Ensure `/.venv/`, `/output/`, `/vector_db/`, `/journals/`, `.env` are listed.

## Usage

*   **Setup Data:** `python main.py process_guidelines` then `python main.py process_journals`
*   **Create Plan:** `python main.py create_plan` (Resets statuses to 'pending')
*   **Run Agent:** `python main.py run_agent [--max_iterations N] [--objective "Goal"]`
*   **(Non-Agentic):** `python main.py generate_report` (Generates DOCX from plan, may need LLM refactor)
*   **(Other):** `check_quality`, `create_visuals`, `manage_refs`