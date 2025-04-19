# Apprenticeship Report Agent (memoire-agent-V2)

This AI agent aims to automate the creation of an MSc Apprenticeship Report ("Mission Professionnelle Digi5") using daily journal entries (DOCX) and official guidelines (PDF). It leverages local embeddings for context retrieval and Google's Gemini API (via LangChain) for reasoning, task execution, and content generation.

## Current Status & Features (Checkpoint: Agent Workflow Functional - Content Saved)

*   ✅ **DOCX/PDF Processing:** Parses journals (YYYY-MM-DD.docx format) and guidelines PDF.
*   ✅ **Local Embeddings:** Uses local Sentence Transformers (`paraphrase-multilingual-mpnet-base-v2`) via ChromaDB. Data remains local during vectorization.
*   ✅ **Vector Storage:** Separate ChromaDB collections (`journal_entries`, `reference_docs`) persisted locally (`vector_db/`).
*   ✅ **LLM:** Uses **Google Gemini API** (e.g., `gemini-1.5-flash`, via `langchain-google-genai`) for all LLM tasks. Configured in `llm_interface.py`.
*   ✅ **Report Planning:** Generates `output/report_plan.json` with unique section IDs and status tracking (`report_planner.py`, `memory_manager.py`).
*   ✅ **LangChain Agent (`run_agent` command):**
    *   Uses `AgentExecutor` with a ReAct agent (`create_react_agent`).
    *   Includes `ConversationSummaryBufferMemory`.
    *   Tools defined as classes in `agent_tools.py` (search journals/guidelines, get/update plan, draft section). Tools use dependency injection.
    *   Agent successfully executes the full drafting workflow section by section.
    *   **Content Saving:** The `DraftSingleSectionTool` now correctly saves the generated content back into the `report_plan.json` file.
*   ✅ **Report Assembly:** A new command `assemble_report` reads the completed `report_plan.json` (with content) and generates the final DOCX report.
*   ⚠️ **Current Issue:** The agent frequently hits the Google Gemini API rate limits (Error 429) when running the full workflow due to multiple sequential API calls (agent thought, tool execution, memory). A temporary `time.sleep(15)` has been added to mitigate this, but a more robust solution might be needed.
*   ⚠️ **Content Quality/Language:** The generated content is currently in English and needs significant improvement in quality and alignment with specific Epitech requirements.

## Phase 3 Roadmap: Francisation, Local LLM, and Enhanced Capabilities

**Goal:** Refine the agent to produce a high-quality report draft in French, offer local execution capabilities, integrate external tools, and improve user control.

**Iteration 1: Francisation and Local LLM Base**
1.  [ ] **Franciser les Prompts:** Translate/adapt all system prompts, tool descriptions, and internal tool prompts to French.
2.  [ ] **Franciser l'Objectif:** Update the default objective in `main.py` to explicitly target the Epitech report in French.
3.  [ ] **Test French Generation (Gemini):** Verify the agent produces French content.
4.  [ ] **Integrate Ollama:**
    *   [ ] Install Ollama & download a suitable French model (e.g., Mistral, Llama 3 Instruct).
    *   [ ] Add `get_ollama_llm()` in `llm_interface.py`.
    *   [ ] Add `--llm ollama` option to `main.py`.
    *   [ ] Test `run_agent --llm ollama` and compare results.

**Iteration 2: Content Improvement & Simple Control**
5.  [ ] **Focus Epitech Guidelines:**
    *   [ ] Align `DEFAULT_REPORT_STRUCTURE` in `config.py` with the official Epitech PDF plan.
    *   [ ] Enhance the use of `SearchGuidelinesTool` in prompts.
6.  [ ] **Improve Content Quality (Iterative):** Analyze French outputs, refine `DraftSingleSectionTool` prompts.
7.  [ ] **Simple Agent Status:** Create a `GetAgentStatusTool` to summarize progress from `report_plan.json`.
8.  [ ] **Setup CI/CD Pipeline (GitHub Actions):** Implement basic automated tests (e.g., linting, running commands with dummy data) to ensure non-regression. *(Prioritized here)*

**Iteration 3: External Capabilities**
9.  [ ] **Integrate Web Search (e.g., BraveSearch):**
    *   [ ] Get API key.
    *   [ ] Create/Use LangChain tool (`BraveSearchTool`).
    *   [ ] Add tool to agent and update prompts.
10. [ ] **Integrate Bibliography Management (Basic):**
    *   [ ] Create `AddCitationTool(ref_manager)`.
    *   [ ] Modify `assemble_report` to call `ref_manager.generate_bibliography_text()`.
    *   [ ] *Challenge:* Prompt the agent to use these tools effectively.

**Iteration 4: Advanced Interaction & Architecture**
11. [ ] **Explore LangGraph:** Consider replacing `AgentExecutor` with LangGraph for finer control over the agent's flow (enabling easier start/stop, validation steps, complex loops).
12. [ ] **Chat/RAG Interface:** Develop an interactive loop in `main.py` allowing users to query the agent's status/memory and issue specific commands (e.g., "revise section X").

## Setup (Local Conda Environment)

1.  **Clone:** `git clone https://github.com/rthrprrt/memoire-agent-V2.git`
2.  **Conda Environment:** `conda create -n <name> python=3.10 -y`, `conda activate <name>`
3.  **Dependencies:** `pip install -r requirements.txt`
4.  **API Key:** Create `.env` file, add `GOOGLE_API_KEY="<your_google_ai_key>"`
5.  **Config:** Verify paths/models in `config.py`.
6.  **Data:** Place renamed journals (`YYYY-MM-DD.docx`) in `journals/`. Place guidelines PDF at path in `config.py`.
7.  **`.gitignore`:** Ensure `/.venv/`, `/output/`, `/vector_db/`, `/journals/`, `.env` are listed.
8.  **(Optional for Local LLM):** Install Ollama ([https://ollama.com/](https://ollama.com/)) and download a model (e.g., `ollama pull mistral`).

## Usage

*   **Setup Data:** `python main.py process_guidelines` then `python main.py process_journals`
*   **Create/Reset Plan:** `python main.py create_plan`
*   **Run Agent (Gemini):** `python main.py run_agent [--max_iterations N]`
*   **Run Agent (Ollama - after It. 1):** `python main.py run_agent --llm ollama [--max_iterations N]`
*   **Assemble Report:** `python main.py assemble_report [--plan_file <path>] [--output_file <path>]`
*   **(Other):** `check_quality`, `create_visuals`, `manage_refs`