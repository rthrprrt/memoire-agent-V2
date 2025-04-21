# Apprenticeship Report Agent (memoire-agent-V2)

This AI agent aims to automate the creation of an MSc Apprenticeship Report ("Mission Professionnelle Digi5") using daily journal entries (DOCX) and official guidelines (PDF). It leverages local embeddings for context retrieval and a configurable LLM (Google Gemini API via LangChain or a local Ollama model) for reasoning, task execution, and content generation.

## Current Status & Features (Checkpoint: Ollama Integrated - Agent Loop Failing)

*   ✅ **DOCX/PDF Processing:** Parses journals (YYYY-MM-DD.docx format) and guidelines PDF.
*   ✅ **Local Embeddings:** Uses local Sentence Transformers (`paraphrase-multilingual-mpnet-base-v2`) via ChromaDB. Data remains local during vectorization.
*   ✅ **Vector Storage:** Separate ChromaDB collections (`journal_entries`, `reference_docs`) persisted locally (`vector_db/`).
*   ✅ **Configurable LLM:** Supports **Google Gemini API** (via `langchain-google-genai`) and **local Ollama models** (via `langchain-community`). Choice via `config.py` or `--llm` argument. (`llm_interface.py`).
*   ✅ **Report Planning:** Generates `output/report_plan.json` with unique section IDs and status tracking (`report_planner.py`, `memory_manager.py`).
*   ✅ **LangChain Agent Framework:** Uses `AgentExecutor` with a ReAct agent (`create_react_agent`) and LangChain tools.
*   ✅ **Core Tools Functional:** Tools for searching vectors, getting/updating the plan, and drafting sections are implemented (`agent_tools.py`).
*   ✅ **Content Saving:** The `DraftSingleSectionTool` saves generated content to the `report_plan.json`.
*   ✅ **Report Assembly:** Command `assemble_report` generates DOCX from the completed plan JSON.
*   ✅ **Francisation (Code):** Agent objective, tool descriptions, and internal prompts are now in French.
*   ✅ **Gemini Workflow:** The agent workflow (Get Pending -> Draft -> Update Status -> Repeat) **works end-to-end using the Google Gemini API** (though subject to rate limits and requires content quality improvements).
*   🔴 **BLOCKING ISSUE (Ollama):** When using a local Ollama model (e.g., Llama 3.1 8B Instruct), the agent **fails to correctly follow the ReAct prompt format**. It gets stuck in loops, generating incorrectly formatted output (missing "Action:", irrelevant text) leading to parsing errors or premature termination. The standard ReAct prompt/AgentExecutor seems incompatible with this Ollama model for the sequential drafting task.
*   ⚠️ **Content Quality:** Generated content (both Gemini/Ollama) needs significant improvement: better alignment with Epitech guidelines, personalization (style, persona), and robust anonymization of sensitive data (names, opinions).

## Phase 3 Roadmap: Stabilize Ollama, Refine Content, Enhance Capabilities

**Goal:** Achieve reliable autonomous report generation using a local LLM, produce high-quality, personalized, and anonymized content in French, and add advanced features.

**Immediate Priority: Fix Ollama Agent Loop**
1.  [ ] **Fix ReAct Incompatibility:**
    *   **Option A (Preferred):** Develop a **custom ReAct prompt** specifically tailored for the Ollama model (Llama 3.1) and the sequential drafting task, emphasizing the required output format and the loop logic.
    *   **Option B (Alternative):** Refactor the agent logic using **LangGraph** to explicitly define the state machine (Get Pending -> Draft -> Update -> Loop/End), making the control flow less dependent on the LLM's interpretation of the ReAct prompt.

**Iteration 1: Stable Local Generation & Personalization** (Requires Fix above)
2.  [ ] **Define Persona/Profile:** Create a detailed author profile to inject into prompts.
3.  [ ] **Define Name->Title Mapping:** Finalize the `COLLABORATOR_TITLES` dictionary in `config.py`.
4.  [ ] **Enhance Drafting Prompt:** Integrate Persona, Style instructions (professional + journal inspiration), Anonymization rules (using mapping), Neutrality/Confidentiality rules into the `DraftSingleSectionTool` prompt.
5.  [ ] **Test & Iterate (Ollama):** Run the full workflow with the fixed agent (custom prompt or LangGraph) and the enhanced drafting prompt. Analyze output quality, rule adherence, and iterate on prompts/mappings.

**Iteration 2: Epitech Alignment & Basic Control & CI/CD**
6.  [ ] **Focus Epitech Guidelines:**
    *   [ ] Align `DEFAULT_REPORT_STRUCTURE` in `config.py` with the official Epitech PDF plan.
    *   [ ] Enhance the use of `SearchGuidelinesTool` in prompts/logic.
7.  [ ] **Simple Agent Status:** Create a `GetAgentStatusTool` to summarize progress from `report_plan.json`.
8.  [ ] **Setup CI/CD Pipeline (GitHub Actions):** Implement basic automated tests (linting, format, command checks).

**Iteration 3: External Capabilities**
9.  [ ] **Integrate Web Search (e.g., BraveSearch):** Add tool and update prompts/logic.
10. [ ] **Integrate Bibliography Management (Basic):** Add `AddCitationTool`, update `assemble_report`, prompt agent for usage.

**Iteration 4: Advanced Interaction & Architecture**
11. [ ] **(If not done in Priority 1):** Implement agent using **LangGraph** for robust flow control.
12. [ ] **Chat/RAG Interface:** Develop interactive capabilities.

## Setup (Local Conda Environment)

1.  **Clone:** `git clone https://github.com/rthrprrt/memoire-agent-V2.git`
2.  **Conda Environment:** `conda create -n <name> python=3.10 -y`, `conda activate <name>`
3.  **Dependencies:** `pip install -r requirements.txt`
4.  **API Key (Optional - for Gemini):** Create `.env` file, add `GOOGLE_API_KEY="<your_google_ai_key>"`
5.  **Config:** Verify paths/models in `config.py`. Set `LLM_PROVIDER="ollama"` and the correct `OLLAMA_MODEL_NAME` (e.g., `llama3.1:8b-instruct-q8_0`). Fill `COLLABORATOR_TITLES`. Update `DEFAULT_REPORT_STRUCTURE`.
6.  **Data:** Place renamed journals (`YYYY-MM-DD.docx`) in `journals/`. Place guidelines PDF at path in `config.py`.
7.  **`.gitignore`:** Ensure `/.venv/`, `/output/`, `/vector_db/`, `/journals/`, `.env` are listed.
8.  **Ollama:** Install Ollama ([https://ollama.com/](https://ollama.com/)) and download the chosen model (e.g., `ollama pull llama3.1:8b-instruct-q8_0`). Ensure Ollama service is running.

## Usage

*   **Setup Data:** `python main.py process_guidelines` then `python main.py process_journals`
*   **Create/Reset Plan:** `python main.py create_plan`
*   **Run Agent (Ollama):** (Set env vars first if needed: `OLLAMA_GPU_LAYERS`, `OLLAMA_KEEP_ALIVE`) `python main.py run_agent --llm ollama [--max_iterations N]`
*   **Run Agent (Gemini):** `python main.py run_agent --llm google [--max_iterations N]`
*   **Assemble Report:** `python main.py assemble_report [--plan_file <path>] [--output_file <path>]`
*   **(Other):** `check_quality`, `create_visuals`, `manage_refs`