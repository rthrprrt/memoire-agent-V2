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

The revised Phase 2 will focus on enhancing the agent's capabilities in knowledge representation, reasoning, and interaction, while prioritizing code stability and maintainability. The key areas of development are:

1.  **Agent Architecture and Communication:** Formalize the communication between the Orchestrator and Worker agents using structured message schemas.
2.  **Knowledge Representation and Reasoning:** Explore and implement enhanced knowledge representation techniques (e.g., semantic networks, ontologies) and integrate a reasoning engine for more intelligent processing of journal information.
3.  **Iterative Report Generation and Feedback:** Develop mechanisms for more targeted and informative feedback loops between the agents during report drafting and revision.
4.  **Modularity and Extensibility:** Design the system with a focus on modularity, potentially using a microservices architecture or a plugin system for future expansion.
5.  **User-Centric Design and Explainability:** Prioritize transparency and user control by providing clear explanations of the agent's reasoning and offering customization options.

**Prioritized Initial Steps:**

*   Set up a CI/CD pipeline with automated testing and code quality checks.
*   Implement formal message schemas and handling for agent communication.
*   Conduct an initial exploration of semantic networks and ontologies for knowledge representation.

## Setup

1.  **Environment:** Create Conda env: `conda create -n apprenticeship-agent-env python=3.10 -y`, then `conda activate apprenticeship-agent-env`.
2.  **Dependencies:** `pip install -r requirements.txt` (Installs `pypdf2`, `google-generativeai`, `sentence-transformers`, `torch`, etc. Removes `ollama` if present).
3.  **API Keys & Config:**
    *   Create `.env` file.
    *   Add `GOOGLE_API_KEY="<your_google_ai_key>"`
    *   Add `DEEPSEEK_API_KEY="<your_deepseek_key>"` (Needed for Phase 2 Orchestrator).
    *   Verify paths and model names in `config.py` (especially `GUIDELINES_PDF_PATH`, `LOCAL_EMBEDDING_MODEL_NAME`, `GEMINI_CHAT_MODEL_NAME`).
5.  **Journals:** Place `.docx` files named `YYYY-MM-DD.docx` in the `journals/` directory.
6.  **Guidelines PDF:** Place the official guidelines PDF at the location specified by `GUIDELINES_PDF_PATH` in `config.py`.
7.  **(Phase 2 Prep):** Download Ollama (if using for local testing/comparison) and pull models (`ollama pull mistral`). *Note: Ollama is no longer the primary LLM in the Phase 2 plan.*

1.  **Clone:** `git clone <your-repo-url>`
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