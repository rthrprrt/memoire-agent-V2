import os
from dotenv import load_dotenv

load_dotenv() # Load variables from .env file

# --- API Configuration ---
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_BASE = "https://api.deepseek.com" # Or /beta if needed
# Consider adding models if you switch between chat/coder etc.
DEEPSEEK_CHAT_MODEL = "deepseek-chat"
# DEEPSEEK_EMBEDDING_MODEL = "text-embedding-v2" # Check DeepSeek's embedding model name if they offer one, otherwise use a compatible one via API or a local library

# --- File Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JOURNAL_DIR = os.path.join(BASE_DIR, "journals")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_db")
DEFAULT_PLAN_FILE = os.path.join(OUTPUT_DIR, "report_plan.json")
DEFAULT_REPORT_OUTPUT = os.path.join(OUTPUT_DIR, "apprenticeship_report.docx")
DEFAULT_PDF_OUTPUT = os.path.join(OUTPUT_DIR, "apprenticeship_report.pdf")

# --- Vector Database Configuration ---
VECTOR_DB_COLLECTION_NAME = "journal_entries"
CHUNK_SIZE = 1000 # Characters or tokens - adjust based on model context window and desired granularity
CHUNK_OVERLAP = 150 # Overlap between chunks

# --- Report Generation ---
# Define your MSc thesis structure requirements here or load from a file
# Example:
DEFAULT_REPORT_STRUCTURE = [
    "Introduction",
    "Company Context (Gecina)",
    "Apprenticeship Role & Objectives",
    "Methodology",
    "Projects Undertaken",
    "   Project A: Description & AI Application",
    "   Project B: Description & AI Application",
    "Skills Developed (Technical & Soft)",
    "Challenges & Solutions",
    "Learning Outcomes & Reflection",
    "Conclusion",
    "Bibliography",
    "Appendices (Optional)"
]

# Define competencies to track (align with your MSc program/role)
# Example:
COMPETENCIES_TO_TRACK = [
    "AI Model Implementation",
    "Data Analysis & Visualization",
    "Project Management (Agile/Scrum)",
    "Stakeholder Communication",
    "Problem Solving",
    "Business Acumen (Real Estate Sector)",
    "Python Programming",
    "Cloud Platforms (AWS/Azure/GCP)",
    "Ethical AI Considerations"
]


# --- Other Constants ---
DATE_FILENAME_FORMAT = "%Y-%m-%d" # Expected format if date is in filename

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

# --- Error Handling ---
if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY not found in .env file. Please set it.")

# ++ AJOUT POUR EMBEDDINGS LOCAUX ++
# Modèle Sentence Transformer choisi (multilingue, équilibré)
LOCAL_EMBEDDING_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
# Spécifier 'cpu' ou 'cuda' si vous avez un GPU Nvidia compatible et PyTorch installé avec CUDA
# Commencer par 'cpu' est plus sûr si vous n'êtes pas sûr de votre config GPU.
LOCAL_EMBEDDING_DEVICE = "cpu"
# ++ FIN AJOUT ++