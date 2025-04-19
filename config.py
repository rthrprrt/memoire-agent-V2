# config.py (Version avec Ollama Llama 3.1 8B INT8 par défaut)

import os
from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# --- Configuration des APIs ---

# --- Fournisseur LLM par défaut ---
# Choisir 'google' ou 'ollama'. Peut être surchargé par l'argument --llm en ligne de commande.
# Peut aussi être défini dans le fichier .env
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama") # <-- Mise à jour du défaut

# --- Configuration Google AI (Gemini) ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_CHAT_MODEL_NAME = os.getenv("GEMINI_CHAT_MODEL_NAME", "gemini-1.5-flash")

# --- Configuration Ollama (Local) ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# --- MODIFICATION : Modèle Ollama par défaut ---
# Nom du modèle Ollama à utiliser. Assurez-vous qu'il est téléchargé.
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "llama3.1:8b-instruct-q8_0")
# --- FIN MODIFICATION ---

# --- Configuration des Embeddings Locaux ---
LOCAL_EMBEDDING_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
LOCAL_EMBEDDING_DEVICE = "cpu"


# --- Configuration des Chemins de Fichiers ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JOURNAL_DIR = os.path.join(BASE_DIR, "journals")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_db")
DEFAULT_PLAN_FILE = os.path.join(OUTPUT_DIR, "report_plan.json")
DEFAULT_REPORT_OUTPUT = os.path.join(OUTPUT_DIR, "apprenticeship_report.docx")
DEFAULT_PDF_OUTPUT = os.path.join(OUTPUT_DIR, "apprenticeship_report.pdf")
GUIDELINES_PDF_PATH = os.path.join(BASE_DIR, "Mémoire_Alternance_Job.pdf")


# --- Configuration de la Base de Données Vectorielle ---
JOURNAL_DB_COLLECTION_NAME = "journal_entries"
REFERENCE_DOCS_COLLECTION_NAME = "reference_docs"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


# --- Configuration du Rapport ---
DATE_FILENAME_FORMAT = "%Y-%m-%d"
# !! IMPORTANT : Mettre à jour cette structure pour correspondre aux sections Epitech !!
DEFAULT_REPORT_STRUCTURE = [
    "Introduction",
    "Description de la mission",
    "   Contexte de l'entreprise (Gecina)",
    "   Mon rôle et mes objectifs",
    "Analyse des compétences",
    "Évaluation de la performance",
    "Réflexion personnelle et professionnelle",
    "Conclusion",
    "Bibliographie",
    "Annexes"
]

# Compétences clés à suivre/mapper (utilisées par CompetencyMapper)
COMPETENCIES_TO_TRACK = [
    "AI Model Implementation",
    "Data Analysis & Visualization",
    "Project Management (Agile/Scrum)",
    "Stakeholder Communication",
    "Problem Solving",
    "Business Acumen (Real Estate Sector)",
    "Python Programming",
    "Cloud Platforms (AWS/Azure/GCP)",
    "Ethical AI Considerations",
    "Prompt Engineering",
    "Power Platform Development"
]

# --- Mapping Collaborateurs (Pour Anonymisation) ---
# À remplir avec les vrais prénoms et les titres souhaités
COLLABORATOR_TITLES = {
    "Jérôme": "mon manager N+1",
    "Romain Hardy": "mon tuteur entreprise",
    "Alexandre": "le contact fonctionnel principal",
    "Clément Venard": "le responsable de la production",
    "Thierry": "le Directeur de l'Innovation",
    "Yann": "le DSI",
    # Ajoutez ici tous les prénoms mentionnés dans vos journaux
    # que vous souhaitez remplacer par un titre.
}


# --- Création des Dossiers de Sortie ---
try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
except OSError as e:
     print(f"AVERTISSEMENT: Impossible de créer les dossiers output/vector: {e}")


# --- Vérifications de Configuration Essentielles ---

if LLM_PROVIDER == "google" and not GOOGLE_API_KEY:
    raise ValueError("ERREUR FATALE: GOOGLE_API_KEY non trouvée dans le fichier .env et LLM_PROVIDER est 'google'. "
                     "Veuillez créer un fichier .env et ajouter votre clé API Google AI, ou choisir LLM_PROVIDER='ollama'.")

if not os.path.exists(GUIDELINES_PDF_PATH):
    print(f"AVERTISSEMENT: PDF Guidelines non trouvé : {GUIDELINES_PDF_PATH}")
    print(f"Assurez-vous que le fichier existe ou mettez à jour GUIDELINES_PDF_PATH dans config.py.")
    print("La commande 'process_guidelines' échouera, et les fonctionnalités liées aux guidelines seront limitées.")