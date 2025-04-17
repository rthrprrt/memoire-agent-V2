# config.py (Version pour Embeddings Locaux + Gemini API pour LLM)

import os
from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# --- Configuration des APIs ---

# Clé API Google AI (Gemini) - Récupérée depuis .env
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# Nom du modèle Gemini à utiliser pour le chat, la génération, l'analyse, etc.
# Ex: "gemini-1.0-pro", "gemini-1.5-flash" (vérifier la disponibilité/recommendations)
GEMINI_CHAT_MODEL_NAME = "gemini-1.0-pro"

# Clé API DeepSeek (Optionnel - Gardé pour référence ou usage futur potentiel)
# DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
# DEEPSEEK_API_BASE = "https://api.deepseek.com"
# DEEPSEEK_CHAT_MODEL = "deepseek-chat"


# --- Configuration des Embeddings Locaux ---

# Modèle Sentence Transformer à utiliser (depuis Hugging Face)
# 'paraphrase-multilingual-mpnet-base-v2' est un bon choix multilingue équilibré.
LOCAL_EMBEDDING_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
# Appareil pour l'inférence ('cpu' ou 'cuda' si GPU Nvidia disponible/configuré)
LOCAL_EMBEDDING_DEVICE = "cpu"


# --- Configuration des Chemins de Fichiers ---

# Répertoire de base du projet
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dossier contenant les journaux de bord (fichiers .docx nommés AAAA-MM-JJ.docx)
JOURNAL_DIR = os.path.join(BASE_DIR, "journals")

# Dossier pour les sorties générées (rapports, plans, visualisations)
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Dossier pour stocker la base de données vectorielle persistante (ChromaDB)
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_db")

# Chemin par défaut pour le plan du rapport généré (JSON)
DEFAULT_PLAN_FILE = os.path.join(OUTPUT_DIR, "report_plan.json")

# Chemin par défaut pour le rapport généré (DOCX)
DEFAULT_REPORT_OUTPUT = os.path.join(OUTPUT_DIR, "apprenticeship_report.docx")

# Chemin (Optionnel) pour l'export PDF (nécessite outil externe)
DEFAULT_PDF_OUTPUT = os.path.join(OUTPUT_DIR, "apprenticeship_report.pdf")

# Chemin vers le fichier PDF contenant les guidelines/attendus du mémoire
# Assurez-vous que ce chemin est correct ou que le fichier est à la racine.
# Utiliser des '/' ou un chemin relatif avec os.path.join est plus sûr sous Windows.
# GUIDELINES_PDF_PATH = "C:/Users/arthu/Desktop/agent_redaction/Mémoire_Alternance_Job.pdf" # Option avec chemin absolu (moins portable)
GUIDELINES_PDF_PATH = os.path.join(BASE_DIR, "Mémoire_Alternance_Job.pdf") # Option si le PDF est à la racine du projet


# --- Configuration de la Base de Données Vectorielle ---

# Nom de la collection ChromaDB pour les entrées de journal
JOURNAL_DB_COLLECTION_NAME = "journal_entries" # Renommé pour clarté vs variable précédente

# Nom de la collection ChromaDB pour les documents de référence (guidelines)
REFERENCE_DOCS_COLLECTION_NAME = "reference_docs"

# Paramètres pour le découpage du texte en chunks
CHUNK_SIZE = 1000 # Taille cible des chunks (en caractères)
CHUNK_OVERLAP = 150 # Chevauchement entre les chunks consécutifs


# --- Configuration du Rapport ---

# Format attendu pour les noms de fichiers des journaux
DATE_FILENAME_FORMAT = "%Y-%m-%d"

# Structure par défaut du rapport de mémoire (utilisée par 'create_plan')
# L'indentation est utilisée pour déterminer les niveaux/sous-sections.
DEFAULT_REPORT_STRUCTURE = [
    "Introduction",
    "Company Context (Gecina)",
    "Apprenticeship Role & Objectives",
    "Methodology",
    "Projects Undertaken",
    "   Project A: Description & AI Application", # Indenté = sous-section
    "   Project B: Description & AI Application", # Indenté = sous-section
    "Skills Developed (Technical & Soft)",
    "Challenges & Solutions",
    "Learning Outcomes & Reflection",
    "Conclusion",
    "Bibliography",
    "Appendices (Optional)"
]

# Compétences clés à suivre/mapper (utilisées par CompetencyMapper)
COMPETENCIES_TO_TRACK = [
    "AI Model Implementation", # Ex: Déploiement Copilot, utilisation Studio
    "Data Analysis & Visualization", # Ex: Analyse de performance, création de graphiques
    "Project Management (Agile/Scrum)", # Ex: Gestion itérative, PoC
    "Stakeholder Communication", # Ex: Présentations, recueil de besoins
    "Problem Solving", # Ex: Identifier limitations, trouver contournements
    "Business Acumen (Real Estate Sector)", # Ex: Compréhension contexte Gecina
    "Python Programming", # Ex: Scripts pour l'agent, analyse (si applicable)
    "Cloud Platforms (AWS/Azure/GCP)", # Ex: Utilisation Azure AD (Entra ID)
    "Ethical AI Considerations" # Ex: Confidentialité, biais, sécurité
]


# --- Création des Dossiers de Sortie ---

# S'assurer que les dossiers nécessaires existent au démarrage
try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
except OSError as e:
     print(f"WARNING: Could not create output/vector directories: {e}")


# --- Vérifications de Configuration Essentielles ---

# Vérifier si la clé API Google est définie
if not GOOGLE_API_KEY:
    raise ValueError("FATAL ERROR: GOOGLE_API_KEY not found in the .env file. "
                     "Please create a .env file and add your Google AI API key.")

# Vérifier si le fichier PDF des guidelines existe (Warning seulement)
if not os.path.exists(GUIDELINES_PDF_PATH):
    print(f"WARNING: Guidelines PDF not found at the specified path: {GUIDELINES_PDF_PATH}")
    print(f"Ensure the file exists or update the GUIDELINES_PDF_PATH in config.py.")
    print("The 'process_guidelines' command will fail, and guideline-related features may be limited.")