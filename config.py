# config.py (Version avec guidelines structurées et mapping collaborateurs)

import os
from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# --- Configuration des APIs ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_CHAT_MODEL_NAME = os.getenv("GEMINI_CHAT_MODEL_NAME", "gemini-1.5-flash")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "llama3.1:8b-instruct-q8_0")

# --- Fournisseurs LLM supportés (pour le CLI --llm) ---
LLM_PROVIDERS = [
    "google",
    "ollama",
]
# Alias historique, si utilisé ailleurs
SUPPORTED_LLM_PROVIDERS = LLM_PROVIDERS

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

# --- AJOUT : Chemins et limites pour l'analyse holistique ---
HOLISTIC_ANALYSIS_FILE = os.path.join(OUTPUT_DIR, "holistic_analysis.json")
# Nombre maximal de caractères à traiter lors de l'analyse holistique
HOLISTIC_ANALYSIS_MAX_CHARS = 20000

# --- Configuration de la Base de Données Vectorielle ---
JOURNAL_DB_COLLECTION_NAME = "journal_entries"
REFERENCE_DOCS_COLLECTION_NAME = "reference_docs"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# --- Configuration du Rapport ---
DATE_FILENAME_FORMAT = "%Y-%m-%d"

# --- Structure basée sur le PDF Epitech ---
DEFAULT_REPORT_STRUCTURE = [
    "Introduction",
    "Description de la mission",
    "Analyse des compétences",
    "Évaluation de la performance",
    "Réflexion personnelle et professionnelle",
    "Conclusion",
    "Bibliographie",
    "Annexes"
]

# --- Guidelines Epitech Structurées par Section ---
STRUCTURED_GUIDELINES = {
    "Introduction": [
        "Décrire l'entreprise (Gecina), secteur, contexte mission.",
        "Analyser entreprise : organigramme, business model, positionnement, offre, concurrents, stratégie développement.",
        "Clarifier objectifs mémoire en lien avec compétences clés RNCP 35284 pertinentes.",
        "Clarifier objectifs mission professionnelle."
    ],
    "Description de la mission": [
        "Expliquer fiche de poste.",
        "Détailler tâches réalisées.",
        "Préciser responsabilités assumées.",
        "Lister projets participés.",
        "Décrire position/rôle dans équipe/projet.",
        "Décrire process d'intervention.",
        "Préciser missions spécifiques réalisées.",
        "Mentionner initiatives personnelles."
    ],
    "Analyse des compétences": [
        "Discuter compétences clés (opérationnelles RNCP et autres, techniques/soft skills) développées ou améliorées.",
        "Relier explicitement ces compétences aux exigences du titre RNCP 35284.",
        "Montrer avec exemples concrets comment les connaissances de la formation Epitech ont été appliquées à des situations réelles/complexes."
    ],
    "Évaluation de la performance": [
        "Analyser la performance personnelle.",
        "Utiliser des exemples spécifiques tirés de l'expérience.",
        "Intégrer des feedbacks reçus (si disponibles).",
        "Réaliser une auto-évaluation critique (le texte doit refléter cette auto-critique)."
    ],
    "Réflexion personnelle et professionnelle": [
        "Réfléchir à l'intégration dans l'entreprise.",
        "Analyser l'impact du travail réalisé.",
        "Décrire l'évolution personnelle pendant la mission.",
        "Identifier des domaines d'amélioration personnels/professionnels.",
        "Identifier des compétences spécifiques à développer pour la future carrière."
    ],
    "Conclusion": [
        "Synthèse des apprentissages : Résumer apports mission (compétences, connaissances, développement personnel).",
        "Implications carrière future : Discuter comment l'expérience prépare/oriente la trajectoire professionnelle."
    ],
    "Bibliographie": [
        "Lister toutes les sources utilisées.",
        "Suivre un style de citation cohérent (APA ou Harvard)."
    ],
    "Annexes": [
        "Inclure documents pertinents demandés (CV, portfolio, rapports, évaluations...)."
    ]
}
# --- FIN Guidelines Structurées ---

# Compétences clés à suivre/mapper
COMPETENCIES_TO_TRACK = [
    "Analyse des besoins SI",  # RNCP
    "Conception de systèmes SI",  # RNCP
    "Gestion de projets SI",  # RNCP
    "Maintenance et évolution SI",  # RNCP
    "Assistance et formation utilisateurs SI",  # RNCP
    "Implémentation IA / Modèles IA",
    "Prompt Engineering",
    "Power Platform Development",
    "Analyse de données",
    "Communication (interne, ESN, Comex)",
    "Adaptabilité / Apprentissage continu",
    "Stratégie IA en entreprise",
    "Veille technologique IA",
    "Gestion du changement (IA)",
    "Python Programming (Agent Dev)",
]

# --- Mapping Collaborateurs (Pour Anonymisation) ---
COLLABORATOR_TITLES = {
    "Beñat Ortega": "le Directeur général",
    "Nicolas Dutreuil": "le Directeur général adjoint finances",
    "Valérie Britay": "la Directrice adjointe pôle bureau",
    "Marie lalande": "la Directrice exécutive en charge de l'ingénieurie et du RSE",
    "Romain Veber": "le directeur executif des investissements et du développement",
    "Pierre-emmanuel Bandioli": "le Directeur exécutif pôle résidentiel",
    "Chrisitne Harné": "la directrice exécutive des ressources humaines",
    "Nicolas Broband": "le Directeur de la communication financière",
    "Thierry Perisser": "le Directeur des systèmes d'informations (DSI)",
    "Romain Hardy": "le Directeur corporate finance",
    "Agnès Arnaud": "l'Assistante de direction",
    "Brahim Annour": "le Directeur de l'innovation",
    "Alexandre Morel": "le Chef de projet Innovation",
    "Souhail Ouakkas": "le Stagiaire IA innovation",
    "Jérôme Carecchio": "le Responsable de Projets Informatique",
    # Ajouter d'autres si nécessaire
}
# --- FIN Mapping ---

# --- Création des Dossiers de Sortie ---
try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
except OSError as e:
    print(f"AVERTISSEMENT: Impossible de créer les dossiers output/vector: {e}")

# --- Vérifications de Configuration Essentielles ---
if LLM_PROVIDER == "google" and not GOOGLE_API_KEY:
    raise ValueError(
        "ERREUR FATALE: GOOGLE_API_KEY non trouvée dans le fichier .env et "
        "LLM_PROVIDER est 'google'. Veuillez créer un fichier .env et ajouter votre clé API Google AI, "
        "ou choisir LLM_PROVIDER='ollama'."
    )

if not os.path.exists(GUIDELINES_PDF_PATH):
    print(f"AVERTISSEMENT: PDF Guidelines non trouvé : {GUIDELINES_PDF_PATH}")
    print("Assurez-vous que le fichier existe ou mettez à jour GUIDELINES_PDF_PATH dans config.py.")
    print("La commande 'process_guidelines' échouera, et les fonctionnalités liées aux guidelines seront limitées.")
