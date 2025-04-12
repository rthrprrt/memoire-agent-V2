# dev.nix (Configuration pour l'environnement de l'agent)

# To learn more about how to use Nix to configure your environment
# see: https://firebase.google.com/docs/studio/customize-workspace
{ pkgs, ... }: {
  # Utilisation du channel stable de Nixpkgs (peut être changé en "unstable" pour des versions plus récentes)
  channel = "stable-24.05";

  # Liste des paquets système et Python nécessaires
  # Utilisez https://search.nixos.org/packages pour trouver/vérifier les noms exacts
  packages = [
    # --- Python et Pip ---
    pkgs.python311                 # Version de Python (ajuster si vous utilisez 3.10 ou 3.12)
    pkgs.python311Packages.pip     # Outil pip pour installer d'autres paquets si besoin

    # --- Dépendances Python de requirements.txt ---
    pkgs.python311Packages.python-dotenv         # Pour charger .env
    pkgs.python311Packages.openai              # Interface API OpenAI (utilisée pour Gemini aussi)
    pkgs.python311Packages.python-docx         # Lecture des fichiers .docx
    pkgs.python311Packages.chromadb            # Base de données vectorielle (peut tirer des dépendances C++)
    pkgs.python311Packages.tiktoken            # Tokenizer utilisé par OpenAI/ChromaDB
    pkgs.python311Packages.matplotlib          # Pour les visualisations
    pkgs.python311Packages.plotly              # Optionnel pour visualisations
    pkgs.python311Packages.pydantic            # Validation de données/modèles
    pkgs.python311Packages.google-generativeai # SDK Google AI (Gemini)
    pkgs.python311Packages.sentence-transformers # Pour les embeddings locaux
    pkgs.python311Packages.torch               # Dépendance majeure pour sentence-transformers (version CPU par défaut)
    # Si vous avez un GPU et savez configurer CUDA dans Nix: pkgs.python311Packages.torchWithCuda
    pkgs.python311Packages.transformers        # Autre dépendance majeure pour sentence-transformers
    pkgs.python311Packages.pypdf2              # Lecture des fichiers PDF

    # --- Dépendances Système Potentielles (souvent gérées par Nix, mais à ajouter si erreurs) ---
    # pkgs.gcc        # Compilateur C/C++ (parfois requis par des paquets Python avec extensions C)
    # pkgs.git        # Pour les opérations Git si besoin dans l'environnement
    # pkgs.pkg-config # Outil de configuration de build
    # pkgs.blas       # Bibliothèques d'algèbre linéaire (pour numpy/scipy/torch)
    # pkgs.lapack     # Autre bibliothèque d'algèbre linéaire
    # pkgs.freetype   # Pour matplotlib
    # pkgs.libpng     # Pour matplotlib

  ];

  # Variables d'environnement (optionnel)
  env = {
    # PYTHON Mieux géré par Nix directement
    # PYTHONPATH = "..."; # Normalement pas nécessaire avec Nix
  };

  # Configuration de l'éditeur IDX (si utilisé)
  idx = {
    # Extensions VSCode/IDX (exemples)
    extensions = [
       "ms-python.python"
       "ms-python.vscode-pylance"
       # "visualstudioexptteam.vscodeintellicode",
       # "nixos.nix-ide" # Pour l'aide à l'édition des fichiers Nix
    ];

    # Prévisualisations (optionnel)
    previews = {
      enable = true;
      previews = {
        # Définir des previews si vous avez un serveur web ou autre
      };
    };

    # Hooks du cycle de vie Workspace
    workspace = {
      # À la création du workspace
      onCreate = {
        # On pourrait lancer 'pip install -r requirements.txt' ici si on préférait cette approche
        # pip-reqs = "pip install -r requirements.txt";
      };
      # Au démarrage/redémarrage du workspace
      onStart = {
        # Lancer des services en arrière-plan si nécessaire
        # start-ollama = "ollama serve &"; # Exemple si on utilisait Ollama
      };
    };
  };

  # Options supplémentaires de l'environnement Nix si nécessaire
  # shellHook = ''
  #   echo "Environnement Nix pour l'Agent Mémoire prêt."
  #   # Autres commandes à exécuter au démarrage du shell
  # '';
}