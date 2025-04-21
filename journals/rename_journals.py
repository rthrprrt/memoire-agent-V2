# rename_journals.py

import os
import datetime
import locale
import logging
import config # Pour récupérer le chemin du dossier journals

# Configuration du logging (optionnel mais utile)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
JOURNAL_DIR = config.JOURNAL_DIR # Utilise le chemin défini dans config.py
EXPECTED_FRENCH_FORMAT = "%A %d %B %Y" # Format des noms de fichiers actuels
TARGET_DATE_FORMAT = "%Y-%m-%d"       # Format AAAA-MM-JJ désiré
FILE_EXTENSION = ".docx"

def rename_journal_files():
    """Scanne le dossier des journaux et renomme les fichiers selon le format AAAA-MM-JJ."""
    logging.info(f"--- Début du script de renommage dans le dossier : {JOURNAL_DIR} ---")
    if not os.path.isdir(JOURNAL_DIR):
        logging.error(f"Le dossier des journaux '{JOURNAL_DIR}' n'a pas été trouvé. Vérifiez config.py.")
        return

    original_locale = None
    processed_count = 0
    renamed_count = 0
    skipped_count = 0
    error_count = 0

    # --- Sauvegarder la locale actuelle et tenter de mettre la locale française ---
    try:
        original_locale = locale.getlocale(locale.LC_TIME)
        logging.info(f"Locale LC_TIME originale : {original_locale}")
    except ValueError:
        logging.warning("Impossible de récupérer la locale LC_TIME actuelle.")
        original_locale = None # On essaiera quand même de la réinitialiser à la fin si possible

    current_locale_set = False
    # Essayer plusieurs identifiants courants pour la locale française
    possible_locales = ['fr_FR.UTF-8', 'fr_FR', 'fra_fra', 'French_France.1252', 'fr-FR']
    for loc in possible_locales:
        try:
            locale.setlocale(locale.LC_TIME, loc)
            current_locale_set = True
            logging.info(f"Locale LC_TIME configurée avec succès sur '{loc}'.")
            break # Arrêter dès qu'une locale fonctionne
        except locale.Error:
            logging.debug(f"Locale '{loc}' non supportée pour LC_TIME.")
        except Exception as e_set:
            logging.warning(f"Erreur lors de la configuration de la locale sur '{loc}': {e_set}")

    if not current_locale_set:
        logging.error("Impossible de configurer une locale française pour LC_TIME. "
                      "Le script ne peut pas analyser les noms de mois/jours en français. Abandon.")
        # Réinitialiser la locale si on avait pu la sauvegarder
        if original_locale:
             try:
                 locale.setlocale(locale.LC_TIME, original_locale)
             except: pass # Ignorer les erreurs de réinitialisation ici
        return

    # --- Parcourir les fichiers du dossier ---
    logging.info(f"Scan des fichiers avec l'extension '{FILE_EXTENSION}'...")
    for filename in os.listdir(JOURNAL_DIR):
        if filename.lower().endswith(FILE_EXTENSION) and not filename.startswith('~'):
            processed_count += 1
            old_path = os.path.join(JOURNAL_DIR, filename)
            base_name = os.path.splitext(filename)[0] # Nom sans l'extension

            try:
                # Essayer d'analyser la date avec le format français
                parsed_date = datetime.datetime.strptime(base_name, EXPECTED_FRENCH_FORMAT).date()

                # Créer le nouveau nom de fichier au format AAAA-MM-JJ
                new_base_name = parsed_date.strftime(TARGET_DATE_FORMAT)
                new_filename = f"{new_base_name}{FILE_EXTENSION}"
                new_path = os.path.join(JOURNAL_DIR, new_filename)

                # Vérifier si le nouveau nom est différent et n'existe pas déjà
                if old_path == new_path:
                    logging.info(f"'{filename}' est déjà au bon format. Ignoré.")
                    skipped_count += 1
                elif os.path.exists(new_path):
                    logging.warning(f"Le fichier cible '{new_filename}' existe déjà. "
                                    f"'{filename}' n'a pas été renommé pour éviter l'écrasement.")
                    skipped_count += 1
                else:
                    # Renommer le fichier
                    os.rename(old_path, new_path)
                    logging.info(f"Renommé : '{filename}' -> '{new_filename}'")
                    renamed_count += 1

            except ValueError:
                # Le nom ne correspond pas au format français attendu
                logging.warning(f"Ignoré : '{filename}' ne correspond pas au format attendu "
                                f"'{EXPECTED_FRENCH_FORMAT}'.")
                skipped_count += 1
            except OSError as e:
                logging.error(f"Erreur OS lors du renommage de '{filename}': {e}")
                error_count += 1
            except Exception as e:
                logging.error(f"Erreur inattendue lors du traitement de '{filename}': {e}")
                error_count += 1

    # --- Réinitialiser la locale à son état original ---
    if original_locale:
        try:
            locale.setlocale(locale.LC_TIME, original_locale)
            logging.info(f"Locale LC_TIME réinitialisée à : {original_locale}")
        except Exception as e:
            logging.warning(f"Impossible de réinitialiser la locale LC_TIME à {original_locale}: {e}")

    logging.info("--- Script de renommage terminé ---")
    logging.info(f"Fichiers traités : {processed_count}")
    logging.info(f"Fichiers renommés : {renamed_count}")
    logging.info(f"Fichiers ignorés/déjà corrects : {skipped_count}")
    logging.info(f"Erreurs : {error_count}")

# --- Point d'entrée du script ---
if __name__ == "__main__":
    print("-----------------------------------------------------")
    print("ATTENTION : Ce script va tenter de renommer les fichiers")
    print(f"dans le dossier : {config.JOURNAL_DIR}")
    print("Format actuel attendu : Jour JJ Mois AAAA.docx")
    print("Format cible : AAAA-MM-JJ.docx")
    print("-----------------------------------------------------")
    print(">>> Il est FORTEMENT recommandé de faire une SAUVEGARDE")
    print(">>> du dossier 'journals' AVANT de continuer.")
    print("-----------------------------------------------------")

    confirmation = input("Voulez-vous vraiment lancer le renommage ? (oui/non): ")

    if confirmation.lower() == 'oui':
        rename_journal_files()
    else:
        print("Renommage annulé par l'utilisateur.")