# llm_interface.py (Version utilisant Google AI / Gemini API)

import google.generativeai as genai
import config # Pour récupérer la clé API et le nom du modèle
from typing import List, Dict, Optional, Any # Ajout de Any pour l'historique Gemini
import logging
import time # Pour les délais de relance

# Configuration du logging pour ce module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')
log = logging.getLogger(__name__)

# --- Configuration Globale de l'API Google ---
# Il est préférable de configurer une seule fois au chargement du module.
try:
    if not config.GOOGLE_API_KEY:
        # Lever une erreur si la clé est manquante, car le module ne peut pas fonctionner sans.
        raise ValueError("GOOGLE_API_KEY not found in config/env. Cannot initialize Google AI.")
    genai.configure(api_key=config.GOOGLE_API_KEY)
    log.info("Google AI (Gemini) API configured successfully using GOOGLE_API_KEY.")
except ValueError as ve:
    log.critical(ve) # Log l'erreur fatale de configuration
    # Vous pourriez choisir de ne pas lever l'exception ici et de laisser l'init de GeminiLLM échouer,
    # mais il est souvent mieux d'échouer tôt si la config est manquante.
    raise # Re-lever l'exception pour arrêter si la clé manque.
except Exception as e_cfg:
    log.critical(f"Unexpected error configuring Google AI API: {e_cfg}", exc_info=True)
    raise # Re-lever pour arrêter

class GeminiLLM:
    """
    Interface pour interagir avec les modèles Google AI (Gemini) via le SDK google-generativeai.
    Gère la préparation de l'historique, les appels API avec relances, et l'extraction des réponses.
    """

    def __init__(self, model_name: str = config.GEMINI_CHAT_MODEL_NAME):
        """
        Initialise l'interface avec un modèle génératif Gemini spécifique.

        Args:
            model_name: Le nom du modèle Gemini à utiliser (ex: 'gemini-1.0-pro').
                        Doit être supporté par l'API et accessible avec votre clé.
        """
        self.model_name = model_name
        try:
            # Crée l'objet modèle génératif
            self.model = genai.GenerativeModel(self.model_name)
            log.info(f"Gemini GenerativeModel '{self.model_name}' initialized successfully.")
            # Optionnel: Vérifier l'existence/accès au modèle avec un petit appel test ?
            # try:
            #      self.model.generate_content("test", generation_config=genai.types.GenerationConfig(candidate_count=1, max_output_tokens=5))
            #      log.info(f"Successfully tested connection to Gemini model '{self.model_name}'.")
            # except Exception as test_e:
            #      log.warning(f"Could not confirm connection/access to Gemini model '{self.model_name}': {test_e}")
        except Exception as e_init:
            log.error(f"Failed to initialize Gemini GenerativeModel '{self.model_name}': {e_init}", exc_info=True)
            # Erreur critique si le modèle ne peut être initialisé
            raise RuntimeError(f"Could not initialize Gemini model {self.model_name}") from e_init

    def _prepare_gemini_history(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Convertit un historique de messages au format OpenAI [{role: 'user'/'assistant'/'system', content: '...'}]
        vers le format attendu par l'API Gemini Chat [{role: 'user'/'model', parts: ['...']}],
        en gérant l'alternance des rôles et l'intégration du prompt système.

        Args:
            messages: Liste des messages au format OpenAI.

        Returns:
            Liste des messages au format Gemini, prête pour l'API.
                 Retourne une liste vide si l'historique est invalide.
        """
        gemini_history = []
        last_role = None
        system_prompt_content = None
        pending_user_content = "" # Pour fusionner le système avec le premier user

        for i, msg in enumerate(messages):
            role = msg.get('role')
            content = msg.get('content', '').strip() # Nettoyer le contenu

            if not role or not content: # Ignorer les messages vides ou sans rôle
                continue

            if role == 'system':
                # Stocker le contenu système pour le préfixer au premier message utilisateur
                if system_prompt_content: # Gérer plusieurs messages système ? Les concaténer ?
                    system_prompt_content += f"\n{content}"
                else:
                    system_prompt_content = content
                continue # Ne pas ajouter directement

            # Adapter le rôle
            gemini_role = 'model' if role == 'assistant' else 'user'

            # Gérer le contenu utilisateur en attente (pour ajout du system prompt)
            if gemini_role == 'user':
                # Si on a un system prompt en attente, on le préfixe
                if system_prompt_content:
                    current_content = f"System Instructions:\n---\n{system_prompt_content}\n---\n\nUser Request:\n---\n{content}\n---"
                    system_prompt_content = None # Marquer comme utilisé
                else:
                    current_content = content

                # Si le rôle précédent était aussi 'user', on fusionne
                if last_role == 'user' and gemini_history:
                    log.debug("Merging consecutive user messages.")
                    # Ajouter au dernier message 'user' existant
                    if isinstance(gemini_history[-1]['parts'][0], str):
                         gemini_history[-1]['parts'][0] += f"\n\n{current_content}"
                    else: # Ne devrait pas arriver si parts contient toujours du texte
                         gemini_history[-1]['parts'].append(current_content) # Moins idéal
                else:
                    # C'est un nouveau message utilisateur (ou le premier après le système)
                    gemini_history.append({'role': 'user', 'parts': [current_content]})
                    last_role = 'user'

            elif gemini_role == 'model':
                 # Si le rôle précédent était aussi 'model', on fusionne
                 if last_role == 'model' and gemini_history:
                     log.debug("Merging consecutive model messages.")
                     if isinstance(gemini_history[-1]['parts'][0], str):
                          gemini_history[-1]['parts'][0] += f"\n\n{content}"
                     else:
                          gemini_history[-1]['parts'].append(content)
                 else:
                     # Ajouter le nouveau message modèle
                     gemini_history.append({'role': 'model', 'parts': [content]})
                     last_role = 'model'

        # Vérification finale : le dernier message ne doit pas être 'model' pour la plupart des appels generate_content
        if gemini_history and gemini_history[-1]['role'] == 'model':
            log.warning("Prepared conversation history ends with 'model' role. This might cause issues with Gemini API if it expects a final 'user' prompt.")
            # On ne modifie pas l'historique ici, mais la logique appelante doit s'en assurer.

        return gemini_history


    def _make_request(self, messages: List[Dict[str, str]], max_retries: int = 3, delay: int = 5, **params) -> Optional[str]:
        """
        Effectue un appel à l'API Gemini Chat (generate_content) avec l'historique fourni,
        gère les relances et les erreurs potentielles.

        Args:
            messages: Historique de la conversation au format OpenAI.
            max_retries: Nombre maximum de tentatives en cas d'erreur transitoire.
            delay: Délai (en secondes) entre les tentatives.
            **params: Paramètres supplémentaires pour la génération (ex: temperature, top_p, top_k, max_output_tokens).

        Returns:
            Le contenu texte de la réponse du modèle, ou None en cas d'échec après relances.
        """
        if not self.model:
            log.error("Gemini model is not initialized. Cannot make request.")
            return None

        # Préparer l'historique au format Gemini
        gemini_history = self._prepare_gemini_history(messages)
        if not gemini_history:
            log.error("Failed to prepare valid history for Gemini format. Cannot make request.")
            return None
        # L'API generate_content attend une liste de Contents, le dernier étant la requête user.
        # Notre préparation assure cela normalement, sauf avertissement.

        # Configurer les paramètres de génération
        # Note: 'max_tokens' de OpenAI correspond à 'max_output_tokens' de Gemini
        generation_config_args = {
            'temperature': params.get("temperature", 0.7),
            'top_p': params.get("top_p"),
            'top_k': params.get("top_k"),
            'max_output_tokens': params.get("max_tokens") or params.get("max_output_tokens", 2048), # Default à 2048 si non fourni
            # 'stop_sequences': params.get("stop"), # Si besoin
            'candidate_count': 1
        }
        # Filtrer les arguments None car l'API Gemini n'aime pas les recevoir explicitement
        generation_config_args = {k: v for k, v in generation_config_args.items() if v is not None}
        generation_config = genai.types.GenerationConfig(**generation_config_args)

        # Configurer les Safety Settings (bloquer contenu dangereux)
        safety_settings = [
            {"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
            for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
                      "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]
        ]

        last_error = None
        for attempt in range(max_retries):
            try:
                log.debug(f"Attempt {attempt + 1}/{max_retries}: Sending request to Gemini model '{self.model_name}'. History length: {len(gemini_history)}")

                # --- Appel API Google AI ---
                response = self.model.generate_content(
                    contents=gemini_history, # Passer l'historique formaté
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                )
                # --- Fin Appel API ---

                log.debug(f"Gemini raw response object: {response}")

                # --- Traitement de la Réponse ---
                # Vérifier si la réponse a été bloquée
                if not response.parts:
                    block_reason = "Unknown"
                    if response.prompt_feedback and response.prompt_feedback.block_reason:
                        block_reason = response.prompt_feedback.block_reason.name # Utiliser .name pour le texte
                    log.error(f"Gemini request blocked (Attempt {attempt + 1}). Reason: {block_reason}")
                    # Pour un blocage de sécurité, on ne relance pas forcément
                    last_error = f"Response blocked by safety settings ({block_reason})."
                    # Décider de relancer ou non ? Pour l'instant, on arrête.
                    # Si c'est une autre raison (ex: pas de parts mais pas de block_reason), on peut relancer.
                    if response.prompt_feedback and response.prompt_feedback.block_reason:
                         # Ne pas relancer si bloqué pour sécurité
                         return f"Error: Response blocked by safety settings ({block_reason}). Provide different input."
                    # Sinon, considérer comme une erreur potentiellement transitoire et laisser la boucle continuer

                else:
                    # Extraire le contenu texte principal
                    content = "".join(part.text for part in response.parts if hasattr(part, 'text'))
                    log.info(f"Gemini request successful (Attempt {attempt + 1}). Response length: {len(content)} chars.")
                    # Log des métadonnées si utiles/disponibles (pas standardisé comme OpenAI)
                    # log.debug(f"Response metadata: {response.usage_metadata}") # Si ça existe
                    return content.strip() # Succès !

            except Exception as e:
                last_error = e
                log.error(f"Google AI API Error (Attempt {attempt + 1}/{max_retries}): {e}", exc_info=False) # exc_info=False pour ne pas surcharger les logs sauf au final

                # Gérer les erreurs spécifiques (ex: clé invalide, quota)
                if "API key not valid" in str(e):
                    log.critical("Invalid Google API Key detected! Please check .env file.")
                    return "Error: Invalid Google API Key." # Arrêter immédiatement
                # Ajouter la gestion d'autres erreurs spécifiques ici (429 Quota, etc.)

                # Attendre avant la prochaine tentative
                if attempt < max_retries - 1:
                    log.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    log.error(f"Max retries ({max_retries}) reached for Google AI request.")
                    # Retourner un message d'erreur clair incluant la dernière erreur rencontrée
                    return f"Error: Failed to get response from Google AI after {max_retries} attempts. Last error: {last_error}"

        # Si la boucle se termine sans succès (ne devrait pas arriver sauf si max_retries=0)
        log.error("Exited retry loop without success.")
        return f"Error: Failed after {max_retries} attempts. Last error: {last_error}"

    # --- Méthodes Spécifiques aux Tâches ---
    # Ces méthodes appellent _make_request et n'ont pas besoin de changer,
    # sauf potentiellement pour ajuster les prompts au style de Gemini.

    def generate_tags(self, text: str, max_tags: int = 10) -> List[str]:
        """Generates tags using Gemini."""
        log.info(f"Generating tags for text snippet (length {len(text)}) using Gemini '{self.model_name}'...")
        prompt = f"""
        Analyze the provided text and extract the most relevant keywords or tags (up to {max_tags}).
        Focus on specific skills, tools, projects, concepts, or significant activities.
        Output ONLY a comma-separated list of tags. No introductory phrases or explanations.

        Text:
        ---
        {text[:2500]}  # Ajuster la longueur si nécessaire
        ---

        Tags:
        """
        messages = [{"role": "user", "content": prompt}]
        response = self._make_request(messages, temperature=0.1, top_p=0.95) # Paramètres ajustés pour tâche d'extraction
        if response and not response.startswith("Error:"):
            tags = [tag.strip() for tag in response.split(',') if tag.strip() and len(tag) > 1]
            log.info(f"Generated tags using Gemini: {tags}")
            return tags[:max_tags]
        log.warning(f"Tag generation with Gemini failed or returned an error: {response}")
        return []

    def draft_report_section(self, section_title: str, context_chunks: List[str], report_structure: Optional[List[str]] = None, instructions: Optional[str] = None) -> Optional[str]:
        """Drafts a report section using Gemini."""
        log.info(f"Drafting section '{section_title}' using Gemini '{self.model_name}'...")
        context = "\n---\n".join(context_chunks)
        # Limiter la taille totale du contexte envoyé pour éviter les erreurs de longueur
        max_context_len = 15000 # Ajustable
        if len(context) > max_context_len:
             log.warning(f"Context for drafting section '{section_title}' exceeds {max_context_len} chars. Truncating.")
             context = context[:max_context_len]

        structure_info = f"Overall report structure: {', '.join(report_structure)}" if report_structure else ""
        extra_instructions = f"Specific instructions for this section: {instructions}" if instructions else ""

        # Prompt pour Gemini (potentiellement avec instructions système intégrées)
        system_instructions = "You are an AI assistant writing a section for a professional MSc Apprenticeship Report. Write in an academic and formal tone. Base your response *only* on the provided 'Journal Context'. Synthesize the information into a coherent narrative for the requested section. Do not add external knowledge."
        user_prompt = f"""
Write the content for the report section titled: "{section_title}"

{structure_info}
{extra_instructions}

Use ONLY the following Journal Context:
---
{context}
---

Draft for section "{section_title}":
"""
        messages = [{"role": "user", "content": f"System Instructions:\n{system_instructions}\n\nUser Request:\n{user_prompt}"}]

        # Demander une réponse potentiellement longue
        response = self._make_request(messages, temperature=0.6, max_tokens=2048) # Utiliser max_tokens
        if response and not response.startswith("Error:"):
            log.info(f"Successfully drafted section '{section_title}'. Length: {len(response)} chars.")
            return response
        log.error(f"Failed to draft section '{section_title}' with Gemini: {response}")
        return None # Retourner None si la génération échoue ou est bloquée

    # Ajouter/Adapter les autres méthodes: summarize_text, analyze_content, check_consistency
    # en s'assurant que les prompts sont clairs et que les messages sont formatés correctement.
    def summarize_text(self, text: str, max_length: int = 150) -> Optional[str]:
         log.info(f"Summarizing text (length {len(text)}) using Gemini '{self.model_name}'...")
         prompt = f"Provide a concise summary (target around {max_length} words) of the following text:\n---\n{text[:15000]}\n---\nSummary:"
         messages = [{"role": "user", "content": prompt}]
         response = self._make_request(messages, temperature=0.5, max_tokens=512)
         if response and not response.startswith("Error:"): return response
         log.warning(f"Summarization failed: {response}")
         return None

    def analyze_content(self, text: str, analysis_prompt: str) -> Optional[str]:
         log.info(f"Analyzing content using Gemini '{self.model_name}'...")
         system_instructions = "You are an analytical assistant. Respond concisely based *only* on the provided text and the user's specific analysis request."
         user_prompt = f"{analysis_prompt}\n\nText to Analyze:\n---\n{text[:15000]}\n---\n\nAnalysis:"
         messages = [{"role": "user", "content": f"System Instructions:\n{system_instructions}\n\nUser Request:\n{user_prompt}"}]
         response = self._make_request(messages, temperature=0.2, max_tokens=1024)
         if response and not response.startswith("Error:"): return response
         log.warning(f"Content analysis failed: {response}")
         return None

    def check_consistency(self, text_segment1: str, text_segment2: str, aspect: str) -> Optional[str]:
         log.info(f"Checking consistency regarding '{aspect}' using Gemini '{self.model_name}'...")
         prompt = f"""
         Analyze the consistency between the two text segments below regarding the specific aspect: '{aspect}'.
         Point out specific discrepancies or confirm consistency. Be factual and concise.

         Segment 1:
         ---
         {text_segment1[:7000]}
         ---
         Segment 2:
         ---
         {text_segment2[:7000]}
         ---
         Consistency Check regarding '{aspect}':
         """
         messages = [{"role": "user", "content": prompt}]
         response = self._make_request(messages, temperature=0.1, max_tokens=512)
         if response and not response.startswith("Error:"): return response
         log.warning(f"Consistency check failed: {response}")
         return None