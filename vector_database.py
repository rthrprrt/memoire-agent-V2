# vector_database.py (Utilisant Sentence Transformers localement)

import chromadb
# Importe la fonction nécessaire pour Sentence Transformers depuis ChromaDB
from chromadb.utils import embedding_functions
import config # Pour lire les chemins et le nom du modèle local
from typing import List, Dict, Any, Optional
import logging
import uuid

# Configure le logging pour ce module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')

class VectorDBManager:
    """
    Manages interactions with the ChromaDB vector store using a local
    Sentence Transformer model for generating embeddings.
    """

    def __init__(self, path: str = config.VECTOR_DB_DIR, collection_name: str = config.VECTOR_DB_COLLECTION_NAME):
        """
        Initializes the ChromaDB client, sets up the embedding function using
        Sentence Transformers based on config.py, and gets/creates the collection.
        """
        self.path = path
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=self.path)

        # --- Configuration de la Fonction d'Embedding (Sentence Transformers) ---
        logging.info(f"Initializing local embedding model: '{config.LOCAL_EMBEDDING_MODEL_NAME}' "
                     f"on device: '{config.LOCAL_EMBEDDING_DEVICE}'")
        logging.info("This may take some time on the first run if the model needs to be downloaded.")

        try:
            # Utilise la fonction intégrée de ChromaDB pour Sentence Transformers
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=config.LOCAL_EMBEDDING_MODEL_NAME,
                device=config.LOCAL_EMBEDDING_DEVICE
                # normalize_embeddings=True # Décommentez si vous voulez normaliser les vecteurs (souvent utile)
            )
            # Test rapide pour voir si le modèle se charge (optionnel mais peut aider au debug)
            # try:
            #     logging.debug("Testing embedding function with a sample text...")
            #     _ = self.embedding_function(["test sentence"])
            #     logging.debug("Embedding function test successful.")
            # except Exception as test_e:
            #     logging.error(f"Failed to test embedding function: {test_e}")
            #     raise # Renvoyer l'erreur si le test échoue

        except Exception as e:
            logging.error(f"ERROR: Failed to initialize SentenceTransformerEmbeddingFunction with model "
                          f"'{config.LOCAL_EMBEDDING_MODEL_NAME}' on device '{config.LOCAL_EMBEDDING_DEVICE}'.")
            logging.error("Please ensure 'sentence-transformers', 'torch' (or relevant backend), "
                          "and 'transformers' libraries are installed (`pip install -r requirements.txt`).")
            logging.error(f"Verify the model name is correct and available on Hugging Face Hub.")
            logging.error("Check internet connection if the model needs downloading for the first time.")
            logging.error(f"Underlying error: {e}")
            # Rendre l'erreur fatale pour ne pas continuer avec des embeddings non fonctionnels
            raise RuntimeError("Failed to setup local embedding model") from e

        # --- Obtention ou Création de la Collection ChromaDB ---
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function, # Utilise la fonction SentenceTransformer
                metadata={"hnsw:space": "cosine"} # Distance cosinus est généralement bonne pour SBERT/embeddings sémantiques
            )
            logging.info(f"Vector DB Manager initialized with local Sentence Transformer ('{config.LOCAL_EMBEDDING_MODEL_NAME}'). "
                         f"Collection '{self.collection_name}' is ready.")
        except Exception as e:
            logging.error(f"Failed to get or create ChromaDB collection '{self.collection_name}' "
                          f"with Sentence Transformer model '{config.LOCAL_EMBEDDING_MODEL_NAME}': {e}")
            raise # Erreur fatale si la collection ne peut pas être initialisée

    def add_entry_chunks(self, entry_data: Dict[str, Any]):
        """
        Adds text chunks from journal entry data to the database.
        Expects a dictionary with 'entry_id', 'chunks' (list of strings),
        'date_iso', 'source_file', and 'tags_str'.
        """
        # Extraction et validation des données nécessaires du dictionnaire
        entry_id = entry_data.get("entry_id")
        chunks = entry_data.get("chunks")
        entry_date_iso = entry_data.get("date_iso")
        source_file = entry_data.get("source_file")
        tags_str = entry_data.get("tags_str", "") # Default to empty string if not provided

        if not all([entry_id, chunks, entry_date_iso, source_file]):
            logging.warning(f"Missing essential data (ID, chunks, date, source) for adding chunks. Entry ID: {entry_id}. Skipping add.")
            return
        if not isinstance(chunks, list) or not all(isinstance(c, str) for c in chunks):
             logging.warning(f"Invalid 'chunks' format for entry {entry_id}. Expected a list of strings. Skipping add.")
             return

        # Création des IDs et métadonnées pour chaque chunk
        chunk_ids = [f"{entry_id}_{uuid.uuid4()}" for _ in chunks]
        metadatas = [{
            "entry_id": entry_id,
            "date": entry_date_iso,
            "source_file": source_file,
            "tags": tags_str, # Stocke les tags comme une seule chaîne séparée par des virgules
        } for _ in chunks]

        try:
            # Ajoute les documents à la collection.
            # ChromaDB utilisera self.embedding_function pour générer les embeddings automatiquement.
            self.collection.add(
                ids=chunk_ids,
                documents=chunks,
                metadatas=metadatas
            )
            logging.info(f"Added {len(chunks)} chunks for entry {entry_id} ({entry_date_iso}) using local embeddings.")
        except Exception as e:
            # Log l'erreur spécifique rencontrée lors de l'ajout à ChromaDB
            logging.error(f"Error adding chunks for entry {entry_id} to ChromaDB (Local Embeddings): {e}")
            # Considérer une stratégie de relance ou un logging plus détaillé si nécessaire

    def search(self, query: str, k: int = 5, filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Searches the vector database for chunks similar to the query text.
        Allows optional metadata filtering.
        Returns a list of dictionaries, each containing chunk info and distance.
        """
        if not query or not isinstance(query, str):
            logging.warning("Search query is empty or not a string. Returning empty list.")
            return []
        if k <= 0:
             logging.warning(f"Number of results k ({k}) must be positive. Setting k=1.")
             k = 1

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                where=filter_dict, # Applique le filtre sur les métadonnées si fourni
                include=['documents', 'metadatas', 'distances'] # Demande ces champs en retour
            )

            # Traitement sécurisé des résultats pour construire la liste de sortie
            processed_results = []
            if results and results.get('ids') and results['ids'][0]:
                num_results = len(results['ids'][0])
                # Itérer en s'assurant que tous les index sont valides pour chaque liste retournée
                for i in range(num_results):
                    try:
                        doc = results['documents'][0][i]
                        meta = results['metadatas'][0][i]
                        dist = results['distances'][0][i]
                        id_ = results['ids'][0][i]
                        # Vérifier que les valeurs essentielles ne sont pas None
                        if doc is not None and meta is not None and dist is not None and id_ is not None:
                             processed_results.append({
                                 "id": id_,
                                 "document": doc,
                                 "metadata": meta,
                                 "distance": dist
                             })
                        else:
                             logging.warning(f"Skipping result at index {i} due to missing data in query response.")
                    except (IndexError, TypeError) as res_err:
                         logging.warning(f"Error processing result at index {i}: {res_err}. Skipping.")
                         continue # Passer au résultat suivant

            # Trier les résultats par distance (plus petite = plus similaire pour cosine/L2)
            processed_results.sort(key=lambda x: x['distance'])
            logging.info(f"Search for '{query[:50]}...' returned {len(processed_results)} valid results.")
            return processed_results

        except Exception as e:
            logging.error(f"Error during ChromaDB query for '{query[:50]}...': {e}")
            return [] # Retourner une liste vide en cas d'erreur

    def get_entry_by_id(self, entry_id: str) -> Optional[Dict[str, Any]]:
         """
         Retrieves all chunks and metadata associated with a specific entry_id.
         Returns the raw dictionary response from ChromaDB or None if not found/error.
         """
         if not entry_id or not isinstance(entry_id, str):
             logging.warning("Invalid entry_id provided for get_entry_by_id.")
             return None
         try:
             results = self.collection.get(
                 where={"entry_id": entry_id},
                 include=['documents', 'metadatas'] # Inclure les champs utiles
             )
             # Vérifier si des résultats ont été trouvés
             if results and results.get('ids'):
                 logging.debug(f"Retrieved {len(results['ids'])} chunks for entry_id {entry_id}.")
                 return results # Renvoie le dictionnaire complet {ids:[], embeddings:None, documents:[], metadatas:[]}
             else:
                 logging.info(f"No chunks found for entry_id {entry_id}.")
                 return None
         except Exception as e:
             logging.error(f"Error retrieving entry by ID '{entry_id}': {e}")
             return None

    def clear_collection(self):
        """
        Deletes all items currently present in the collection.
        Uses the collection.delete() method without arguments for efficiency.
        """
        collection_name = self.collection_name # Copie locale pour les messages
        logging.warning(f"Attempting to clear ALL entries from collection '{collection_name}'!")
        try:
            count_before = self.collection.count()
            if count_before > 0:
                logging.info(f"Collection '{collection_name}' contains {count_before} items. Proceeding with deletion...")
                # Utilise la méthode delete sans arguments pour supprimer tout (plus efficace)
                self.collection.delete()
                # Vérification (peut prendre un instant pour se mettre à jour)
                import time
                time.sleep(0.5) # Petite pause pour laisser la DB se mettre à jour
                count_after = self.collection.count()
                if count_after == 0:
                    logging.info(f"Successfully cleared collection '{collection_name}'. Previous count: {count_before}.")
                else:
                    # Cela peut arriver dans certains cas, par ex. si la suppression est asynchrone
                    logging.warning(f"Deletion command sent for collection '{collection_name}', "
                                    f"but count after check is {count_after} (was {count_before}).")
            else:
                logging.info(f"Collection '{collection_name}' is already empty. No deletion needed.")
        except Exception as e:
            logging.error(f"An error occurred while trying to clear collection '{collection_name}': {e}")
            # Peut-être logger l'état actuel de la collection si possible
            try:
                 logging.error(f"Current collection count (after error): {self.collection.count()}")
            except:
                 logging.error("Could not retrieve collection count after error.")