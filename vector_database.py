# vector_database.py (Debugging version)

import chromadb
from chromadb.utils import embedding_functions
import config # Ensure this import is present and correct
from typing import List, Dict, Any, Optional
import logging
import uuid
import time

# Utiliser un logger spécifique pour ce module pour un meilleur suivi
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')
log = logging.getLogger(__name__) # Utilisation de __name__

class VectorDBManager:
    """
    Manages interactions with ChromaDB vector stores for journal entries
    and reference documents, using a local Sentence Transformer model.
    Includes extra debugging logs for initialization.
    """

    def __init__(self, path: str = config.VECTOR_DB_DIR,
                 journal_collection_name: str = config.VECTOR_DB_COLLECTION_NAME,
                 ref_collection_name_arg: str = config.REFERENCE_DOCS_COLLECTION_NAME): # Renommé l'argument pour le debug

        log.info("--- Entering VectorDBManager __init__ ---")
        log.info(f"Received path: {path}")
        log.info(f"Received journal_collection_name: {journal_collection_name}")
        # Log la valeur de l'argument tel que reçu (avec sa valeur par défaut normalement)
        log.info(f"Received ref_collection_name_arg: {ref_collection_name_arg}")

        self.path = path
        self.journal_collection_name = journal_collection_name

        # --- Débogage : Log juste avant l'assignation qui pose problème ---
        try:
            # Afficher explicitement la variable que nous allons utiliser
            log.info(f"Value of 'ref_collection_name_arg' right before assignment: {ref_collection_name_arg}")

            # C'est la ligne (ou proche) qui causait l'erreur NameError
            self.reference_collection_name = ref_collection_name_arg
            log.info(f"Successfully assigned self.reference_collection_name = {self.reference_collection_name}")

        except NameError as ne:
            # Si l'erreur se produit quand même, logger l'état local
            log.error(f"!!! NameError occurred during assignment of self.reference_collection_name: {ne}")
            try:
                # Tenter d'afficher les variables locales pour comprendre le contexte
                log.error(f"Local variables available at error point: {locals()}")
            except Exception as e_locals:
                log.error(f"Could not retrieve local variables: {e_locals}")
            raise # Re-lancer l'erreur pour arrêter l'exécution
        except Exception as e:
             log.error(f"!!! An unexpected error occurred during assignment: {e}", exc_info=True)
             raise

        # --- Initialisation du Client ChromaDB ---
        try:
            self.client = chromadb.PersistentClient(path=self.path)
            log.info("ChromaDB PersistentClient initialized.")
        except Exception as e_client:
             log.error(f"Failed to initialize ChromaDB PersistentClient: {e_client}", exc_info=True)
             raise RuntimeError("Failed to initialize ChromaDB client") from e_client

        # --- Configuration de la Fonction d'Embedding ---
        log.info(f"Initializing local embedding model: '{config.LOCAL_EMBEDDING_MODEL_NAME}' on device: '{config.LOCAL_EMBEDDING_DEVICE}'")
        try:
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=config.LOCAL_EMBEDDING_MODEL_NAME,
                device=config.LOCAL_EMBEDDING_DEVICE
            )
            log.info("SentenceTransformerEmbeddingFunction initialized.")
            # Optionnel: Test rapide de l'embedding function
            # try:
            #     log.debug("Testing embedding function...")
            #     _ = self.embedding_function(["test"])
            #     log.debug("Embedding function test successful.")
            # except Exception as test_e:
            #     log.warning(f"Embedding function test failed: {test_e}", exc_info=True)
        except Exception as e_embed:
            log.error(f"ERROR initializing SentenceTransformerEmbeddingFunction: {e_embed}", exc_info=True)
            raise RuntimeError("Failed to setup local embedding model") from e_embed

        # --- Configuration des Collections ---
        try:
            # Collection des Journaux
            self.journal_collection = self.client.get_or_create_collection(
                name=self.journal_collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
            log.info(f"Journal collection '{self.journal_collection_name}' ready.")

            # Collection des Références
            self.reference_collection = self.client.get_or_create_collection(
                name=self.reference_collection_name, # Utilise l'attribut assigné plus haut
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
            log.info(f"Reference document collection '{self.reference_collection_name}' ready.")
            log.info("--- Vector DB Manager initialization complete ---")

        except Exception as e_coll:
            log.error(f"Failed to get or create ChromaDB collection(s): {e_coll}", exc_info=True)
            raise

    # --- Méthodes pour ajouter des données ---
    def add_entry_chunks(self, entry_data: Dict[str, Any]):
        """Adds journal chunks to the journal collection."""
        entry_id = entry_data.get("entry_id")
        chunks = entry_data.get("chunks")
        if not all([entry_id, chunks, entry_data.get("date_iso"), entry_data.get("source_file")]):
            log.warning(f"Missing data for journal entry {entry_id}. Skipping add.")
            return
        chunk_ids = [f"{entry_id}_{uuid.uuid4()}" for _ in chunks]
        metadatas = [{"entry_id": entry_id, "date": entry_data.get("date_iso"), "source_file": entry_data.get("source_file"), "tags": entry_data.get("tags_str", ""),"source_type": "journal" } for _ in chunks]
        try:
            self.journal_collection.add(ids=chunk_ids, documents=chunks, metadatas=metadatas)
            log.info(f"Added {len(chunks)} journal chunks for entry {entry_id} using local embeddings.")
        except Exception as e: log.error(f"Error adding journal chunks for entry {entry_id}: {e}", exc_info=True)

    def add_reference_chunks(self, doc_name: str, chunks: List[str]):
        """Adds reference chunks to the reference collection."""
        if not chunks: log.warning(f"No reference chunks for '{doc_name}'. Skipping."); return
        chunk_ids = [f"ref_{doc_name}_{uuid.uuid4()}" for _ in chunks]
        metadatas = [{"document_name": doc_name, "source_type": "reference"} for _ in chunks]
        try:
            self.reference_collection.add(ids=chunk_ids, documents=chunks, metadatas=metadatas)
            log.info(f"Added {len(chunks)} reference chunks from '{doc_name}' using local embeddings.")
        except Exception as e: log.error(f"Error adding reference chunks from '{doc_name}': {e}", exc_info=True)

    # --- Méthodes pour rechercher ---
    def search_journals(self, query: str, k: int = 5, filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Searches the journal collection."""
        return self._search_collection(self.journal_collection, query, k, filter_dict)

    def search_references(self, query: str, k: int = 3, filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Searches the reference collection."""
        return self._search_collection(self.reference_collection, query, k, filter_dict)

    def _search_collection(self, collection: chromadb.Collection, query: str, k: int, filter_dict: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Internal method to query a specific ChromaDB collection."""
        if not query or not isinstance(query, str): return []
        if k <= 0: k = 1
        try:
            results = collection.query(query_texts=[query], n_results=k, where=filter_dict, include=['documents', 'metadatas', 'distances'])
            processed_results = []
            if results and results.get('ids') and results['ids'][0]:
                num_results = len(results['ids'][0])
                for i in range(num_results):
                    try:
                        doc, meta, dist, id_ = results['documents'][0][i], results['metadatas'][0][i], results['distances'][0][i], results['ids'][0][i]
                        if all(v is not None for v in [doc, meta, dist, id_]): processed_results.append({"id": id_, "document": doc, "metadata": meta, "distance": dist})
                        else: log.warning(f"Skipping result index {i} in '{collection.name}' due to missing data.")
                    except (IndexError, TypeError) as res_err: log.warning(f"Error processing result index {i} in '{collection.name}': {res_err}. Skipping."); continue
            processed_results.sort(key=lambda x: x['distance'])
            log.info(f"Search in '{collection.name}' for '{query[:50]}...' returned {len(processed_results)} results.")
            return processed_results
        except Exception as e: log.error(f"Error during query in '{collection.name}' for '{query[:50]}...': {e}", exc_info=True); return []

    # --- Méthode pour récupérer par ID (journaux seulement pour l'instant) ---
    def get_journal_entry_by_id(self, entry_id: str) -> Optional[Dict[str, Any]]:
         """Retrieves all chunks for a specific journal entry_id."""
         if not entry_id: return None
         try:
             results = self.journal_collection.get(where={"entry_id": entry_id}, include=['documents', 'metadatas'])
             if results and results.get('ids'): return results
             else: log.info(f"No journal chunks found for entry_id {entry_id}."); return None
         except Exception as e: log.error(f"Error retrieving journal entry by ID '{entry_id}': {e}", exc_info=True); return None

    # --- Méthodes pour vider les collections ---
    def clear_journal_collection(self):
         """Deletes all items from the journal collection."""
         self._clear_collection_internal(self.journal_collection, self.journal_collection_name)

    def clear_reference_collection(self):
         """Deletes all items from the reference collection."""
         self._clear_collection_internal(self.reference_collection, self.reference_collection_name)

    def _clear_collection_internal(self, collection: chromadb.Collection, collection_name: str):
        """Internal method to delete all items from a specific collection."""
        log.warning(f"Attempting to clear collection '{collection_name}'!")
        try:
            count_before = collection.count()
            if count_before > 0:
                log.info(f"'{collection_name}' has {count_before} items. Deleting...")
                collection.delete() # Supprime tout
                time.sleep(0.5)
                count_after = collection.count()
                if count_after == 0: log.info(f"Successfully cleared '{collection_name}'.")
                else: log.warning(f"Deletion sent for '{collection_name}', but count is {count_after} (was {count_before}).")
            else: log.info(f"'{collection_name}' is already empty.")
        except Exception as e: log.error(f"Error clearing collection '{collection_name}': {e}", exc_info=True)