# vector_database.py (Version finale propre avec fix clear_collection)

import chromadb
from chromadb.utils import embedding_functions
import config # Pour lire les chemins et le nom du modèle local
from typing import List, Dict, Any, Optional
import logging
import uuid
import time # Importé pour la pause dans clear_collection

# Configuration du logging pour ce module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')
log = logging.getLogger(__name__)

class VectorDBManager:
    """
    Manages interactions with ChromaDB vector stores for journal entries
    and reference documents, using a local Sentence Transformer model.
    """

    def __init__(self, path: str = config.VECTOR_DB_DIR,
                 journal_collection_name: str = config.JOURNAL_DB_COLLECTION_NAME, # Utilise le nom corrigé de config.py
                 ref_collection_name: str = config.REFERENCE_DOCS_COLLECTION_NAME):
        """
        Initializes the ChromaDB client, sets up the embedding function using
        Sentence Transformers based on config.py, and gets/creates the collections.
        """
        log.info("--- Initializing VectorDBManager ---")
        self.path = path
        self.journal_collection_name = journal_collection_name
        self.reference_collection_name = ref_collection_name # Utilise l'argument directement
        log.info(f"Journal Collection Name: '{self.journal_collection_name}'")
        log.info(f"Reference Collection Name: '{self.reference_collection_name}'")

        # --- Initialisation du Client ChromaDB ---
        try:
            self.client = chromadb.PersistentClient(path=self.path)
            log.info("ChromaDB PersistentClient initialized.")
        except Exception as e_client:
             log.error(f"Failed to initialize ChromaDB PersistentClient: {e_client}", exc_info=True)
             raise RuntimeError("Failed to initialize ChromaDB client") from e_client

        # --- Configuration de la Fonction d'Embedding (Partagée) ---
        log.info(f"Initializing local embedding model: '{config.LOCAL_EMBEDDING_MODEL_NAME}' on device: '{config.LOCAL_EMBEDDING_DEVICE}'")
        try:
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=config.LOCAL_EMBEDDING_MODEL_NAME,
                device=config.LOCAL_EMBEDDING_DEVICE
            )
            log.info("SentenceTransformerEmbeddingFunction initialized.")
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
            log.info(f"Journal collection '{self.journal_collection_name}' ready (Count: {self.journal_collection.count()}).")

            # Collection des Références
            self.reference_collection = self.client.get_or_create_collection(
                name=self.reference_collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
            log.info(f"Reference document collection '{self.reference_collection_name}' ready (Count: {self.reference_collection.count()}).")
            log.info("--- Vector DB Manager initialization complete ---")

        except Exception as e_coll:
            log.error(f"Failed to get or create ChromaDB collection(s): {e_coll}", exc_info=True)
            raise

    # --- Méthodes pour ajouter des données ---
    def add_entry_chunks(self, entry_data: Dict[str, Any]):
        """Adds journal chunks to the journal collection."""
        entry_id = entry_data.get("entry_id"); chunks = entry_data.get("chunks")
        if not all([entry_id, chunks, entry_data.get("date_iso"), entry_data.get("source_file")]):
            log.warning(f"Missing data for journal entry {entry_id}. Skipping add."); return
        chunk_ids = [f"{entry_id}_{uuid.uuid4()}" for _ in chunks]
        metadatas = [{"entry_id": entry_id, "date": entry_data.get("date_iso"), "source_file": entry_data.get("source_file"), "tags": entry_data.get("tags_str", ""),"source_type": "journal" } for _ in chunks]
        try: self.journal_collection.add(ids=chunk_ids, documents=chunks, metadatas=metadatas); log.info(f"Added {len(chunks)} journal chunks for entry {entry_id}.")
        except Exception as e: log.error(f"Error adding journal chunks for {entry_id}: {e}", exc_info=True)

    def add_reference_chunks(self, doc_name: str, chunks: List[str]):
        """Adds reference chunks to the reference collection."""
        if not chunks: log.warning(f"No ref chunks for '{doc_name}'."); return
        chunk_ids = [f"ref_{doc_name}_{uuid.uuid4()}" for _ in chunks]
        metadatas = [{"document_name": doc_name, "source_type": "reference"} for _ in chunks]
        try: self.reference_collection.add(ids=chunk_ids, documents=chunks, metadatas=metadatas); log.info(f"Added {len(chunks)} ref chunks from '{doc_name}'.")
        except Exception as e: log.error(f"Error adding ref chunks from '{doc_name}': {e}", exc_info=True)

    # --- Méthodes pour rechercher ---
    def search_journals(self, query: str, k: int = 5, filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Searches the journal collection."""
        return self._search_collection(self.journal_collection, query, k, filter_dict)

    def search_references(self, query: str, k: int = 3, filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Searches the reference collection."""
        return self._search_collection(self.reference_collection, query, k, filter_dict)

    def _search_collection(self, collection: chromadb.Collection, query: str, k: int, filter_dict: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Internal method to query a specific ChromaDB collection."""
        if not query or not isinstance(query, str): log.warning(f"Invalid search query type: {type(query)}"); return []
        if k <= 0: k = 1
        processed_results = []
        try:
            results = collection.query(query_texts=[query], n_results=k, where=filter_dict, include=['documents', 'metadatas', 'distances'])
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
        except Exception as e: log.error(f"Error during query in '{collection.name}' for '{query[:50]}...': {e}", exc_info=True)
        return processed_results

    # --- Méthode pour récupérer par ID (journaux) ---
    def get_journal_entry_by_id(self, entry_id: str) -> Optional[Dict[str, Any]]:
         """Retrieves all chunks for a specific journal entry_id."""
         if not entry_id: log.warning("get_journal_entry_by_id called with empty ID."); return None
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

    # Méthode interne corrigée pour vider une collection
    def _clear_collection_internal(self, collection: chromadb.Collection, collection_name: str):
        """Internal method to delete all items from a specific collection by retrieving IDs first."""
        log.warning(f"Attempting to clear ALL entries from collection '{collection_name}'!")
        try:
            count_before = collection.count()
            if count_before > 0:
                log.info(f"Collection '{collection_name}' contains {count_before} items. Retrieving all IDs...")
                # Récupérer tous les IDs (méthode plus sûre)
                # Le include=[] est important pour ne récupérer que les IDs
                all_ids = collection.get(include=[])['ids']

                if all_ids:
                    log.info(f"Found {len(all_ids)} IDs to delete. Proceeding with deletion...")
                    # Supprimer en utilisant la liste des IDs récupérés
                    collection.delete(ids=all_ids)

                    # Vérification (peut prendre un instant)
                    time.sleep(0.5) # Donner un peu de temps à la DB
                    count_after = collection.count()
                    if count_after == 0:
                        log.info(f"Successfully cleared collection '{collection_name}'. Deleted {len(all_ids)} items (was {count_before}).")
                    else:
                        # Peut arriver si de nouveaux éléments sont ajoutés pendant la suppression, ou limite de get()
                        log.warning(f"Deletion command executed for {len(all_ids)} IDs in '{collection_name}', "
                                    f"but collection count is now {count_after} (was {count_before}). Check for concurrent operations or get() limitations.")
                else:
                    # Cas où count > 0 mais get() ne renvoie rien
                    log.warning(f"Collection '{collection_name}' reported {count_before} items, but failed to retrieve IDs for deletion. Collection might be in an inconsistent state.")

            else: # count_before == 0
                log.info(f"Collection '{collection_name}' is already empty. No deletion needed.")
        except Exception as e:
            log.error(f"An error occurred while trying to clear collection '{collection_name}': {e}", exc_info=True)
            try: log.error(f"Current collection count (after error): {collection.count()}")
            except: log.error("Could not retrieve collection count after error during clearing.")