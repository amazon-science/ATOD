# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# SPDX-License-Identifier: CC-BY-NC-4.0

from typing import List, Dict, Any
import re
import warnings
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os
import faiss

# Suppress transformer warnings for cleaner output
warnings.filterwarnings("ignore", message=".*encoder_attention_mask.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*BertSdpaSelfAttention.*", category=FutureWarning)

def simple_tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

# Shared SentenceTransformer cache: loading one embedder per model_name and
# sharing it across (possibly parallel) FaissRetriever instances avoids loading
# the model dozens of times under multi-worker evaluation. encode() is a
# stateless forward pass and safe to call concurrently.
import threading as _threading
_EMBEDDER_CACHE = {}
_EMBEDDER_LOCK = _threading.Lock()


def _get_shared_embedder(model_name: str):
    with _EMBEDDER_LOCK:
        model = _EMBEDDER_CACHE.get(model_name)
        if model is None:
            model = SentenceTransformer(model_name)
            _EMBEDDER_CACHE[model_name] = model
        return model

class FaissRetriever:
    """Vector database retrieval using FAISS"""
    def __init__(self, collection_name: str = "memories", model_name: str = None):
        """Initialize FAISS retriever.

        Args:
            collection_name: Name of the collection (used for file naming)
            model_name: SentenceTransformer model name or local path
        """
        self.collection_name = collection_name

        # Auto-detect model path if not provided
        if model_name is None:
            import os
            # Find the project root by looking for the models directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = current_dir
            while project_root != '/':
                if os.path.exists(os.path.join(project_root, 'models', 'all-MiniLM-L6-v2')):
                    model_name = os.path.join(project_root, 'models', 'all-MiniLM-L6-v2')
                    break
                project_root = os.path.dirname(project_root)

            if model_name is None:
                model_name = 'sentence-transformers/all-MiniLM-L6-v2'  # Fallback to download

        self.model = _get_shared_embedder(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity

        # Storage for documents and metadata
        self.documents = []
        self.metadatas = []
        self.doc_ids = []

        # File paths for persistence
        self.index_path = f"{collection_name}_faiss.index"
        self.metadata_path = f"{collection_name}_metadata.json"

        # Load existing data if available
        self._load_data()

    def _ensure_links_is_list(self, meta):
        """Helper method to ensure links field is always a list"""
        if 'links' in meta and not isinstance(meta['links'], list):
            if meta['links'] is None:
                meta['links'] = []
            elif isinstance(meta['links'], str):
                meta['links'] = [meta['links']]
            else:
                meta['links'] = []
        return meta

    def _load_data(self):
        """Load existing index and metadata from disk"""
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
                # Load FAISS index
                self.index = faiss.read_index(self.index_path)
                # Load metadata
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.documents = data['documents']
                    self.metadatas = data['metadatas']
                    self.doc_ids = data['doc_ids']
                    # Ensure links field is always a list
                    for meta in self.metadatas:
                        self._ensure_links_is_list(meta)
        except Exception as e:
            print(f"Could not load existing data: {e}")
            self._reset()

    def _save_data(self):
        """Save index and metadata to disk"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, self.index_path)

            # Save metadata
            data = {
                'documents': self.documents,
                'metadatas': self.metadatas,
                'doc_ids': self.doc_ids
            }
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception as e:
            print(f"Could not save data: {e}")

    def _reset(self):
        """Reset the collection"""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.documents = []
        self.metadatas = []
        self.doc_ids = []

        # Remove files if they exist
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.metadata_path):
            os.remove(self.metadata_path)

    def add_document(self, document: str, metadata: Dict, doc_id: str):
        """Add a document to FAISS index."""
        # Check if document already exists
        if doc_id in self.doc_ids:
            # Update existing document
            idx = self.doc_ids.index(doc_id)
            self.documents[idx] = document
            self.metadatas[idx] = self._ensure_links_is_list(metadata)
            # Rebuild index (FAISS doesn't support direct updates)
            self._rebuild_index()
        else:
            # Add new document
            embedding = self.model.encode([document])
            # Normalize for cosine similarity
            faiss.normalize_L2(embedding)
            self.index.add(embedding)
            self.documents.append(document)
            self.metadatas.append(self._ensure_links_is_list(metadata))
            self.doc_ids.append(doc_id)

        self._save_data()

    def _rebuild_index(self):
        """Rebuild the entire FAISS index"""
        if not self.documents:
            self.index = faiss.IndexFlatIP(self.dimension)
            return

        # Generate embeddings for all documents
        embeddings = self.model.encode(self.documents)

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        # Create new index and add all embeddings
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings.astype(np.float32))

    def delete_document(self, doc_id: str):
        """Delete a document from FAISS index."""
        if doc_id in self.doc_ids:
            idx = self.doc_ids.index(doc_id)

            # Remove from lists
            del self.documents[idx]
            del self.metadatas[idx]
            del self.doc_ids[idx]

            # Rebuild index
            self._rebuild_index()
            self._save_data()

    def search(self, query: str, k: int = 5):
        """Search for similar documents."""
        if self.index.ntotal == 0:
            return {
                'documents': [[]],
                'metadatas': [[]],
                'ids': [[]],
                'distances': [[]]
            }
        # Workaround: if there are few memories, just return all documents
        if self.index.ntotal <= 3:
            results = {
                'documents': [[]],
                'metadatas': [[]],
                'ids': [[]],
                'distances': [[]]
            }
            for idx in range(len(self.documents)):
                content = self.documents[idx]
                doc_id = self.doc_ids[idx]
                meta = self._ensure_links_is_list(self.metadatas[idx])
                results['documents'][0].append(content)
                results['metadatas'][0].append(meta)
                results['ids'][0].append(doc_id)
                results['distances'][0].append(1.0)
            return results

        # Generate query embedding
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        search_k = min(max(k * 2, self.index.ntotal), self.index.ntotal)
        scores, indices = self.index.search(query_embedding.astype(np.float32), search_k)

        results = {
            'documents': [[]],
            'metadatas': [[]],
            'ids': [[]],
            'distances': [[]]
        }
        seen_ids = set()
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.documents):
                content = self.documents[idx]
                doc_id = self.doc_ids[idx]
                meta = self._ensure_links_is_list(self.metadatas[idx])

                if doc_id not in seen_ids:
                    results['documents'][0].append(content)
                    results['metadatas'][0].append(meta)
                    results['ids'][0].append(doc_id)
                    results['distances'][0].append(float(scores[0][i]))
                    seen_ids.add(doc_id)
                    if len(results['documents'][0]) >= k:
                        break
        return results
