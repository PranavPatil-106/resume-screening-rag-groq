import os
from dotenv import load_dotenv
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

load_dotenv()

class HuggingFaceVectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):

        """
        Initialize the vector store with FAISS and sentence transformers.
        
        Args:
            model_name: Sentence transformer model name
        """

        self.hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        self.model_name = model_name
        self.use_hf_api = False
        if self.hf_token:
            # Prefer using HF Inference API when token available (avoids local model download)
            self.use_hf_api = True
            self.model = None
        else:
            try:
                self.model = SentenceTransformer(model_name)
            except Exception as e:
                print(f"Warning: Could not load {model_name}, falling back to a simpler approach")
                # Fallback to a basic embedding approach
                self.model = None
        self.index = None
        self.vectors = []
        self.documents = []
        self.dimension = None

    def embed_text(self, text: str):

        """
        Generate embeddings for the given text.
        Uses HF Inference API if HF token is available; otherwise uses local sentence-transformers.
        Falls back to a simple hash-based vector if neither is available.
        """

        if self.use_hf_api:
            import requests
            headers = {"Authorization": f"Bearer {self.hf_token}", "Accept": "application/json"}
            url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.model_name}"
            try:
                resp = requests.post(url, headers=headers, json={"inputs": text, "options": {"wait_for_model": True}}, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                # Some models return [seq_len, hidden]; we average to get a single embedding
                arr = np.array(data, dtype=np.float32)
                if arr.ndim == 2:
                    emb = arr.mean(axis=0)
                elif arr.ndim == 3:
                    emb = arr.mean(axis=(0, 1))
                else:
                    emb = arr.astype(np.float32)
                return emb
            except Exception as e:
                print(f"Warning: HF Inference API embedding failed: {e}. Falling back to local/heuristic embedding.")
                # Continue to local/heuristic fallback
        if self.model is not None:
            return self.model.encode([text])[0]
        else:
            # Fallback: simple TF-IDF-like approach using word counts
            import hashlib
            words = text.lower().split()
            # Create a simple hash-based embedding
            embedding = np.zeros(384)
            for i, word in enumerate(words[:384]):
                hash_val = int(hashlib.md5(word.encode()).hexdigest(), 16)
                embedding[i] = (hash_val % 1000) / 1000.0
            return embedding

    def add_documents(self, docs):

        """
        Add documents to the vector store by generating embeddings.
        
        Args:
            docs: List of documents with 'resume_id' and 'content' keys
        """

        # Collect all documents and generate embeddings
        embeddings = []
        for doc in docs:
            self.documents.append(doc["content"])
            embedding = self.embed_text(doc["content"])
            embeddings.append(embedding)
            self.vectors.append({
                "resume_id": doc["resume_id"],
                "embedding": embedding,
                "content": doc["content"]
            })
        
        # Initialize FAISS index if not already done
        if self.index is None and embeddings:
            self.dimension = len(embeddings[0])
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            embeddings_array = np.array(embeddings).astype('float32')
            faiss.normalize_L2(embeddings_array)
            self.index.add(embeddings_array)
        elif embeddings:
            # Add new embeddings to existing index
            embeddings_array = np.array(embeddings).astype('float32')
            faiss.normalize_L2(embeddings_array)
            self.index.add(embeddings_array)

    def similarity_search(self, query, top_k=3):

        """
        Perform similarity search to find the most relevant documents.
        
        Args:
            query: Search query text
            top_k: Number of top results to return
            
        Returns:
            List of top_k most similar documents
        """
        
        if not self.vectors or self.index is None:
            return []
        
        query_emb = self.embed_text(query)
        query_emb = query_emb.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_emb)
        
        # Search using FAISS
        scores, indices = self.index.search(query_emb, min(top_k, len(self.vectors)))
        
        # Return results in the same format as before
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.vectors):
                results.append(self.vectors[idx])
        
        return results
