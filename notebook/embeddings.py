import numpy as np
from sqlalchemy import text
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingsManager:

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        self.model = SentenceTransformer(self.model_name)

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(texts)
        return embeddings  