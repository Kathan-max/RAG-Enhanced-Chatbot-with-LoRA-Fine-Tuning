from sentence_transformers import SentenceTransformer
from config.settings import *

class SentenceEncoder:
    
    def __init__(self, **kwargs):
        self.model_name = kwargs.get("model_name", DEFAULT_EMBEDDING_MODEL)
        self.model = SentenceTransformer(self.model_name)
    
    def get_embeddings(self, sentences):
        return self.model.encode(sentences)