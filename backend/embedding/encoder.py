from embedding.encoders.sentence_encoder import SentenceEncoder
from config.settings import *

class Encoder:
    
    def __init__(self, **kwargs):
        self.encoder_name = kwargs.get('encoder_name', DEFAULT_ENCODER_NAME)
        self.model_name = kwargs.get('model_name', DEFAULT_EMBEDDING_MODEL)
        self.encoder = None

        if self.encoder_name == SENTENCE_ENCODER:
            self.encoder = SentenceEncoder(model_name = self.model_name)
        
    def get_embeddings(self, sentences):
        return self.encoder.get_embeddings(sentences=sentences)