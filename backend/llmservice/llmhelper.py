from config.settings import *
from utils.logger import Logger
import tiktoken


class LLmHelper:
    
    def __init__(self,  generation_model):
        self.generation_model = generation_model
        self.logger = Logger(name="RAGLogger").get_logger()
        self.max_tokens = MAX_ALLOWED_TOKENS.get(self.generation_model, None)
        self.token_encoding_obj = tiktoken.encoding_for_model(self.generation_model)        
    
    def idealChunkTokens(self, top_k, chunks_coverage):
        """Calculate ideal chunk token size based on model and configuration."""
        try:
            generation_model = self.generation_model
            self.logger.info("Calculating ideal chunk tokens for model: %s", self.generation_model)
            
            total_chunks = top_k
            max_tokens = self.max_tokens

            if max_tokens is None:
                self.logger.error("Model %s not defined in MAX_ALLOWED_TOKENS configuration", generation_model)
                raise ValueError(f"Model {generation_model} not defined in the configurations")
            
            if total_chunks == 0:
                self.logger.error("TOTAL_CHUNKS_CONSIDERED is 0, would cause division by zero")
                raise ValueError("TOTAL_CHUNKS_CONSIDERED cannot be 0")
            
            ideal_tokens = (max_tokens * (chunks_coverage / 100)) / total_chunks
            self.logger.info("Calculated ideal chunk tokens: %d", int(ideal_tokens))
            return int(ideal_tokens)
            
        except Exception as e:
            self.logger.error("Error calculating ideal chunk tokens: %s", str(e))
            raise
    
    def getTokens(self, sent):
        """Get token count for a sentence."""
        try:
            if not sent:
                return 0
            
            tokens = len(self.token_encoding_obj.encode(sent))
            self.logger.debug("Sentence has %d tokens", tokens)
            return tokens
            
        except Exception as e:
            self.logger.error("Error counting tokens: %s", str(e))
            raise