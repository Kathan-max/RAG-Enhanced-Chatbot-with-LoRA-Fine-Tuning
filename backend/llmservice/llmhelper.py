from config.settings import *
from utils.logger import Logger
import tiktoken
import openai
from llmservice.adaptiveJsonExtractor import AdaptiveJsonExtractor
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import HuggingFaceHub

load_dotenv(dotenv_path="./.env")

class LLmHelper:
    
    def __init__(self,  generation_model):
        self.generation_model = generation_model
        self.logger = Logger(name="RAGLogger").get_logger()
        self.max_tokens = MAX_ALLOWED_TOKENS.get(self.generation_model, None)
        self.token_encoding_obj = tiktoken.encoding_for_model(self.generation_model)
        self.adaptive_json_extractor = AdaptiveJsonExtractor()
        self.openai_api_key = os.getenv('OPEN_AI_API_KEY')
        os.environ["OPENAI_API_KEY"] = os.getenv('OPEN_AI_API_KEY')
        os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
        os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
        self.openai_client = openai.OpenAI()
    
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
