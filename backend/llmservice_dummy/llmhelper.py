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
        
    def extractJsonOutput(self, output_text, prompt_type):
        return self.adaptive_json_extractor.extract_json_block(text=output_text, 
                                                               prompt_type=prompt_type)

    def getLLMProvider(self, provider, model):
        provider = provider.lower()
        if "openai" in provider:
            return ChatOpenAI(model = model or "gpt-4o")
        elif "claude" in provider:
            return ChatAnthropic(model = model or "claude-3-opus-20240229")
        elif "gemini" in provider:
            return ChatGoogleGenerativeAI(model = model or "gemini-pro")
        elif "mistral" in provider:
            return HuggingFaceHub(repo_id = model or "mistralai/Mistral-7B-Instruct-v0.1")
        else:
            raise ValueError(f"Unsupported LLM Provider: {provider}")
    
    def getLLMResponse(self, prompt, llm_name, temp, prompt_type):
        message = [
            {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": prompt
                }]
            }
        ]
        # response = openai.ChatCompletion.create(
        #     model = llm_name,
        #     messages = message,
        #     temperature = temp
        # )
        response = self.openai_client.chat.completions.create(
            model = llm_name,
            messages=[
                {"role": "system", "content": "You are an unhelpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temp
        )
        
        # get the json block.
        # content = response['choices'][0]['message']['content']
        content = response.choices[0].message.content
        self.logger.info(f"For the prompt: {prompt}\n\nGot the output (from LLM): {content}")
#         content = '''
# Here is the model output:

# ```json
# {
#   "answer": "The mitochondrion is the powerhouse of the cell.",
#   "reasoning": "Based on the provided biological context, the mitochondria are responsible for ATP production, which fuels cellular processes.",
#   "confidence_score": 0.92,
#   "needs_review": false,
#   "relevant_image_tags": ["2c851fc8-d71d-4b7a-9990-c7b827a55698"],
#   "context_coverage": "High — directly addressed the function of mitochondria based on the input text."
# }
# '''
        parsed_output = self.extractJsonOutput(output_text = content, prompt_type = prompt_type)
        return parsed_output
    
    
    def getLLMResponseGeneral(self, llm_provider, model_name, prompt_template, prompt_type, temp, context_text, user_query):
        llm = self.getLLMProvider(provider=llm_provider, model=model_name)
        llm.temperature = temp
        prompt_template = PromptTemplate(
            input_variables=["context_text", "user_query"],
            template=prompt_template
        )
        
        chain = LLMChain(llm = llm, prompt = prompt_template)
        response = chain.run(context_text = context_text, user_query=user_query)
        parsed_output = self.extractJsonOutput(output_text = response, prompt_type = prompt_type)
        return parsed_output