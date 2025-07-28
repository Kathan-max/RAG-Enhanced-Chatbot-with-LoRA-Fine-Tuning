import os
from langchain.tools.base import BaseTool
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional


# load_dotenv(dotenv_path="./.env")
load_dotenv(dotenv_path="F://RAG//backend//.env")

class LLMProviderTool(BaseTool):
    name: str = "llm_provider"
    description: str = "Executes LLM requests with specified provider and prompt"
    # providers_config: Dict[str, Any]
    
    # def __post_init__(self):
    #     self._setup_providers()
        
    # def __init__(self, providers_config: Dict[str, Any]):
    def __init__(self):
        super().__init__()
        self._setup_providers()
    
    def _setup_providers(self):
        os.environ['OPENAI_API_KEY'] = os.getenv("OPEN_AI_API_KEY")
        os.environ['GOOGLE_API_KEY'] = os.getenv("GEMINI_API_KEY")
        # os.environ['MISTRAL_API_KEY'] = os.getenv("MISTRAL_API_KEY")
        os.environ['ANTHROPIC_API_KEY'] = os.getenv("ANTHROPIC_API_KEY")
        os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    
    def get_llm_provider(self, provider, model, temperature):
        provider = provider.lower()
        if "openai" in provider:
            return ChatOpenAI(model = model or "gpt-4o", temperature=temperature)
        elif "claude" in provider:
            return ChatAnthropic(model = model or "claude-3-opus-20240229", temperature=temperature)
        elif "gemini" in provider or "google" in provider:
            return ChatGoogleGenerativeAI(model = model or "gemini-pro", temperature=temperature)
        elif "mistral" in provider:
            return HuggingFaceHub(repo_id=model or "mistralai/Mistral-7B-Instruct-v0.1")
        else:
            raise ValueError(f"Unsupported LLM Provider: {provider}")
        
    def _run(self, provider, model, prompt_template, context_text, user_query, temperature = 0.1, **kwargs):
        try:
            llm = self.get_llm_provider(provider, model, temperature = temperature)
            
            prompt = PromptTemplate.from_template(prompt_template)
            chain = LLMChain(llm = llm, prompt = prompt)
            
            response = chain.run(
                context_text = context_text,
                user_query = user_query,
                **kwargs
            )
            return response
        except Exception as e:
            return f"Error executing LLM Request: {str(e)}"
        