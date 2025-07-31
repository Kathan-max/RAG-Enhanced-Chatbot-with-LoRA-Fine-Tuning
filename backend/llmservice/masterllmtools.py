from langchain.tools.base import BaseTool
from llmservice.llmprovider import LLMProviderTool
# from llmprovider import LLMProviderTool
from typing import Dict

class MasterLLMTool(BaseTool):
    model_config = {'arbitrary_types_allow7ed': True, 'extra': 'allow'}
    name: str = "master_llm_processor"
    description: str = "Handles Master LLM processing including initial response, opinion, and evaluation"
    
    def __init__(self, llm_provider_tool: LLMProviderTool, prompts: Dict[str, str]):
        super().__init__()
        self.llm_provider = llm_provider_tool
        self.prompts = prompts
    
    def _run(self, operations: str, **kwargs):
        if operations == "initial_response":
            return self._initial_response(**kwargs)
        elif operations == "opinion_check":
            return self._opinion_check(**kwargs)
        elif operations == "final_evaluation":
            return self._final_evaluation(**kwargs)
        elif operations == "conflict_resolution":
            return self._conflict_resolution(**kwargs)
        else:
            return f"unknown Operations: {operations}"
    
    def _initial_response(self, context_text, user_query, provider = "openai", model = "gpt-4o", temperature = 0.1):
        return self.llm_provider._run(
            provider=provider,
            model=model,
            prompt_template=self.prompts["DirectResponsePrompt"],
            context_text=context_text,
            user_query=user_query,
            temperature = temperature
        )
    
    def _opinion_check(self, context_text, user_query, master_answer, jury_response, provider = "openai", model = "gpt-4o", temperature = 0.1):
        return self.llm_provider._run(
            provider=provider,
            model = model,
            prompt_template=self.prompts['MasterOpinionPrompt'],
            context_text=context_text,
            user_query=user_query,
            master_answer = master_answer,
            jury_response = jury_response,
            temperature = temperature
        )
    
    def _final_evaluation(self, context_text, user_query, master_answer, all_jury_response, provider = "openai", model = "gpt-4o", temperature = 0.1):
        return self.llm_provider._run(
            provider=provider,
            model = model,
            prompt_template=self.prompts['MasterEvaluationPrompt'],
            context_text=context_text,
            user_query=user_query,
            master_answer =master_answer,
            all_jury_response = all_jury_response,
            temperature=temperature
        )

class JuryLLMTool(BaseTool):
    model_config = {'arbitrary_types_allowed': True, 'extra': 'allow'}
    name: str = "jury_llm_processor"
    description: str = "Handles Jury LLM processing for reviewing and improving responses"
    
    def __init__(self, llm_provider_tool: LLMProviderTool, prompts):
        super().__init__()
        self.llm_provider = llm_provider_tool
        self.prompts = prompts
    
    def _run(self, context_text, user_query, previous_answer, provider, model, temperature = 0.1):
        return self.llm_provider._run(
            provider=provider,
            model = model,
            prompt_template=self.prompts['SlaveRescursivePrompt'],
            context_text=context_text,
            user_query=user_query,
            temperature=temperature,
            previous_answer = previous_answer
        )

class ChainOfThoughtsTool(BaseTool):
    model_config = {'arbitrary_types_allowed': True, 'extra': 'allow'}
    name: str = "chain_of_thoughts_processor"
    description: str = "Handles iterative self-improvement through Chain of Thoughts"
    
    def __init__(self, llm_provider_tool: LLMProviderTool, prompts: Dict[str, str]):
        super().__init__()
        self.llm_provider = llm_provider_tool
        self.prompts = prompts
    
    def _run(self, context_text, user_query, previous_response, iteration_number, provider = "openai", model = "gpt-4o", temperature = 0.1):
        return self.llm_provider._run(
            provider=provider,
            model = model,
            prompt_template=self.prompts['ChainOfThoughtsPrompt'],
            context_text=context_text,
            user_query= user_query,
            previous_response=previous_response,
            iteration_number=iteration_number
        )