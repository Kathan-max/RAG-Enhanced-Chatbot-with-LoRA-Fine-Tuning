from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.base import BaseTool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from typing import Optional, Dict, List, Any, Union
import json
import logging
from dataclasses import dataclass
from enum import Enum
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ProcessingMode(Enum):
    SINGLE_LLM = "single"
    MULTI_LLM_JURY = "jury"
    CHAIN_OF_THOUGHTS = "cot"

@dataclass
class LLMResponse:
    content: str
    confidence_score: float
    reasoning: str
    metadata: Dict[str, Any]

@dataclass
class PipelineState:
    user_query: str
    context_text: str
    master_response: Optional[LLMResponse] = None
    jury_responses: List[LLMResponse] = None
    final_response: Optional[LLMResponse] = None
    current_iteration: int = 0
    max_iterations: int = 3
    processing_mode: ProcessingMode = ProcessingMode.MULTI_LLM_JURY
    continue_pipeline: bool = True

class LLMProviderTool(BaseTool):
    """Base tool for LLM provider interactions"""
    name = "llm_provider"
    description = "Executes LLM requests with specified provider and prompt"
    
    def __init__(self, providers_config: Dict[str, Any]):
        super().__init__()
        self.providers_config = providers_config
        self._setup_providers()
    
    def _setup_providers(self):
        """Initialize LLM providers"""
        os.environ["OPENAI_API_KEY"] = os.getenv('OPEN_AI_API_KEY')
        os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
        os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    
    def get_llm_provider(self, provider: str, model: str, temperature: float = 0.7):
        """Get LLM provider instance"""
        provider = provider.lower()
        if "openai" in provider:
            return ChatOpenAI(model=model or "gpt-4o", temperature=temperature)
        elif "claude" in provider:
            return ChatAnthropic(model=model or "claude-3-opus-20240229", temperature=temperature)
        elif "gemini" in provider:
            return ChatGoogleGenerativeAI(model=model or "gemini-pro", temperature=temperature)
        elif "mistral" in provider:
            return HuggingFaceHub(repo_id=model or "mistralai/Mistral-7B-Instruct-v0.1")
        else:
            raise ValueError(f"Unsupported LLM Provider: {provider}")
    
    def _run(self, provider: str, model: str, prompt_template: str, 
             context_text: str = "", user_query: str = "", 
             temperature: float = 0.7, **kwargs) -> str:
        """Execute LLM request"""
        try:
            llm = self.get_llm_provider(provider, model, temperature)
            
            # Create prompt template
            prompt = PromptTemplate.from_template(prompt_template)
            chain = LLMChain(llm=llm, prompt=prompt)
            
            # Execute with parameters
            response = chain.run(
                context_text=context_text, 
                user_query=user_query,
                **kwargs
            )
            
            return response
        except Exception as e:
            return f"Error executing LLM request: {str(e)}"

class ContextRelevanceTool(BaseTool):
    """Tool for evaluating context relevance"""
    name = "context_relevance_evaluator"
    description = "Evaluates the relevance and sufficiency of context for answering the query"
    
    def __init__(self, llm_provider_tool: LLMProviderTool):
        super().__init__()
        self.llm_provider = llm_provider_tool
    
    def _run(self, context_text: str, user_query: str) -> str:
        prompt_template = """You are evaluating the relevance and sufficiency of provided context for answering the user query.

**CONTEXT:**
{context_text}

**QUERY:**
{user_query}

**TASK:**
Assess if the context is sufficient, relevant, and appropriate for answering the query.

**OUTPUT FORMAT:**
```json
{{
  "relevance_score": 8,
  "sufficiency_score": 7,
  "missing_elements": ["What key information is missing"],
  "context_quality": "excellent|good|fair|poor",
  "recommendation": "How to proceed given context quality",
  "image_relevance": "How well image descriptions support the query"
}}
```"""
        
        return self.llm_provider._run(
            provider="openai",
            model="gpt-4o",
            prompt_template=prompt_template,
            context_text=context_text,
            user_query=user_query
        )

class MasterLLMTool(BaseTool):
    """Tool for Master LLM operations"""
    name = "master_llm_processor"
    description = "Handles Master LLM processing including initial response, opinion, and evaluation"
    
    def __init__(self, llm_provider_tool: LLMProviderTool, prompts: Dict[str, str]):
        super().__init__()
        self.llm_provider = llm_provider_tool
        self.prompts = prompts
    
    def _run(self, operation: str, **kwargs) -> str:
        """Execute Master LLM operation"""
        if operation == "initial_response":
            return self._initial_response(**kwargs)
        elif operation == "opinion_check":
            return self._opinion_check(**kwargs)
        elif operation == "final_evaluation":
            return self._final_evaluation(**kwargs)
        elif operation == "conflict_resolution":
            return self._conflict_resolution(**kwargs)
        else:
            return f"Unknown operation: {operation}"
    
    def _initial_response(self, context_text: str, user_query: str, 
                         provider: str = "openai", model: str = "gpt-4o") -> str:
        return self.llm_provider._run(
            provider=provider,
            model=model,
            prompt_template=self.prompts["DirectResponsePrompt"],
            context_text=context_text,
            user_query=user_query
        )
    
    def _opinion_check(self, context_text: str, user_query: str, 
                      master_answer: str, jury_response: str = "",
                      provider: str = "openai", model: str = "gpt-4o") -> str:
        return self.llm_provider._run(
            provider=provider,
            model=model,
            prompt_template=self.prompts["MasterOpinionPrompt"],
            context_text=context_text,
            user_query=user_query,
            master_answer=master_answer,
            jury_response=jury_response
        )
    
    def _final_evaluation(self, context_text: str, user_query: str,
                         master_answer: str, all_jury_responses: str,
                         provider: str = "openai", model: str = "gpt-4o") -> str:
        return self.llm_provider._run(
            provider=provider,
            model=model,
            prompt_template=self.prompts["MasterEvaluationPrompt"],
            context_text=context_text,
            user_query=user_query,
            master_answer=master_answer,
            all_jury_responses=all_jury_responses
        )
    
    def _conflict_resolution(self, context_text: str, user_query: str,
                           conflicting_responses: str,
                           provider: str = "openai", model: str = "gpt-4o") -> str:
        return self.llm_provider._run(
            provider=provider,
            model=model,
            prompt_template=self.prompts["ConflictResolutionPrompt"],
            context_text=context_text,
            user_query=user_query,
            conflicting_responses=conflicting_responses
        )

class JuryLLMTool(BaseTool):
    """Tool for Jury LLM operations"""
    name = "jury_llm_processor"
    description = "Handles Jury LLM processing for reviewing and improving responses"
    
    def __init__(self, llm_provider_tool: LLMProviderTool, prompts: Dict[str, str]):
        super().__init__()
        self.llm_provider = llm_provider_tool
        self.prompts = prompts
    
    def _run(self, context_text: str, user_query: str, previous_answer: str,
             provider: str = "claude", model: str = "claude-3-opus-20240229") -> str:
        """Execute Jury LLM review"""
        return self.llm_provider._run(
            provider=provider,
            model=model,
            prompt_template=self.prompts["SlaveRescursivePrompt"],
            context_text=context_text,
            user_query=user_query,
            previous_answer=previous_answer
        )

class ChainOfThoughtsTool(BaseTool):
    """Tool for Chain of Thoughts processing"""
    name = "chain_of_thoughts_processor"
    description = "Handles iterative self-improvement through Chain of Thoughts"
    
    def __init__(self, llm_provider_tool: LLMProviderTool, prompts: Dict[str, str]):
        super().__init__()
        self.llm_provider = llm_provider_tool
        self.prompts = prompts
    
    def _run(self, context_text: str, user_query: str, previous_response: str,
             iteration_number: int, provider: str = "openai", model: str = "gpt-4o") -> str:
        """Execute Chain of Thoughts iteration"""
        return self.llm_provider._run(
            provider=provider,
            model=model,
            prompt_template=self.prompts["ChainOfThoughtsPrompt"],
            context_text=context_text,
            user_query=user_query,
            previous_response=previous_response,
            iteration_number=iteration_number
        )

class MultiLLMOrchestrator:
    """Main orchestrator for the multi-LLM pipeline using LangChain agents"""
    
    def __init__(self, prompts: Dict[str, str], providers_config: Dict[str, Any]):
        self.prompts = prompts
        self.providers_config = providers_config
        
        # Initialize tools
        self.llm_provider_tool = LLMProviderTool(providers_config)
        self.context_relevance_tool = ContextRelevanceTool(self.llm_provider_tool)
        self.master_llm_tool = MasterLLMTool(self.llm_provider_tool, prompts)
        self.jury_llm_tool = JuryLLMTool(self.llm_provider_tool, prompts)
        self.cot_tool = ChainOfThoughtsTool(self.llm_provider_tool, prompts)
        
        # Setup agent
        self._setup_agent()
    
    def _setup_agent(self):
        """Setup the orchestrator agent"""
        tools = [
            self.context_relevance_tool,
            self.master_llm_tool,
            self.jury_llm_tool,
            self.cot_tool
        ]
        
        # Create orchestrator LLM
        orchestrator_llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
        
        # Create agent prompt
        agent_prompt = PromptTemplate.from_template("""
You are the Multi-LLM Pipeline Orchestrator. Your job is to coordinate the flow of a sophisticated RAG pipeline.

Available tools:
{tools}

Current pipeline state:
- User Query: {user_query}
- Context: {context_text}
- Processing Mode: {processing_mode}
- Current Iteration: {current_iteration}
- Max Iterations: {max_iterations}

Your task is to execute the appropriate sequence of tool calls based on the processing mode:

1. SINGLE_LLM: Use master_llm_processor with "initial_response" operation
2. MULTI_LLM_JURY: Execute full jury pipeline with master coordination
3. CHAIN_OF_THOUGHTS: Use chain_of_thoughts_processor for iterative improvement

Always start by evaluating context relevance, then proceed with the appropriate processing mode.

{agent_scratchpad}
""")
        
        # Create agent
        self.agent = create_react_agent(orchestrator_llm, tools, agent_prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=tools,
            verbose=True,
            max_iterations=10,
            handle_parsing_errors=True
        )
    
    def process_query(self, user_query: str, context_text: str, 
                     processing_mode: ProcessingMode = ProcessingMode.MULTI_LLM_JURY,
                     max_iterations: int = 3) -> Dict[str, Any]:
        """Process user query through the multi-LLM pipeline"""
        
        pipeline_state = PipelineState(
            user_query=user_query,
            context_text=context_text,
            processing_mode=processing_mode,
            max_iterations=max_iterations
        )
        
        try:
            if processing_mode == ProcessingMode.SINGLE_LLM:
                return self._single_llm_process(pipeline_state)
            elif processing_mode == ProcessingMode.MULTI_LLM_JURY:
                return self._multi_llm_jury_process(pipeline_state)
            elif processing_mode == ProcessingMode.CHAIN_OF_THOUGHTS:
                return self._chain_of_thoughts_process(pipeline_state)
            else:
                raise ValueError(f"Unknown processing mode: {processing_mode}")
        
        except Exception as e:
            logging.error(f"Error in pipeline processing: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "pipeline_state": pipeline_state
            }
    
    def _single_llm_process(self, state: PipelineState) -> Dict[str, Any]:
        """Execute single LLM processing"""
        # Context relevance check
        relevance_result = self.context_relevance_tool._run(
            state.context_text, state.user_query
        )
        
        try:
            relevance_data = json.loads(relevance_result)
            if relevance_data.get("context_quality") == "poor":
                return {
                    "success": False,
                    "message": "Context quality insufficient",
                    "relevance_assessment": relevance_data
                }
        except json.JSONDecodeError:
            pass  # Continue with processing
        
        # Master LLM initial response
        master_response = self.master_llm_tool._run(
            operation="initial_response",
            context_text=state.context_text,
            user_query=state.user_query
        )
        
        return {
            "success": True,
            "final_response": master_response,
            "processing_mode": "single_llm",
            "relevance_assessment": relevance_result
        }
    
    def _multi_llm_jury_process(self, state: PipelineState) -> Dict[str, Any]:
        """Execute multi-LLM jury processing"""
        results = {
            "success": True,
            "processing_mode": "multi_llm_jury",
            "iterations": []
        }
        
        # Context relevance check
        relevance_result = self.context_relevance_tool._run(
            state.context_text, state.user_query
        )
        results["relevance_assessment"] = relevance_result
        
        # Master initial response
        master_response = self.master_llm_tool._run(
            operation="initial_response",
            context_text=state.context_text,
            user_query=state.user_query
        )
        
        current_response = master_response
        jury_responses = []
        
        # Jury iterations
        for iteration in range(state.max_iterations):
            # Master opinion check
            opinion_result = self.master_llm_tool._run(
                operation="opinion_check",
                context_text=state.context_text,
                user_query=state.user_query,
                master_answer=master_response,
                jury_response="\n".join(jury_responses) if jury_responses else ""
            )
            
            try:
                opinion_data = json.loads(opinion_result)
                if not opinion_data.get("continue_pipeline", True):
                    break
            except json.JSONDecodeError:
                pass
            
            # Jury LLM processing
            jury_providers = ["claude", "gemini", "openai"]
            jury_models = ["claude-3-opus-20240229", "gemini-pro", "gpt-4o"]
            
            provider = jury_providers[iteration % len(jury_providers)]
            model = jury_models[iteration % len(jury_models)]
            
            jury_response = self.jury_llm_tool._run(
                context_text=state.context_text,
                user_query=state.user_query,
                previous_answer=current_response,
                provider=provider,
                model=model
            )
            
            jury_responses.append(jury_response)
            current_response = jury_response
            
            results["iterations"].append({
                "iteration": iteration + 1,
                "provider": provider,
                "model": model,
                "jury_response": jury_response,
                "opinion_check": opinion_result
            })
        
        # Final evaluation
        final_response = self.master_llm_tool._run(
            operation="final_evaluation",
            context_text=state.context_text,
            user_query=state.user_query,
            master_answer=master_response,
            all_jury_responses="\n---\n".join(jury_responses)
        )
        
        results["master_initial"] = master_response
        results["jury_responses"] = jury_responses
        results["final_response"] = final_response
        
        return results
    
    def _chain_of_thoughts_process(self, state: PipelineState) -> Dict[str, Any]:
        """Execute Chain of Thoughts processing"""
        results = {
            "success": True,
            "processing_mode": "chain_of_thoughts",
            "iterations": []
        }
        
        # Context relevance check
        relevance_result = self.context_relevance_tool._run(
            state.context_text, state.user_query
        )
        results["relevance_assessment"] = relevance_result
        
        # Initial response
        current_response = self.master_llm_tool._run(
            operation="initial_response",
            context_text=state.context_text,
            user_query=state.user_query
        )
        
        # Chain of Thoughts iterations
        for iteration in range(state.max_iterations):
            cot_response = self.cot_tool._run(
                context_text=state.context_text,
                user_query=state.user_query,
                previous_response=current_response,
                iteration_number=iteration + 1
            )
            
            results["iterations"].append({
                "iteration": iteration + 1,
                "response": cot_response
            })
            
            try:
                cot_data = json.loads(cot_response)
                current_response = cot_data.get("improved_answer", current_response)
                
                if not cot_data.get("continue_iteration", True):
                    break
            except json.JSONDecodeError:
                current_response = cot_response
        
        results["final_response"] = current_response
        return results

# Example usage and testing
if __name__ == "__main__":
    # Your PROMPTS dictionary here (from your original code)
    PROMPTS = {
        "DirectResponsePrompt": """You are an AI assistant that uses the provided context to answer the user's question. 

**CONTEXT HANDLING:**
- Use provided context passages to answer the user's question accurately
- Some context may include image tags like: <image_dec><image_id>hash_123</image_id>Description text</image_dec>
- Include relevant <image_id>hash_123</image_id> tags in your answer where images would help understanding
- Do NOT describe images yourself - only use the provided descriptions

**TASK:**
1. Provide a detailed answer based on context and query

**OUTPUT FORMAT:**
```json
{{
  "answer": "<Your complete answer with <image_id>hash</image_id> tags where appropriate>",
  "reasoning": "<How you derived this answer from the context>", 
  "relevant_image_tags": ["hash_1", "hash_2"],
  "context_coverage": "<How well the context addresses the query (complete/partial/limited)>"
}}
```

**CONTEXT:**
{context_text}

**QUERY:**
{user_query}""",
        # Add other prompts here...
    }
    
    providers_config = {
        "openai": {"default_model": "gpt-4o"},
        "claude": {"default_model": "claude-3-opus-20240229"},
        "gemini": {"default_model": "gemini-pro"}
    }
    
    # Initialize orchestrator
    orchestrator = MultiLLMOrchestrator(PROMPTS, providers_config)
    
    # Example usage
    user_query = "What is machine learning?"
    context_text = "Machine learning is a subset of artificial intelligence..."
    
    # Single LLM processing
    result = orchestrator.process_query(
        user_query=user_query,
        context_text=context_text,
        processing_mode=ProcessingMode.SINGLE_LLM
    )
    
    print("Single LLM Result:", result)
    
    # Multi-LLM jury processing
    result = orchestrator.process_query(
        user_query=user_query,
        context_text=context_text,
        processing_mode=ProcessingMode.MULTI_LLM_JURY,
        max_iterations=2
    )
    
    print("Multi-LLM Jury Result:", result)