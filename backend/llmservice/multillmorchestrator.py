from llmservice.llmprovider import LLMProviderTool
# from llmprovider import LLMProviderTool
from llmservice.masterllmtools import MasterLLMTool, JuryLLMTool, ChainOfThoughtsTool
# from masterllmtools import MasterLLMTool, JuryLLMTool, ChainOfThoughtsTool
from llmservice.llmmodels import ProcessingMode, LLMReponse, PipelineState
# from llmmodels import ProcessingMode, LLMReponse, PipelineState
from llmservice.prompts import PROMPTS
# from prompts import PROMPTS
from llmservice.adaptiveJsonExtractor import AdaptiveJsonExtractor
# from adaptiveJsonExtractor import AdaptiveJsonExtractor
from prompts import PROMPTS
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.agents import Tool, AgentExecutor, create_react_agent
from utils.logger import Logger
import json

class MultiLLMOrchestrator:
    
    # def __init__(self, providers_config):
    def __init__(self):
        self.prompts = PROMPTS
        # self.providers_config = providers_config
        
        # self.llm_provider_tool = LLMProviderTool(providers_config=providers_config)
        self.llm_provider_tool = LLMProviderTool()
        self.master_llm_tool = MasterLLMTool(self.llm_provider_tool, self.prompts)
        self.jury_llm_tool = JuryLLMTool(self.llm_provider_tool, self.prompts)
        self.cot_tool = ChainOfThoughtsTool(self.llm_provider_tool,self.prompts)
        self.logger = Logger(name="RAGLogger").get_logger()
        self.adaptive_extractor = AdaptiveJsonExtractor()
        self._setup_agent()
    
    def _setup_agent(self):
        tools = [
            self.master_llm_tool,
            self.jury_llm_tool,
            self.cot_tool
        ]
        
        orchestrator_llm = ChatOpenAI(model = "gpt-4o", temperature=0.1)
        agent_prompt = PromptTemplate.from_template("""
You are the Multi-LLM Pipeline Orchestrator. Your job is to coordinate the flow of a sophisticated RAG pipeline.

You have access to the following tools:
{tools}

Tool Names: {tool_names}

Current pipeline state:
- User Query: {input}
- Context: {context_text}
- Processing Mode: {processing_mode}
- Current Iteration: {current_iteration}
- Max Iteration: {max_iterations}

Your task is to execute the appropriate sequence of tool calls based on the processing mode:

1. SINGLE_LLM: Use master_llm_processor with "initial_response" operation
2. MULTI_LLM_JURY: Execute full jury pipeline with master coordination
3. CHAIN_OF_THOUGHTS: Use chain_of_thoughts_processor for iterative improvement

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}
""")
        self.agent = create_react_agent(orchestrator_llm, tools, agent_prompt)
        self.agent_executor = AgentExecutor(
            agent = self.agent,
            tools = tools,
            verbose = True,
            max_iterations = 5,
            handle_parsing_erros = True
        )
    
    def process_query(self, user_query, context_text, processing_mode: ProcessingMode = ProcessingMode.MULTI_LLM_JURY, 
                      max_iterations = 3):
        pipeline_state = PipelineState(
            user_query = user_query,
            context_text = context_text,
            max_iterations = max_iterations,
            processing_mode = processing_mode
        )
        
        try:
            if processing_mode == ProcessingMode.SINGLE_LLM:
                return self._single_llm_process(pipeline_state)
            elif processing_mode == ProcessingMode.MULTI_LLM_JURY:
                return self._multi_llm_process(pipeline_state)
            elif processing_mode == ProcessingMode.CHAIN_OF_THOUGHTS:
                return self._chain_of_thoughts_process(pipeline_state)
            else:
                raise ValueError(f"Unkown processing mode: {processing_mode}")
        
        except Exception as e:
            self.logger.error(f"Error in pipeline processing: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "pipeline_state": pipeline_state
            }
        
    def _single_llm_process(self, state: PipelineState):
        master_response = self.master_llm_tool._run(
            operations = "initial_response",
            context_text = state.context_text,
            user_query = state.user_query
        )
        
        return {
            "success": True,
            "final_response": master_response,
            "processing_mode": "single_llm",
            "relevance_assessment": "not_available"
        }
    
    def _multi_llm_process(self, state: PipelineState):
        results = {
            "success": True,
            "processing_mode": "multi_llm_jury",
            "iterations": [],
            "relevance_assessment": "not_available"
        }
        
        # Master initial response
        master_response = self.master_llm_tool._run(
            operations = "initial_response",
            context_text = state.context_text,
            user_query = state.user_query
        )
        
        current_response = master_response
        jury_responses = []
        
        for iteration in range(state.max_iterations):
            opinion_result = self.master_llm_tool._run(
                operations = "opinion_check",
                context_text = state.context_text,
                user_query = state.user_query,
                master_answer = master_response,
                jury_response = "\n".join(jury_responses) if jury_responses else ""
            )
            
            try:
                # opinion_data = json.loads(opinion_result)
                opinion_data = self.adaptive_extractor.extract_json_block(text = opinion_result, 
                                                       prompt_type="MasterOpinionPrompt")
                if not opinion_data.get("continue_pipeline", True):
                    break
            except Exception as e:
                self.logger.error(f"Json parsing error: {e}")
                raise
            
            jury_providers = ["gemini", "openai"]
            jury_models = ["gemini-pro", "gpt-4o"]
            
            provider = jury_providers[iteration % len(jury_providers)] # circular indexing
            model = jury_models[iteration % len(jury_models)]
            
            jury_response = self.jury_llm_tool._run(
                context_text = state.context_text,
                user_query = state.user_query,
                previous_answer = current_response,
                provider = provider,
                model = model
            )
            
            jury_responses.append(jury_response)
            current_response = jury_response
            
            results['iterations'].append({
                "iteration": iteration + 1,
                "provider": provider,
                "model": model,
                "jury_response": jury_response,
                "opinion_check": opinion_result
            })
        
        final_response = self.master_llm_tool._run(
            operations = "final_evaluation",
            context_text = state.context_text,
            user_query = state.user_query,
            master_answer = master_response,
            all_jury_response="\n---\n".join(jury_responses)
        )
        
        results["master_initial"] = master_response
        results["jury_responses"] = jury_responses
        results["final_response"] = final_response
        
        return results
    
    def _chain_of_thoughts_process(self, state: PipelineState):
        results = {
            "success": True,
            "processing_mode": "chain_of_thoughts",
            "iterations": [],
            "relevance_assessment": "not_available"
        }
        current_response = self.master_llm_tool._run(
            operations="initial_response",
            context_text=state.context_text,
            user_query=state.user_query
        )
        
        for iteration in range(state.max_iterations):
            
            cot_response = self.cot_tool._run(
                context_text=state.context_text,
                user_query=state.user_query,
                previous_response=current_response,
                iteration_number=iteration + 1,
            )

            results['iterations'].append({
                'iteration': iteration + 1,
                "response": cot_response
            })
            
            try:
                cot_data = self.adaptive_extractor.extract_json_block(text = cot_response, prompt_type="ChainOfThoughtsPrompt")
                current_response = cot_data.get("improved_answer", current_response)
                if not cot_data.get("continue_iteration", True):
                    break
            except Exception as e:
                pass
                self.logger.error(f"Json parsing error: {e}")
        
        results['final_response'] = current_response
        return results

if __name__ == "__main__":
    # providers_config = {
    #     "openai": {"default_model": "gpt-4o"},
    #     "gemini": {"default_model": "gemini-pro"}
    # }
    
    # orchestrator = MultiLLMOrchestrator(providers_config=providers_config)
    orchestrator = MultiLLMOrchestrator()
    
    user_query = "What is Machine Learning"
    context_text = "Machine learning is a subset of artificial intelligence..."
    
    result = orchestrator.process_query(
        user_query=user_query,
        context_text=context_text,
        processing_mode=ProcessingMode.MULTI_LLM_JURY
    )
    
    print("Result: ", result)