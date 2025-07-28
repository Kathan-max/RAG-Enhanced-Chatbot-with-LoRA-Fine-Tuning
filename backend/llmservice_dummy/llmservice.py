from llmservice.prompts import *
from llmservice.llmhelper import LLmHelper
from config.settings import *

class LLMService:
    
    def __init__(self):
        self.llm_helper = LLmHelper(generation_model = DEFAULT_GENERATION_MODEL)
    
    def directLLMResponse(self, llm_to_use, user_query, context_text, temp):
        # prompt_to_use = PROMPTS.get("DirectResponsePrompt").replace("{context_text}", context_text).replace("{user_query}", user_query)
        # parsed_output = self.llm_helper.getLLMResponse(prompt = prompt_to_use,
        #                                                llm_name = llm_to_use,
        #                                                temp = temp,
        #                                                prompt_type = "DirectResponsePrompt")
        parsed_output = self.llm_helper.getLLMResponseGeneral(llm_provider =  "openai", model_name = llm_to_use, 
                                                              prompt_template = PROMPTS.get("DirectResponsePrompt"), 
                                                              temp = temp, context_text = context_text, 
                                                              user_query = user_query, prompt_type = "DirectResponsePrompt")
        return parsed_output
    
    def get_provider(self, master_model):
        if "gpt" in master_model:
            return "openai"
        elif "gemini" in master_model:
            return "gemini"
        elif "mistral" in master_model:
            return "mistral"
        else:
            return "claude"
    
    def getMasterResponse(self, master_llm_provider, master_model, user_query, context_text, temp):
        return self.llm_helper.getLLMResponseGeneral(llm_provider = master_llm_provider, model_name = master_model, 
                                                     prompt_template = PROMPTS.get("MasterLLMPrompt"), 
                                                     temp = temp, context_text = context_text,
                                                     user_query = user_query, prompt_type = "MasterLLMPrompt")
        
    def masterLLMProcessing(self, user_query, retieved_chunks, masterLLM, temp = DEFAULT_TEMPERATURE, active_llm = [], use_multLLM = False):
        context_text = "\n\n".join([f"Context {i+1}: {chunk['content']}" for i, chunk in enumerate(retieved_chunks)])
        if not use_multLLM:
            return self.directLLMResponse(llm_to_use=masterLLM, user_query=user_query, context_text=context_text, temp = temp) 
        
        master_llm_provider = self.get_provider(master_model)
        master_output = self.getMasterResponse(master_llm_provider, master_model = masterLLM, user_query = user_query, context_text = context_text, temp = temp)
        stop_indicator = False
        if "needs_review" in master_output:
            if master_output['needs_review'] == "true":
                stop_indicator = False
            else:
                stop_indicator = True
            
        while(not stop_indicator):
            
            pass
        