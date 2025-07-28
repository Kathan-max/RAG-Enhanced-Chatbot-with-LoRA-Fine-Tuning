import json
import re


class AdaptiveJsonExtractor:
    
    def __init__(self):
        self.prompt_schemas = {
            'DirectResponsePrompt': {
                "required": ["answer", "reasoning", "context_coverage", "relevant_image_tags"],
                "optional": []
            }, 
            'MasterLLMPrompt': {
                "required": ["answer", "reasoning", "confidence_score", "needs_review", "relevant_image_tags", "context_coverage"],
                "optional": ["review_reason"]    
            }, 
            'MasterOpinionPrompt': {
                "required": ["continue_pipeline", "reasoning", "current_quality_score"],
                "optional": ["improvement_needed", "estimated_remaining_value"]
            }, 
            'MasterEvaluationPrompt': {
                "required": ["final_answer", "synthesis_reasoning", "confidence_score"],
                "optional": ["key_improvements", "relevant_image_tags", "context_fidelity"]
            }, 
            'SlaveRescursivePrompt': {
                "required": ["action", "reasoning", "confidence_score", "relevant_image_tags"],
                "optional": ["improved_answer", "changes_made", "additional_context_used"]
            }, 
            'ConflictResolutionPrompt': {
                "required": ["resolved_answer", "conflict_analysis", "confidence_score"],
                "optional": ["context_support", "relevant_image_tags"]
            }, 
            'ContextRelevancePrompt': {
                "required": ["relevance_score", "sufficiency_score", "context_quality"],
                "optional": ["missing_elements", "recommendation", "image_relevance"]
            }, 
            'ChainOfThoughtsPrompt': {
                "required": ["improved_answer", "improvement_strategy", "confidence_score", "continue_iteration", "relevant_image_tags"],
                "optional": ["changes_made", "reasoning", "iteration_summary", "next_focus", "quality_progression"]
            }   
        }
        
        self.common_fields = [
            "answer", "improved_answer", "final_answer", "resolved_answer",
            "reasoning", "synthesis_reasoning", "conflict_analysis",
            "relevant_image_tags", "confidence_score", "continue_iteration",
            "needs_review", "continue_pipeline"
        ]
        
    def extract_orchestrator_json_block(self, text, processing_mode):
        pass
    
    
    
    def extract_json_block(self, text, prompt_type):
        
        parsed_json = self._find_and_parse_json(text)
        if not parsed_json:
            return None
        
        if prompt_type and prompt_type in self.prompt_schemas:
            return self._extract_by_schema(parsed_json, prompt_type)
        
        detected_type = self._detect_prompt_type(parsed_json)
        if detected_type:
            return self._extract_by_schema(parsed_json, detected_type)
        
        return self._extract_common_fields(parsed_json)

    def _find_and_parse_json(self, text):
        json_obj = self._extract_with_brace_matching(text)
        if json_obj:
            return json_obj
        
        json_obj = self._extract_with_regex(text)
        if json_obj:
            return json_obj
        
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass
        
        return None
    
    def _extract_with_brace_matching(self, text):
        brace_stack = []
        json_start = -1
        
        for i, char in enumerate(text):
            if char == '{':
                if json_start == -1:
                    json_start = i
                brace_stack.append('{')
            elif char == '}':
                if brace_stack:
                    brace_stack.pop()
                    if not brace_stack:
                        json_end = i + 1
                        json_str = text[json_start: json_end]
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError:
                            json_start = -1
                            continue
        return None
    
    def _extract_with_regex(self, text):
        patterns = [
            r'```json\s*(\{.*?\})\s*```',
            r'```\s*(\{.*?\})\s*```',
            r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})',  # Nested braces
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)   
                except json.JSONDecodeError:
                    continue
                    
        return None
    
    def _detect_prompt_type(self, parsed_json):
        json_keys = set(parsed_json.keys())
        
        best_match = None
        best_score = 0
        
        for prompt_type, schema in self.prompt_schemas.items():
            required_keys = set(schema["required"])
            optional_keys = set(schema.get("optional", []))
            all_schema_keys = required_keys | optional_keys
            
            required_matches = len(required_keys & json_keys)
            optional_matches = len(optional_keys & json_keys)
            
            extra_keys = len(json_keys - all_schema_keys)
            
            score = (required_matches*2) + optional_matches - (extra_keys * 0.5)
            
            if required_matches >= len(required_keys)*0.5 and score > best_score:
                best_score = score
                best_match = prompt_type
        
        return best_match
    
    def _extract_by_schema(self, parsed_json, prompt_type):
        schema = self.prompt_schemas[prompt_type]
        extracted = {"_prompt_type": prompt_type}
        
        for key in schema['required']:
            extracted[key] = parsed_json.get(key)
        
        for key in schema.get("optional", []):
            if key in parsed_json:
                extracted[key] = parsed_json[key]
        
        for key, value in parsed_json.items():
            if key not in extracted:
                extracted[f"extra_{key}"] = value
        
        return extracted
    
    def _extract_common_fields(self, parsed_json):
        extracted = {"_prompt_type": "unknown"}
        
        for key in self.common_fields:
            if key in parsed_json:
                extracted[key] = parsed_json[key]
        for key, value in parsed_json.items():
            if key not in extracted:
                extracted[key] = value
        return extracted
    


# if __name__ == "__main__":
#     raw_output = '''
# Here is the model output:

# ```json
# {
#   "answer": "The mitochondrion is the powerhouse of the cell.",
#   "reasoning": "Based on the provided biological context, the mitochondria are responsible for ATP production, which fuels cellular processes.",
#   "confidence_score": 0.92,
#   "needs_review": false,
#   "relevant_image_tags": ["<image_id>bio_001</image_id>"],
#   "context_coverage": "High — directly addressed the function of mitochondria based on the input text."
# }
# '''
#     extractor = AdaptiveJsonExtractor()
#     output = extractor.extract_json_block(text=raw_output, prompt_type="MasterLLMPrompt")
#     print(output)