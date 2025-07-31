import json
import re
from llmservice.llmmodels import ProcessingMode

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
        
    def extract_orchestrator_json_block(self, text):
        if '```json' in text:
            pattern = r'```json\n(.*?)\n```'
            match = re.search(pattern, text, re.DOTALL)
            if match:
                json_str = match.group(1)
                return json.loads(json_str)
        return {"answer": text}
    
    def extract_orchestrator_json_block_v2(self, text):
        if '```json' in text:
            pattern = r'```json\n(.*?)\n```'
            match = re.search(pattern, text, re.DOTALL)
            if match:
                json_str = match.group(1)
                try:
                    # First attempt: try parsing as-is
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    print(f"Initial JSON parse failed: {e}")
                    try:
                        # Second attempt: fix common escape sequence issues
                        cleaned_json = self._fix_escape_sequences(json_str)
                        return json.loads(cleaned_json)
                    except json.JSONDecodeError as e:
                        print(f"Cleaned JSON parse failed: {e}")
                        try:
                            # Third attempt: use raw string decoder
                            return self._parse_json_with_raw_strings(json_str)
                        except Exception as e:
                            print(f"Raw string parsing failed: {e}")
                            # Fallback: extract answer manually
                            return self._extract_answer_manually(text)
        return {"answer": text}
    
    def _fix_escape_sequences(self, json_str):
        """Fix common escape sequence issues in JSON strings"""
        # Replace problematic LaTeX escape sequences
        # Handle \\operatorname, \\text, etc.
        json_str = json_str.replace('\\\\operatorname', '\\\\\\\\operatorname')
        json_str = json_str.replace('\\\\text', '\\\\\\\\text')
        json_str = json_str.replace('\\\\', '\\\\\\\\')
        
        # Fix other common escape issues
        json_str = re.sub(r'(?<!\\)\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', json_str)
        
        return json_str
    
    def _parse_json_with_raw_strings(self, json_str):
        """Alternative parsing method for strings with complex escape sequences"""
        try:
            # Use ast.literal_eval approach for safer parsing
            import ast
            
            # Try to extract key-value pairs manually
            result = {}
            
            # Extract answer
            answer_match = re.search(r'"improved_answer":\s*"(.*?)"(?=,\s*"|\s*})', json_str, re.DOTALL)
            if answer_match:
                result["answer"] = answer_match.group(1).replace('\\"', '"')
            
            # Extract reasoning
            reasoning_match = re.search(r'"reasoning":\s*"(.*?)"(?=,\s*"|\s*})', json_str, re.DOTALL)
            if reasoning_match:
                result["reasoning"] = reasoning_match.group(1).replace('\\"', '"')
            
            # Extract relevant_image_tags
            image_tags_match = re.search(r'"relevant_image_tags":\s*\[(.*?)\]', json_str, re.DOTALL)
            if image_tags_match:
                tags_content = image_tags_match.group(1)
                # Extract individual tags
                tag_matches = re.findall(r'"([^"]+)"', tags_content)
                result["relevant_image_tags"] = tag_matches
            else:
                result["relevant_image_tags"] = []
            
            # Extract other fields if needed
            for field in ["improvement_strategy", "confidence_score", "iteration_summary"]:
                field_match = re.search(rf'"{field}":\s*"(.*?)"(?=,\s*"|\s*}})', json_str, re.DOTALL)
                if field_match:
                    result[field] = field_match.group(1).replace('\\"', '"')
                elif field == "confidence_score":
                    # Handle numeric confidence score
                    field_match = re.search(rf'"{field}":\s*(\d+)', json_str)
                    if field_match:
                        result[field] = int(field_match.group(1))
            
            return result
            
        except Exception as e:
            print(f"Raw string parsing error: {e}")
            raise e
    
    def _extract_answer_manually(self, text):
        """Fallback method to extract at least the answer from malformed JSON"""
        try:
            # Try to find the improved_answer field
            answer_pattern = r'"improved_answer":\s*"(.*?)"'
            match = re.search(answer_pattern, text, re.DOTALL)
            if match:
                answer = match.group(1)
                # Clean up the answer
                answer = answer.replace('\\"', '"')
                answer = answer.replace('\\n', '\n')
                return {
                    "answer": answer,
                    "reasoning": "Extracted from malformed JSON",
                    "relevant_image_tags": []
                }
        except Exception as e:
            print(f"Manual extraction failed: {e}")
        
        # Ultimate fallback
        return {
            "answer": "Unable to parse response. Please try again.",
            "reasoning": "JSON parsing failed",
            "relevant_image_tags": []
        }
    
    def _safe_json_loads(self, json_str):
        """Safely load JSON with better error handling"""
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSON decode error at position {e.pos}: {e.msg}")
            # Show context around the error
            start = max(0, e.pos - 50)
            end = min(len(json_str), e.pos + 50)
            context = json_str[start:end]
            print(f"Context around error: ...{context}...")
            raise e
    
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