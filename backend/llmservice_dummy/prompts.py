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

    # used when we do not have the multiLLM enabled    
    "MasterLLMPrompt": """You are the Master LLM in a multi-LLM RAG pipeline. Your role is to provide the initial comprehensive answer and decide if additional LLM opinions are needed.

**CONTEXT HANDLING:**
- Use provided context passages to answer the user's question accurately
- Some context may include image tags like: <image_dec><image_id>hash_123</image_id>Description text</image_dec>
- Include relevant <image_id>hash_123</image_id> tags in your answer where images would help understanding
- Do NOT describe images yourself - only use the provided descriptions

**TASK:**
1. Provide a comprehensive answer based on context and query
2. Assess if your answer needs additional LLM review (complexity, uncertainty, multiple perspectives needed)
3. Rate your confidence in the answer (1-10 scale)

**OUTPUT FORMAT:**
```json
{{
  "answer": "<Your complete answer with <image_id>hash</image_id> tags where appropriate>",
  "reasoning": "<How you derived this answer from the context>",
  "confidence_score": 8,
  "needs_review": true,
  "review_reason": "<Why additional LLM input would be valuable>",
  "relevant_image_tags": ["hash_1", "hash_2"],
  "context_coverage": "<How well the context addresses the query (complete/partial/limited)>"
}}
```

**CONTEXT:**
{context_text}

**QUERY:**
{user_query}""",

# ------
 # We will get yes or no from the master and also the Master's output,
    "MasterOpinionPrompt": """You are the Master LLM evaluating whether to continue the multi-LLM pipeline after receiving input from a jury LLM.

**ORIGINAL CONTEXT:**
{context_text}

**ORIGINAL QUERY:**
{user_query}

**YOUR INITIAL ANSWER:**
{master_answer}

**JURY LLM RESPONSE:**
{jury_response}

**TASK:**
Evaluate if the jury LLM's input significantly improved the answer or if more jury input is needed.

**DECISION CRITERIA:**
- Did jury add valuable new insights?
- Are there remaining gaps or uncertainties?
- Would another perspective help?
- Is the answer now comprehensive enough?

**OUTPUT FORMAT:**
```json
{{
  "continue_pipeline": true,
  "reasoning": "<Why continue/stop the pipeline>",
  "current_quality_score": 8,
  "improvement_needed": "<What aspects need more work>",
  "estimated_remaining_value": "<Expected benefit from additional LLMs (high/medium/low)>"
}}
```""",

# -----
 # Other LLM's output evaluation and Move further based on that or not or just use the previous one.
    "MasterEvaluationPrompt": """You are the Master LLM performing final evaluation and synthesis of all jury LLM responses.

**ORIGINAL CONTEXT:**
{context_text}

**ORIGINAL QUERY:**
{user_query}

**YOUR INITIAL ANSWER:**
{master_answer}

**ALL JURY RESPONSES:**
{all_jury_responses}

**TASK:**
Synthesize all inputs into the best possible final answer, incorporating valuable insights while maintaining accuracy.

**SYNTHESIS GUIDELINES:**
- Preserve factual accuracy from context
- Include best insights from each LLM
- Resolve any contradictions logically
- Maintain proper image tag placement
- Ensure coherent, comprehensive response

**OUTPUT FORMAT:**
```json
{{
  "final_answer": "<Synthesized answer incorporating best elements from all responses>",
  "synthesis_reasoning": "<How you combined different LLM inputs>",
  "confidence_score": 9,
  "key_improvements": ["<What each jury LLM contributed>"],
  "relevant_image_tags": ["hash_1", "hash_2"],
  "context_fidelity": "<How well final answer stays grounded in provided context>"
}}
```""",
    
# ----
# We will use this when this LLM is a Slave and not the Master to add or change the previous output.
    "SlaveRescursivePrompt": """You are a Jury LLM in a multi-LLM RAG pipeline. Your role is to review and improve upon the current answer.

**ORIGINAL CONTEXT:**
{context_text}

**USER QUERY:**
{user_query}

**CURRENT ANSWER (from previous LLM):**
{previous_answer}

**TASK:**
Review the current answer and decide whether to:
1. **ENHANCE**: Add valuable information, nuance, or alternative perspectives
2. **CORRECT**: Fix factual errors or misinterpretations
3. **RESTRUCTURE**: Improve organization or clarity
4. **VALIDATE**: Confirm the answer is already comprehensive and accurate

**EVALUATION CRITERIA:**
- Factual accuracy against context
- Completeness of answer
- Clarity and organization
- Proper use of image references
- Alternative perspectives or insights

**OUTPUT FORMAT:**
```json
{{
  "action": "ENHANCE|CORRECT|RESTRUCTURE|VALIDATE",
  "improved_answer": "<Your enhanced/corrected answer or 'VALIDATED' if no changes needed>",
  "changes_made": ["<List of specific improvements>"],
  "reasoning": "<Why these changes improve the answer>",
  "confidence_score": 8,
  "relevant_image_tags": ["hash_1", "hash_2"],
  "additional_context_used": "<Any context elements the previous LLM missed>"
}}
```

**IMPORTANT:**
- If validating, still provide confidence score and reasoning
- Maintain all relevant image tags from context
- Don't add information not supported by context
- Focus on value-add improvements, not minor rewording""",


# ---
# Additional specialized prompts for enhanced functionality
    "ConflictResolutionPrompt": """You are the Master LLM resolving conflicts between jury LLM responses.

**ORIGINAL CONTEXT:**
{context_text}

**USER QUERY:**
{user_query}

**CONFLICTING RESPONSES:**
{conflicting_responses}

**TASK:**
Resolve conflicts by determining which response elements are most accurate according to the provided context.

**OUTPUT FORMAT:**
```json
{{
  "resolved_answer": "<Answer resolving conflicts using context as ground truth>",
  "conflict_analysis": "<What conflicts existed and how resolved>",
  "confidence_score": 8,
  "context_support": "<How context supports resolution decisions>",
  "relevant_image_tags": ["hash_1", "hash_2"]
}}
```""",

        "ContextRelevancePrompt": """You are evaluating the relevance and sufficiency of provided context for answering the user query.

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
  "missing_elements": ["<What key information is missing>"],
  "context_quality": "excellent|good|fair|poor",
  "recommendation": "<How to proceed given context quality>",
  "image_relevance": "<How well image descriptions support the query>"
}}
```""",

#---
# Prompt for the chain of thoughts process
"ChainOfThoughtsPrompt": """You are an LLM in a Chain of Thoughts pipeline, tasked with iteratively improving your previous response through self-reflection and refinement.

**ORIGINAL CONTEXT:**
{context_text}

**USER QUERY:**
{user_query}

**YOUR PREVIOUS RESPONSE (Iteration {iteration_number}):**
{previous_response}

**ITERATION GOAL:**
Analyze your previous response and improve it through deeper reasoning, better organization, or additional insights from the context.

**SELF-EVALUATION CRITERIA:**
1. **Accuracy**: Is everything factually correct according to context?
2. **Completeness**: Did I miss any important information from context?
3. **Clarity**: Is the explanation clear and well-structured?
4. **Reasoning**: Can I provide better logical flow or deeper analysis?
5. **Context Usage**: Did I fully utilize all relevant context including images?

**IMPROVEMENT STRATEGIES:**
- **DEEPEN**: Add more detailed explanations or nuanced understanding
- **RESTRUCTURE**: Improve organization and logical flow
- **EXPAND**: Include overlooked context elements or perspectives  
- **REFINE**: Enhance clarity, precision, or coherence
- **VALIDATE**: Confirm accuracy and fix any errors
- **SYNTHESIZE**: Better connect different pieces of information

**OUTPUT FORMAT:**
```json
{{
  "improved_answer": "<Your refined and improved answer with <image_id>hash</image_id> tags>",
  "improvement_strategy": "DEEPEN|RESTRUCTURE|EXPAND|REFINE|VALIDATE|SYNTHESIZE",
  "changes_made": ["<List specific improvements made>"],
  "reasoning": "<Why these changes make the answer better>",
  "confidence_score": 8,
  "continue_iteration": true,
  "iteration_summary": "<What this iteration accomplished>",
  "next_focus": "<What the next iteration should focus on, if continuing>",
  "relevant_image_tags": ["hash_1", "hash_2"],
  "quality_progression": "<How answer quality has improved from previous iteration>"
}}
```

**ITERATION GUIDELINES:**
- If your previous response was already comprehensive and accurate, focus on refinement
- If continuing iterations, suggest specific areas for the next iteration to focus on
- Set continue_iteration to false when you believe no further improvement is possible
- Each iteration should add meaningful value, not just rephrase

**IMPORTANT:**
- Always stay grounded in the provided context
- Maintain all relevant image tags from context
- Build upon your previous response rather than completely rewriting
- Be honest about limitations and uncertainty"""

}

ADAPTATIONS = {
    'gpt-3.5-turbo': {
        'style': 'concise',
        'complexity': 'moderate',
        'json_strict': True
    },
    'gpt-3.5-turbo-16k': {
        'style': 'concise',
        'complexity': 'moderate',
        'json_strict': True
    },
    'gpt-3': {
        'style': 'concise',
        'complexity': 'moderate',
        'json_strict': True
    },
    'gpt-4': {
        'style': 'comprehensive',
        'complexity': 'high',
        'json_strict': True
    },
    'gpt-4-32k': {
        'style': 'comprehensive',
        'complexity': 'high',
        'json_strict': True
    },
    'gpt-4-turbo': {
        'style': 'comprehensive',
        'complexity': 'high',
        'json_strict': True
    },
    'gpt-4o': {
        'style': 'comprehensive',
        'complexity': 'high',
        'json_strict': True
    },
    'Mistral-7B': {
        'style': 'balanced',
        'complexity': 'moderate',
        'json_strict': True
    },
    'deepseek': {
        'style': 'analytical',
        'complexity': 'high',
        'json_strict': True
    },
    'Llama-2': {
        'style': 'detailed',
        'complexity': 'moderate',
        'json_strict': True
    },
    'Clause 3': {
        'style': 'thoughtful',
        'complexity': 'high',
        'json_strict': True
    },
    'Claude 2': {
        'style': 'thoughtful',
        'complexity': 'high',
        'json_strict': True
    },
    'Claude Instant 2': {
        'style': 'thoughtful',
        'complexity': 'high',
        'json_strict': True
    },
    'Gemini-Pro': {
        'style': 'structured',
        'complexity': 'high',
        'json_strict': True
    },
    'Llama-2': {
        'style': 'direct',
        'complexity': 'moderate',
        'json_strict': True
    }
}