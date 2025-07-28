from enum import Enum
from dataclasses import dataclass
from typing import Any
from typing import List, Dict, Optional


class ProcessingMode(Enum):
    SINGLE_LLM = "single"
    MULTI_LLM_JURY = "jury"
    CHAIN_OF_THOUGHTS = "cot"

@dataclass
class LLMReponse:
    content: str
    confidence_score: float
    reasoning: str
    metadata: Dict[str, Any] # for images

@dataclass
class PipelineState:
    user_query: str
    context_text: str
    master_respons: Optional[LLMReponse] = None
    jury_responses: List[LLMReponse] = None
    final_response: Optional[LLMReponse] = None
    current_iterations: int = 0
    max_iterations: int = 3
    processing_mode: ProcessingMode = ProcessingMode.SINGLE_LLM
    continue_pipeline: bool = True
