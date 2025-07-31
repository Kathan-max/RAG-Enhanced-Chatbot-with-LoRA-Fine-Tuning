from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import os
import tempfile
import base64
import asyncio
import random
import json
from dotenv import load_dotenv
from ingestion.ingestion import Ingestaion
from embedding.encoder import Encoder
from config.settings import *
from retriever.retriever import Retriever
# from llmservice.llmservice import LLMService
from llmservice.multillmorchestrator import MultiLLMOrchestrator
from llmservice.adaptiveJsonExtractor import AdaptiveJsonExtractor
from llmservice.llmmodels import ProcessingMode
from utils.logger import Logger
from database.vector.vectorDB import VectorDB

app = FastAPI(title="LLM Chat Application API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatRequest(BaseModel):
    query: str
    multiLLM: bool
    fetchChains: bool
    noOfNeighbours: int
    activeLLMs: List[str] = []
    expertLLM: Optional[str] = None
    chainOfThought: bool

class ChatResponse(BaseModel):
    response: str
    reasoning: str
    miscellaneous: dict = {}
    images: Dict[str, str] = {}

# Dummy vector database storage (in production, use a real vector database)
# vector_db = {
#     "total_vectors": 0,
#     "total_documents": 0,
#     "documents": []
# }
load_dotenv(".env")
encoder = Encoder(encoder_name = os.getenv("ENCODER_NAME"), model_name = os.getenv("ENCODING_MODEL"))
ingestion_obj = Ingestaion(parser = os.getenv("PARSER"),
                 chunker = os.getenv("CHUNKER"),
                 encoder = encoder)
# llmserviceObj = LLMService()
llmserviceObj = MultiLLMOrchestrator()
json_extractor = AdaptiveJsonExtractor()
retriever = Retriever()
logger = Logger(name="RAGLogger").get_logger()
vectordb = VectorDB()

@app.get("/")
async def root():
    return {"message": "LLM Chat Application API is running"}

def filter_image_dict(image_dict, image_ids_lst):
    final_image_dict = {image_id: image_dict[image_id] for image_id in image_ids_lst}
    return final_image_dict

def getContextText(retieved_chunks):
    return "\n\n".join([f"Context {i+1}: {chunk['content']}" for i, chunk in enumerate(retieved_chunks)])

"""
class ChatResponse(BaseModel):
    response: str
    reasoning: str
    miscellaneous: Dict[str, str] = {}
    images: Dict[str, str] = {}
"""

def validate_obj(reqKeys,obj):
    for key in reqKeys:
        if key not in obj:
            return False
    
    return True

def format_data(data):
    return {
        "answer": data['answer'],
        "reasoning": data['reasoning'],
        "relevant_image_tags": data["relevant_image_tags"],
        "miscellaneous": {}
    }

def invalide_data():
    return {
        "answer": "We were unable to process the current request please try again later...", 
        "reasoning": "",
        "relevant_image_tags": [],
        "miscellaneous": {}
    }

def correct_validate_final_response(llmresponse_obj, processing_mode):
    print("llmresponse_obj: ", llmresponse_obj)
    if 'final_response' not in llmresponse_obj:
        print(f"Error while validating the final_response: {llmresponse_obj} has no final_response obj.")
    
    final_response = llmresponse_obj['final_response']
    if processing_mode == ProcessingMode.SINGLE_LLM:
        data = json_extractor.extract_orchestrator_json_block(text = final_response)
        validation = validate_obj(['answer', 'reasoning', 'relevant_image_tags'], data)
        if validation:
            return format_data(data)
            
        else:
            return invalide_data()
            
    elif processing_mode == ProcessingMode.CHAIN_OF_THOUGHTS:
        iterations = llmresponse_obj['iterations']
        miscellaneous_dict = {
            "iteration_info":[]
        }
        combined_reasoning = ""
        image_tags = []
        for ite in iterations:
            ite_no = ite['iteration']
            ite_dict = json_extractor.extract_orchestrator_json_block(text=ite['response'])
            miscellaneous_dict['iteration_info'].append({
                "improved_answer": f"Iteration: {ite_no}, {ite_dict['improved_answer']}",
                "improvement_strategy": ite_dict['improvement_strategy'],
                "iteration_summary": ite_dict['iteration_summary'],
                "confidence_score": ite_dict['confidence_score']
            })
            combined_reasoning += "\n" + "iteration: " + str(ite_no) + str(ite_dict['reasoning'])
            image_tags += ite_dict['relevant_image_tags']
        data = {}
        if "```json" in final_response:
            data = json_extractor.extract_orchestrator_json_block(final_response)
            data['relevant_image_tags'] += image_tags
            data['relevant_image_tags'] = list(set(data['relevant_image_tags']))
        else:
            data = {
                "answer": final_response,
                "reasoning": combined_reasoning,
                "relevant_image_tags": set(image_tags),
            }
        
        data["miscellaneous"] = miscellaneous_dict
        validation = validate_obj(['answer', 'reasoning', 'relevant_image_tags'], data)
        if validation:
            return format_data(data)
        else:
            return invalide_data()
    else:  # MULTI_LLM_JURY mode
        print("Multi LLM Response:")
        if "Error" in final_response:
            print("Jury responses were empty, using master_initial if available.")
            if "master_initial" in llmresponse_obj:
                master_response_dict = json_extractor.extract_orchestrator_json_block(text=llmresponse_obj["master_initial"])
                return {
                    "answer": master_response_dict.get("answer", "No valid answer."),
                    "reasoning": master_response_dict.get("reasoning", ""),
                    "relevant_image_tags": master_response_dict.get("relevant_image_tags", []),
                    "miscellaneous": {"note": "Fallback to master initial due to jury failure"}
                }
            else:
                return {
                    "answer": "No valid answer generated.",
                    "reasoning": "Master and jury both failed.",
                    "relevant_image_tags": [],
                    "miscellaneous": {}
                }
        else:
            final_dict = json_extractor.extract_orchestrator_json_block(text=llmresponse_obj["final_response"])
            return {
                "answer": final_dict.get("answer", "No valid answer."),
                "reasoning": final_dict.get("reasoning", ""),
                "relevant_image_tags": final_dict.get("relevant_image_tags", []),
                "miscellaneous": {"note": "Processed via jury-enhanced response"}
            }
                
        

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    print("Request Received: ",request)
    """
    Handle chat requests with support for MultiLLM configuration
    """
    # try:
    user_query = request.query
    print("Received the query: ", user_query)
    retrieved_chunks, image_dicts = retriever.retrieveTopK(query=user_query, top_k=TOP_K, encoder=encoder, 
                                                           similarity_threshold = SIMILARITY_TH, fetch_chains=request.fetchChains, 
                                                           num_of_neighbors=request.noOfNeighbours)
    
    # self, user_query, retieved_chunks, masterLLM, temp = DEFAULT_TEMPERATURE, use_multLLM = False
    context_text = getContextText(retrieved_chunks)
    # llmresponse = llmserviceObj.masterLLMProcessing(user_query=user_query, retieved_chunks=retrieved_chunks, 
    #                                                 masterLLM=request.expertLLM, temp=DEFAULT_TEMPERATURE, 
    #                                                 use_multLLM = request.multiLLM, active_llm = request.activeLLMs)
    """
    class ChatRequest(BaseModel):
        query: str
    multiLLM: bool
    fetchChains: bool
    noOfNeighbours: int
    activeLLMs: List[str] = []
    expertLLM: Optional[str] = None
    """
    if request.chainOfThought:
        processing_mode = ProcessingMode.CHAIN_OF_THOUGHTS
    elif request.multiLLM:
        processing_mode = ProcessingMode.MULTI_LLM_JURY
    else:
        processing_mode = ProcessingMode.SINGLE_LLM
    
    llmresponse_obj = llmserviceObj.process_query(user_query=user_query, context_text= context_text,
                                                  processing_mode=processing_mode, max_iterations = DEFAULT_THINKING_ITERATIONS)
    
    final_answer_obj = correct_validate_final_response(llmresponse_obj, processing_mode)
    
    print("LLM response: ", final_answer_obj)
    answer = final_answer_obj.get('answer', "Was not able to fetch correct output.")
    reasoning = final_answer_obj.get('reasoning', 'No output...')
    final_image_dict = filter_image_dict(image_dict=image_dicts, image_ids_lst=final_answer_obj.get('relevant_image_tags', []))
    
    return ChatResponse(response=final_answer_obj["answer"], 
                        reasoning=final_answer_obj["reasoning"], 
                        images=final_image_dict,
                        miscellaneous = final_answer_obj["miscellaneous"])
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")

@app.post("/api/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    train_model: bool = Form(False)
):
    """
    Handle document uploads with optional model training
    """
    try:
        uploaded_files = []
        json_files = []
        
        for file in files:
            # Create temporary file
            with open(TEMP_FILES_DOC+file.filename, "wb") as temp_file: 
            # with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_file_path = TEMP_FILES_DOC+file.filename
                print(temp_file_path)
            
            # Process file (dummy processing)
            # file_info = {
            #     "filename": file.filename,
            #     "size": len(content),
            #     "temp_path": temp_file_path,
            #     "train_model": train_model
            # }
            uploaded_files.append(temp_file_path)
            extension = temp_file_path.split('.')[1]
            json_files.append(temp_file_path.replace(TEMP_FILES_DOC, TEMP_FILES_JSONS).replace(extension, ".json"))
            
            
            # Clean up temporary file
            # os.unlink(temp_file_path)
        print("Ingesting the files...")
        ingestion_obj.ingest_files(file_path_lists=uploaded_files, 
                                    save_json=True, 
                                    train_query_opt=train_model)
        print("File ingestion compelete...")
        # removing the temp files
        # Remving the pdf files first
        for pdf_file in os.listdir(TEMP_FILES_DOC):
            file_path = os.path.join(TEMP_FILES_DOC, pdf_file)
            os.unlink(file_path)
        
        # Remving the Json files Now
        for json_file in os.listdir(TEMP_FILES_JSONS):
            file_path = os.path.join(TEMP_FILES_JSONS, json_file)
            os.unlink(file_path)
        
        return {"success": True, "files_processed": len(uploaded_files)}
        
    except Exception as e:
        print("Error: ", e)
        raise HTTPException(status_code=500, detail=f"Error uploading documents: {str(e)}")

@app.get("/api/analytics")
async def get_analytics():
    """
    Get comprehensive vector database analytics
    """
    # try:
    analytics_data = vectordb.get_analytics_data()
    return analytics_data
    # except Exception as e:
    #     logger.error(f"Error fetching analytics: {e}")
        # raise HTTPException(status_code=500, detail=f"Error fetching analytics: {str(e)}")

@app.get("/api/analytics/simple")
async def get_simple_analytics():
    """
    Get simplified vector database statistics
    # """
    # try:
    simple_stats = vectordb.get_simple_stats()
    return simple_stats
    # except Exception as e:
    #     logger.error(f"Error fetching simple analytics: {e}")
        # raise HTTPException(status_code=500, detail=f"Error fetching simple analytics: {str(e)}")

@app.get("/api/health")
async def health_check():
    """
    Enhanced health check endpoint with database status
    """
    # try:
        # Basic health check
    health_status = {"status": "healthy", "message": "API is running properly"}
    
    # Add database connectivity check
    # try:
    simple_stats = vectordb.get_simple_stats()
    health_status["database"] = {
        "status": simple_stats.get("database_status", "unknown"),
        "total_vectors": simple_stats.get("total_vectors", 0),
        "connection": "ok" if "error" not in simple_stats else "error"
    }
        # except Exception as db_error:
        #     health_status["database"] = {
        #         "status": "error",
        #         "error": str(db_error),
        #         "connection": "failed"
        #     }
        
    return health_status
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

# Additional endpoints for LLM configuration
@app.get("/api/llm-models")
async def get_available_llm_models():
    """
    Get list of available LLM models
    """
    return {
        "active_llms": [
            "GPT-4",
            "Claude-3",
            "Gemini-Pro",
            "Llama-2",
            "Mistral-7B"
        ],
        "expert_llms": [
            "GPT-4-Expert",
            "Claude-3-Expert",
            "Gemini-Pro-Expert",
            "Custom-Expert"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)