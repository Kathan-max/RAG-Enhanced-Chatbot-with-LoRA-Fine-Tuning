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
from llmservice.llmmodels import ProcessingMode
from utils.logger import Logger

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
retriever = Retriever()
logger = Logger(name="RAGLogger").get_logger()

@app.get("/")
async def root():
    return {"message": "LLM Chat Application API is running"}

def filter_image_dict(image_dict, image_ids_lst):
    final_image_dict = {image_id: image_dict[image_id] for image_id in image_ids_lst}
    return final_image_dict

def getContextText(retieved_chunks):
    return "\n\n".join([f"Context {i+1}: {chunk['content']}" for i, chunk in enumerate(retieved_chunks)])

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
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
    context_text = getContextText(retieved_chunks)
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
    
    print("LLM response: ", llmresponse)
    answer = llmresponse.get('answer', "Was not able to fetch correct output.")
    reasoning = llmresponse.get('reasoning', 'No output...')
    final_image_dict = filter_image_dict(image_dict=image_dicts, image_ids_lst=llmresponse.get('relevant_image_tags', []))
    
    return ChatResponse(response=answer, reasoning=reasoning, images=final_image_dict)
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
    Get vector database analytics
    """
    try:
        pass
        # return {
        #     "total_vectors": vector_db["total_vectors"],
        #     "total_documents": vector_db["total_documents"],
        #     "documents": vector_db["documents"]
        # }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching analytics: {str(e)}")

@app.get("/api/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy", "message": "API is running properly"}

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