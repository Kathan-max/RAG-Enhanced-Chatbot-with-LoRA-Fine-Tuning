from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
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
    activeLLMs: List[str] = []
    expertLLM: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    images: List[str] = []

# Dummy vector database storage (in production, use a real vector database)
# vector_db = {
#     "total_vectors": 0,
#     "total_documents": 0,
#     "documents": []
# }
load_dotenv(".env")

ingestion_obj = Ingestaion(parser = os.getenv("PARSER"),
                 chunker = os.getenv("CHUNKER"),
                 encoder = Encoder(encoder_name = os.getenv("ENCODER_NAME"),
                                   model_name = os.getenv("ENCODING_MODEL")))

@app.get("/")
async def root():
    return {"message": "LLM Chat Application API is running"}

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Handle chat requests with support for MultiLLM configuration
    """
    try:
        # Simulate processing time
        await asyncio.sleep(random.uniform(1, 3))
        
        # Generate dummy response based on configuration
        if request.multiLLM:
            response = f"[MultiLLM Response] Processing your query: '{request.query}' with {len(request.activeLLMs)} active LLMs"
            if request.expertLLM:
                response += f" and expert model: {request.expertLLM}"
            response += f"\n\nActive LLMs: {', '.join(request.activeLLMs)}"
            response += "\n\nThis is a comprehensive response utilizing multiple AI models to provide you with the best possible answer."
        else:
            response = f"[Standard Response] Thank you for your query: '{request.query}'. This is a standard single-model response."
        
        # Simulate image generation for some queries
        images = []
        if "image" in request.query.lower() or "picture" in request.query.lower() or "draw" in request.query.lower():
            # Generate dummy base64 image data (1x1 pixel PNG)
            dummy_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
            images.append(dummy_image)
            response += "\n\nHere's your generated image: <img-0>"
        
        return ChatResponse(response=response, images=images)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")

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