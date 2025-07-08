import os
from mistralai import Mistral
import json
import shutil
import base64
import hashlib
import datetime
from typing import Dict, Any, List
from mistralai.models import OCRResponse
from dotenv import load_dotenv
from config.settings import *
from utils.logger import Logger
import fitz
import uuid
from tqdm import tqdm


load_dotenv(dotenv_path='././.env')

class MistralOCR:
    
    def __init__(self, **kwargs):
        self.chunk_length = kwargs.get('chunk_length', DEFAULT_CHUNK_LENGTH) # in terms of tokens
        self.over_lap = kwargs.get('over_lap', DEFAULT_OVER_LAP) 
        self.client = Mistral(api_key=os.getenv('MISTRAL_API_KEY'))
        self.ocr_model = os.getenv('MISTRAL_MODEL')
        self.logger = Logger(name = "RAGLogger").get_logger()
        
    def encodePdf(self, pdf_path):
        try:
            with open(pdf_path, "rb") as pdf_file:
                return base64.b64encode(pdf_file.read()).decode('utf-8')
        except FileNotFoundError:
            self.logger.error(f"Error, file not found at the path: {pdf_path}")
            return None
        except Exception as e:
            self.logger.error(f"Error: {e}")
            return None
    
    def requestOcrModel(self, base64_pdf):
        # base64_pdf = self.encode_pdf(pdf_path)
        if not base64_pdf:
            return {}
        
        pdf_response = self.client.ocr.process(
            model=self.ocr_model,
            document={
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{base64_pdf}"
            },
            include_image_base64=True
        )
        
        return json.loads(pdf_response.model_dump_json())
    
    
    # def save_json(self, json_response, output_json_path):
        
    #     with open(output_json_path, 'w', encoding="utf-8") as f:
    #         json.dump(json_response, f, indent=4)
    
    def processFileChunks(self, base64_dict_list):
        final_mistral_results = []
        document_chunk_annotations = []
        for base64_obj in tqdm(base64_dict_list, total=len(base64_dict_list), desc="Parsing Documents: "):
            ocr_response_json = self.requestOcrModel(base64_obj['base64_str'])
            if not "pages" in ocr_response_json:
                continue
            pages_list = ocr_response_json['pages']
            for idx, page in enumerate(pages_list):
                pages_list[idx]['index'] = idx + base64_obj['start_page']
                pages_list[idx]['file_chunk_id'] = base64_obj['file_chunk_id']
                
                
            final_mistral_results.extend(pages_list)
            if "document_annotation" in ocr_response_json:
                document_chunk_annotations[base64_obj["file_chunk_id"]] = ocr_response_json["document_annotation"]
        return final_mistral_results, document_chunk_annotations
        
    def formatFileMetadata(self, file_metadata):
        return {
            "title": metadata.get("title"),
            "author": metadata.get("author"),
            "subject": metadata.get("subject"),
            "keywords": metadata.get("keywords"),
            "creator": metadata.get("creator"),
            "producer": metadata.get("producer"),
            "creation_date": metadata.get("creationDate"),
            "mod_date": metadata.get("modDate"),
            "page_count": page_count
        }
    
    def splitFiles(self, pdf_path):
        doc = fitz.open(pdf_path)
        metadata = doc.metadata
        
        total_pages = len(doc)
        base64_chunks = []
        pages_per_chunk = PDF_SPLIT_SIZE
        
        
        for start_page in range(0, total_pages, pages_per_chunk):
            end_page = min(start_page+pages_per_chunk - 1, total_pages - 1)
            new_doc = fitz.open()
            new_doc.insert_pdf(doc, from_page = start_page, to_page = end_page)
            
            pdf_bytes = new_doc.write()
            base64_str = base64.b64encode(pdf_bytes).decode('utf-8')
            base64_chunks.append(
                {
                    "file_chunk_id": str(uuid.uuid4()),
                    "start_page":start_page,
                    "end_page": end_page,
                    "base64_str": base64_str,
                }
            )
            new_doc.close()
        doc.close()
        return base64_chunks
            
    
    def formatJsonList(self, file_metadata, final_mistral_results):
        
        pass        
    
    
    def extractInfo(self, pdf_path, save_json = False, output_json_path = ""):
        file_metadata = self.getFileMetadata(pdf_path)
        base64_dict_list = self.splitFiles(pdf_path)
        final_mistral_results, document_chunk_annotations = self.processFileChunks(base64_dict_list)
        final_json_output = self.formatJsonList(file_metadata, final_mistral_results)
        if save_json:
            self.saveJsonFile(output_json_path)
        return final_json_output

# if __name__ == "__main__":
    
#     pdf_path = "F://RAG//backend//data//raw_document_data//mistral7b.pdf"
#     ocr = MistralOCR()
#     json_output = ocr.request_ocr_model(pdf_path)
#     ocr.save_json(json_output, 'F://RAG//backend//data//raw_document_data//output.json')