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
from mistralai.extra import response_format_from_pydantic_model
from enum import Enum
from pydantic import BaseModel, Field


load_dotenv(dotenv_path='././.env')

class ImageType(str, Enum):
    GRAPH = "graph"
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"

class Image(BaseModel):
    image_type: ImageType = Field(..., description="The type of the image. Must be one of 'graph', 'text', 'table' or 'image'.")
    description: str = Field(..., description="A description of the image.")
    
class Document(BaseModel):
    language: str = Field(..., description="The language of the document in ISO 639-1 code format (e.g., 'en', 'fr').")
    summary: str = Field(..., description="A summary of the document.")
    authors: list[str] = Field(..., description="A list of authors who contributed to the document.")


class MistralOCR:

    def __init__(self, **kwargs):
        self.client = Mistral(api_key=os.getenv('MISTRAL_API_KEY'))
        self.ocr_model = os.getenv('MISTRAL_MODEL')
        self.logger = Logger(name="RAGLogger").get_logger()

    def requestOcrModel(self, base64_pdf):
        if not base64_pdf:
            self.logger.warning("Empty Base64 PDF string received for OCR.")
            return {}

        self.logger.info("Sending request to Mistral OCR API...")
        pdf_response = self.client.ocr.process(
            model=self.ocr_model,
            document={
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{base64_pdf}"
            },
            bbox_annotation_format=response_format_from_pydantic_model(Image),
            document_annotation_format=response_format_from_pydantic_model(Document),
            include_image_base64=True,
        )

        self.logger.info("Received response from Mistral OCR API.")
        return json.loads(pdf_response.model_dump_json())

    def processFileChunks(self, base64_dict_list):
        final_mistral_results = []
        document_chunk_annotations = {}

        self.logger.info(f"Processing {len(base64_dict_list)} file chunks...")

        for base64_obj in tqdm(base64_dict_list, total=len(base64_dict_list), desc="Parsing Documents: "):
            self.logger.debug(f"Processing chunk from page {base64_obj['start_page']} to {base64_obj['end_page']}...")
            ocr_response_json = self.requestOcrModel(base64_obj['base64_str'])

            if "pages" not in ocr_response_json:
                self.logger.warning(f"No pages found in OCR response for chunk: {base64_obj['file_chunk_id']}")
                continue

            pages_list = ocr_response_json['pages']
            for idx, page in enumerate(pages_list):
                pages_list[idx]['index'] = idx + base64_obj['start_page']
                pages_list[idx]['file_chunk_id'] = base64_obj['file_chunk_id']

            final_mistral_results.extend(pages_list)

            if "document_annotation" in ocr_response_json:
                document_chunk_annotations[base64_obj["file_chunk_id"]] = json.loads(ocr_response_json["document_annotation"])

        self.logger.info("Finished processing all file chunks.")
        return final_mistral_results, document_chunk_annotations

    def formatFileMetadata(self, metadata, filename, extension):
        self.logger.info("Formatting file-level metadata...")
        return {
            "title": filename,
            "extension": extension,
            "author": metadata.get("author"),
            "subject": metadata.get("subject"),
            "keywords": metadata.get("keywords"),
            "creator": metadata.get("creator"),
            "producer": metadata.get("producer"),
        }

    def splitFiles(self, pdf_path):
        self.logger.info(f"Splitting PDF: {pdf_path}")
        doc = fitz.open(pdf_path)
        filename = os.path.basename(pdf_path).split(".")[0]
        extension = os.path.basename(pdf_path).split(".")[1]
        metadata = doc.metadata
        file_metadata = self.formatFileMetadata(metadata, filename, extension)
        total_pages = len(doc)
        base64_chunks = []
        pages_per_chunk = PDF_SPLIT_SIZE

        self.logger.info(f"Total pages in PDF: {total_pages}. Splitting into chunks of {pages_per_chunk} pages...")

        for start_page in range(0, total_pages, pages_per_chunk):
            end_page = min(start_page + pages_per_chunk - 1, total_pages - 1)
            self.logger.debug(f"Creating chunk: pages {start_page} to {end_page}")
            new_doc = fitz.open()
            new_doc.insert_pdf(doc, from_page=start_page, to_page=end_page)

            pdf_bytes = new_doc.write()
            base64_str = base64.b64encode(pdf_bytes).decode('utf-8')
            base64_chunks.append(
                {
                    "file_chunk_id": str(uuid.uuid4()),
                    "start_page": start_page,
                    "end_page": end_page,
                    "base64_str": base64_str,
                }
            )
            new_doc.close()

        doc.close()
        self.logger.info("Finished splitting PDF into base64 chunks.")
        return file_metadata, base64_chunks

    def formatJsonList(self, file_metadata, final_mistral_results, document_chunk_annotations):
        self.logger.info("Formatting final JSON output...")
        return {
            "file_metadata": file_metadata,
            "ocr_results": final_mistral_results,
            "document_annotations": document_chunk_annotations
        }

    def saveJsonFile(self, data, output_path, indent=4):
        self.logger.info(f"Saving JSON to: {output_path}")
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=indent)
        self.logger.info(f"Saved the file successfully: {output_path}")

    def extractInfo(self, pdf_path, save_json=False, output_json_path=""):
        self.logger.info(f"Starting document parsing for: {pdf_path}")
        file_metadata, base64_dict_list = self.splitFiles(pdf_path)

        self.logger.info("Sending file chunks to OCR model...")
        final_mistral_results, document_chunk_annotations = self.processFileChunks(base64_dict_list)

        final_json_output = self.formatJsonList(file_metadata, final_mistral_results, document_chunk_annotations)

        if save_json and output_json_path != "":
            self.logger.info("Saving OCR result to JSON...")
            self.saveJsonFile(final_json_output, output_json_path)

        self.logger.info("Document parsing complete.")
        return final_json_output
