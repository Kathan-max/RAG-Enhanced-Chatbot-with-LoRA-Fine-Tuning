import os
import uuid
import json
import re
from config.settings import *
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from tqdm import tqdm
import tiktoken

class groupEAlgo:
    def __init__(self, **kwargs):
        self.chunk_overlap = kwargs.get('chunk_overlap', 0)
        self.page_combo = kwargs.get('page_combo', 2)
        self.sentence_combo = kwargs.get('sentence_combo', 4)
        self.encoder = kwargs.get('encoder', None)
        self.nlp = spacy.load("en_core_web_sm")
        self.generation_model = kwargs.get('generation_model', 'gpt-4')
        self.max_chunk_length = self.ideal_chunk_tokens(self.generation_model)
        self.token_encoding_obj = tiktoken.encoding_for_model(self.generation_model)
        
    def ideal_chunk_tokens(self, generation_model):
        total_chunks = TOTAL_CHUNKS_CONSIDERED
        chunks_coverage = TOTAL_CHUNKS_COVERAGE
        max_tokens = MAX_ALLOWED_TOKENS.get(generation_model, None)
        if max_tokens:
            return (max_tokens*(chunks_coverage/100))/total_chunks
        else:
            print("Model not defined in the configurations... Please define the model")
            return 0
        

        
    def read_file(self, path):
        with open(path, 'r', encoding="utf-8") as f:
            data = json.load(f)
        
        return data

    def extract_file_info(self, json_obj):
        file_info = {}
        if "metadata" in json_obj:
            if "file_info" in json_obj['metadata']:
                file_data = json_obj['metadata']['file_info']
                file_info = {
		            "filename": os.path.basename(file_data.get('file_path', '')),
		            "filepath": file_data.get('file_path', ''),
                    "fileext": os.path.basename(file_data.get('file_path', '')).split('.')[1],
                    "filehash": file_data.get('file_hash_sha256', str(uuid.uuid4()))
	            }
            
                if "document_info" in json_obj['metadata']:
                    doc_data = json_obj['metadata']['document_info']
                    file_info['producer'] = doc_data.get('producer', '')
            
                if "technical_info" in json_obj['metadata']:
                    tech_data = json_obj['metadata']['technical_info']
                    file_info['number_of_pages'] = tech_data.get('number_of_pages', 0)

        return file_info

    