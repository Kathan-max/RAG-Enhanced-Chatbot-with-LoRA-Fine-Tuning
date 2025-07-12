import os
from parsing.parser import Parser
from chunking.chunker import Chunker
from config.settings import *
from database.vector.vectorDB import VectorDB
import threading
from utils.logger import Logger


class Ingestaion:
    
    def __init__(self, **kwargs):
        self.parser = Parser(parser = kwargs['parser'],
                             chunk_length = kwargs.get('chunk_length', DEFAULT_CHUNK_LENGTH),
                             over_lap = kwargs.get('over_lap', DEFAULT_OVER_LAP))
        
        self.chunker = Chunker(chunker = kwargs['chunker'],
                               chunk_overlap = kwargs.get('chunk_overlap', DEFAULT_OVER_LAP),
                               page_combo = kwargs.get('page_combo', PAGE_COMBO),
                               sentence_combo = kwargs.get('kwargs.get', SENTENCE_COMBO),
                               encoder = kwargs.get('encoder', None),
                               generation_model = kwargs.get('generation_model', DEFAULT_GENERATION_MODEL))
        
        self.logger = Logger(name="RAGLogger").get_logger()

        self.vectordb = VectorDB()
    # Part-1
        # parsing    
    def parse_document(self, pdf_path, save_json = False, output_json_path = ""):
        self.parser.extractInfo(pdf_path=pdf_path, 
                                save_json=save_json, 
                                output_json_path=output_json_path)
    
    def parse_docs(self, pdf_paths, save_json = False):
        output_json_paths = []
        for pdf_path in pdf_paths:
            if JSON_OUTPUT_DIR.endswith('/'):
                output_json_path = JSON_OUTPUT_DIR + os.path.basename(pdf_path).split('.')[0] + '.json'
            else:
                output_json_path = JSON_OUTPUT_DIR + '/' + os.path.basename(pdf_path).split('.')[0] + '.json'
            output_json_paths.append(output_json_path)
            self.parse_document(pdf_path=pdf_path, 
                                save_json=save_json, 
                                output_json_path=output_json_path)
        return output_json_paths
        
    # Part-2
        # chunking
    def find_chunks(self, output_json_paths):
        return self.chunker.find_chunks_files(output_json_paths=output_json_paths)
    
    # Part-3
        # ingestion to the vector database
    def ingest_chunks_imgs(self, chunks, image_objs):
        self.logger.info(f"Ingesting the Chunks in Vector DB...")
        self.vectordb.upload_chunks(chunks=chunks)
        self.logger.info(f"Ingesting the Images in Vector DB...")
        self.vectordb.upload_images(image_objs=image_objs)
    
    # Part-4
        # Training the Query optimizer model (if training is On)
            # Generate raw user queries and optimized queries based on document annotations
            # Train the QWEN / LLaMa model 
    def train_query_optimizer(self, chunks):
        pass
    
    def ingest_files(self, file_path_lists, save_json, train_query_opt = False):
        output_json_paths = self.parse_docs(pdf_paths=file_path_lists, save_json=save_json)
        # output_json_paths = ["F://RAG//backend//data//raw_jsons//1706.json", "F://RAG//backend//data//raw_jsons//mistral7b.json"]
        # output_json_paths = ["F://RAG//backend//data//raw_jsons//1706.json"]
        chunks, image_objs = self.find_chunks(output_json_paths=output_json_paths)
        if train_query_opt:
            t1 = threading.Thread(target=self.ingest_chunks_imgs(chunks=chunks, image_objs=image_objs))
            t2 = threading.Thread(target=self.train_query_optimizer(chunks=chunks))
            t1.start()
            t2.start()
            t1.join() # only wait for the data to be ingested. no need to wait for the query optimizer to be trained.
        else:
            self.ingest_chunks_imgs(chunks=chunks, image_objs=image_objs)