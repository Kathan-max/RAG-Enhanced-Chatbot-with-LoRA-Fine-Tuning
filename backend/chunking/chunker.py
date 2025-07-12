from chunking.chunkers.groupEmbeddingAlgo import GroupEAlgo
from config.settings import *
from utils.logger import Logger

class Chunker:
    
    def __init__(self, **kwargs):
        self.logger = Logger(name = "RAGLogger").get_logger()
        self.chunker_str = kwargs['chunker']
        if self.chunker_str == 'BigEmbedding':
            self.chunker = GroupEAlgo(chunk_overlap = kwargs.get('chunk_overlap', DEFAULT_OVER_LAP),
                                      page_combo = kwargs.get('page_combo', PAGE_COMBO),
                                      sentence_combo = kwargs.get('kwargs.get', SENTENCE_COMBO),
                                      encoder = kwargs.get('encoder', None),
                                      generation_model = kwargs.get('generation_model', DEFAULT_GENERATION_MODEL))
            
    
    
    def find_chunks(self, output_json_path):
        return self.chunker.findChunks(output_json_path=output_json_path)
    
    
    def find_chunks_files(self, output_json_paths):
        all_chunks = []
        all_image_objs = {}
        for output_json in output_json_paths:
            chunks, image_objs = self.find_chunks(output_json_path=output_json)
            all_chunks.extend(chunks)
            all_image_objs.update(image_objs)
        return all_chunks, all_image_objs

