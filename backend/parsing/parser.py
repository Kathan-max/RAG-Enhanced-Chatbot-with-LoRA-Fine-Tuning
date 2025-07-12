from parsing.parsers.mistral_ocr import MistralOCR
from config.settings import *
from utils.logger import Logger

class Parser:
    
    def __init__(self, **kwargs):
        self.parser_str = kwargs['parser']
        if self.parser_str == 'MISTRAL':
            self.parser = MistralOCR(chunk_length = kwargs.get('chunk_length', DEFAULT_CHUNK_LENGTH),
                                     over_lap = kwargs.get('over_lap', DEFAULT_OVER_LAP))
        self.logger = Logger(name="RAGLogger").get_logger()
        
    
    def extractInfo(self, pdf_path, save_json=False, output_json_path=""):
        self.parser.extractInfo(pdf_path=pdf_path, 
                                       save_json=save_json, 
                                       output_json_path=output_json_path)