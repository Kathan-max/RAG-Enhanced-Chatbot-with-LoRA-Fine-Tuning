DEFAULT_CHUNK_LENGTH = 1000 # in terms of tokens
DEFAULT_OVER_LAP = 50 # in terms of tokens
PDF_SPLIT_SIZE = 5 # number of pages to be considered in one PDF split
PAGE_COMBO = 2
SENTENCE_COMBO = 4
DEFAULT_GENERATION_MODEL = 'gpt-4'
JSON_OUTPUT_DIR = 'data/raw_jsons/'

MAX_ALLOWED_TOKENS = {
    'gpt-3.5-turbo': 4096,
    'gpt-3.5-turbo-16k': 16384,
    'gpt-4': 8192,
    'gpt-4-32k': 32768,
    'gpt-4-turbo': 128000,
    'gpt-4o': 128000,
    'Claude 2': 10000,
    'Claude Instant 2': 10000
}

DEFAULT_ENCODER_NAME = 'sentence_encoder'
DEFAULT_EMBEDDING_MODEL = 'all-mpnet-base-v2'
SENTENCE_ENCODER = 'sentence_encoder'
TEXT_CHUNK_TYPE = 'sentence_chunk'
IMAGE_STORAGE_FOLDER_PATH = 'images/document_images/'
TOTAL_CHUNKS_CONSIDERED = 5
TOTAL_CHUNKS_COVERAGE = 50 # in terms of percentage
IMAGES_URL = 'document_images'
UPLOAD_BATCH = 50
TEMP_FILES_DOC = "F://RAG//backend//data//raw_document_data//"
TEMP_FILES_JSONS = "F://RAG//backend//data//raw_jsons//"
# MIN_TABLES_COUNT = 1
# MIN_IMAGES_COUNT = 1
# UNREAD_DATA_DIR_PATH = 'data/raw_pdfs/unRead'
# JSON_OUTPUT_DIR = 'data/raw_jsons'
# JSON_SPLIT_DIR_JSON = 'data/temp_split/jsons/'
# JSON_SPLIT_DIR_PDF = 'data/temp_split/pdf/'
# MAX_MISTRAL_PAGES = 8
# MAX_OVERLAP_PAGES = 0
# CHUNKING_LOGIC = 'bigE'
# MAX_CHUNK_LENGTH = 1000 # number of tokens
# CHUNK_OVERLAP = 0
# SENTENCE_COMBO = 4
# PAGE_COMBO = 2
# # MISTRAL_OCR_IMAGE_PATTERN = r'!\[(img-\d+\.jpeg)\]\(img-\d+\.jpeg\)'
# MISTRAL_OCR_IMAGE_PATTERN = r'!\[img-\d+\.jpeg\]\(img-\d+\.jpeg\)'
# PARSER_PATTERN_DICT = {
#     'MistralOCR': {
#         'EXACT_MATCH_PATTERN': r'!\[img-\d+\.jpeg\]\(img-\d+\.jpeg\)', # pulls out the entire tag from the message, ex: ![img-1.jpeg](img-1.jpeg)
#         'IMAGE_ID_PATTERN': r'!\[(img-\d+\.jpeg)\]\(img-\d+\.jpeg\)' # pull out the actual image tag from the image pattern it self, ex: img-1.jpeg
#     }
# }
# SENTENCE_ENCODER = 'sentenceEncoder'
# SENTENCE_ENCODER_MODEL = 'all-mpnet-base-v2'
# GENERATION_MODEL = 'gpt-4' # final model which will generate the output for the user.
# TOTAL_CHUNKS_COVERAGE = 50
# TOTAL_CHUNKS_CONSIDERED = 5

# IMAGES_URL = 'document_images'
# UPLOAD_BATCH = 50
# TOP_K = 7
# SIMILARITY_THRESHOLD = 0.4
# TEMPERATURE = 0.2
# # MISTRAL_OCR_EXACT_MATCH = r'!\[img-\d+\.jpeg\]\(img-\d+\.jpeg\)'