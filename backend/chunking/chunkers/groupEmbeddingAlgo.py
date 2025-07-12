import os
import uuid
import json
import re
from config.settings import *
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from tqdm import tqdm
import tiktoken
from embedding.encoder import Encoder
from utils.logger import Logger

class GroupEAlgo:
    def __init__(self, **kwargs):
        # Initialize logger first
        self.logger = Logger(name="RAGLogger").get_logger()
        self.logger.info("Initializing GroupEAlgo with parameters: %s", kwargs)
        
        try:
            self.chunk_overlap = kwargs.get('chunk_overlap', 0)
            self.page_combo = kwargs.get('page_combo', 2)
            self.sentence_combo = kwargs.get('sentence_combo', 4)
            
            # Initialize encoder with error handling
            try:
                self.encoder = kwargs.get('encoder', Encoder(encoder_name=DEFAULT_ENCODER_NAME, model_name=DEFAULT_EMBEDDING_MODEL))
                self.logger.info("Encoder initialized successfully")
            except Exception as e:
                self.logger.error("Failed to initialize encoder: %s", str(e))
                raise
            
            # Initialize spaCy model with error handling
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self.logger.info("SpaCy model loaded successfully")
            except Exception as e:
                self.logger.error("Failed to load SpaCy model 'en_core_web_sm': %s", str(e))
                raise
            
            self.generation_model = kwargs.get('generation_model', DEFAULT_GENERATION_MODEL)
            
            # Initialize token encoding with error handling
            try:
                self.token_encoding_obj = tiktoken.encoding_for_model(self.generation_model)
                self.logger.info("Token encoding initialized for model: %s", self.generation_model)
            except Exception as e:
                self.logger.error("Failed to initialize token encoding for model %s: %s", self.generation_model, str(e))
                raise
            
            self.max_chunk_length = self.idealChunkTokens(self.generation_model)
            self.logger.info("GroupEAlgo initialized successfully with max_chunk_length: %d", self.max_chunk_length)
            
        except Exception as e:
            self.logger.error("Failed to initialize GroupEAlgo: %s", str(e))
            raise
        
    def idealChunkTokens(self, generation_model):
        """Calculate ideal chunk token size based on model and configuration."""
        try:
            self.logger.info("Calculating ideal chunk tokens for model: %s", generation_model)
            
            total_chunks = TOTAL_CHUNKS_CONSIDERED
            chunks_coverage = TOTAL_CHUNKS_COVERAGE
            max_tokens = MAX_ALLOWED_TOKENS.get(generation_model, None)
            
            if max_tokens is None:
                self.logger.error("Model %s not defined in MAX_ALLOWED_TOKENS configuration", generation_model)
                raise ValueError(f"Model {generation_model} not defined in the configurations")
            
            if total_chunks == 0:
                self.logger.error("TOTAL_CHUNKS_CONSIDERED is 0, would cause division by zero")
                raise ValueError("TOTAL_CHUNKS_CONSIDERED cannot be 0")
            
            ideal_tokens = (max_tokens * (chunks_coverage / 100)) / total_chunks
            self.logger.info("Calculated ideal chunk tokens: %d", int(ideal_tokens))
            return int(ideal_tokens)
            
        except Exception as e:
            self.logger.error("Error calculating ideal chunk tokens: %s", str(e))
            raise

    def readFile(self, path):
        """Read JSON file with error handling."""
        try:
            self.logger.info("Reading file: %s", path)
            
            if not os.path.exists(path):
                self.logger.error("File does not exist: %s", path)
                raise FileNotFoundError(f"File not found: {path}")
            
            with open(path, 'r', encoding="utf-8") as f:
                data = json.load(f)
            
            self.logger.info("Successfully read file: %s", path)
            return data
            
        except json.JSONDecodeError as e:
            self.logger.error("Invalid JSON in file %s: %s", path, str(e))
            raise
        except Exception as e:
            self.logger.error("Error reading file %s: %s", path, str(e))
            raise

    def replaceImageTag(self, raw_page_data, image_list):
        """Replace image tags in raw page data with structured image descriptions."""
        try:
            self.logger.info("Replacing image tags for %d images", len(image_list))
            
            image_objs = {}
            processed_images = 0
            
            for image in image_list:
                try:
                    img_id = image['id']
                    img_tag = f"![{img_id}]({img_id})"
                    img_uid = str(uuid.uuid4())
                    image_base64 = image['image_base64']
                    # Parse image annotation with error handling
                    try:
                        img_annotation_dict = json.loads(image['image_annotation'])
                    except json.JSONDecodeError as e:
                        self.logger.error("Invalid JSON in image annotation for image %s: %s", img_id, str(e))
                        img_annotation_dict = {}
                    
                    img_desc = f"<image><image_id>{img_uid}</image_id>{img_annotation_dict.get('description', '')}</image>"
                    raw_page_data = raw_page_data.replace(img_tag, img_desc)
                    
                    image_objs[f"<image_id>{img_uid}</image_id>"] = { 
                        "uid": img_uid,
                        "image_description": img_annotation_dict.get('description', ""),
                        "image_type": img_annotation_dict.get('image_type', ""),
                        "image_base64": image_base64,
                    }
                    processed_images += 1
                    
                except Exception as e:
                    self.logger.error("Error processing image %s: %s", image.get('id', 'unknown'), str(e))
                    continue
            
            self.logger.info("Successfully processed %d/%d images", processed_images, len(image_list))
            return raw_page_data, image_objs
            
        except Exception as e:
            self.logger.error("Error in replaceImageTag: %s", str(e))
            raise
    
    def splitText(self, raw_text):
        """Split text into sentences using regex."""
        try:
            if not raw_text or not raw_text.strip():
                self.logger.warning("Empty or whitespace-only text provided for splitting")
                return []
            
            sentences = re.split(r'(?<=[.!?])\s+', raw_text.strip())
            self.logger.info("Split text into %d sentences", len(sentences))
            return sentences
            
        except Exception as e:
            self.logger.error("Error splitting text: %s", str(e))
            raise
    
    def replaceImageDesc(self, raw_text):
        """Replace image descriptions with placeholders for sentence splitting."""
        try:
            image_blocks = re.findall(r'<image>.*?</image>', raw_text, re.DOTALL)
            image_dicts = {}
            
            for i, block in enumerate(image_blocks):
                placeholder = f"__IMAGE_BLOCK__<id>{i}</id>__"
                image_dicts[placeholder] = block
                raw_text = raw_text.replace(block, placeholder)
            
            self.logger.info("Replaced %d image blocks with placeholders", len(image_blocks))
            return raw_text, image_dicts
            
        except Exception as e:
            self.logger.error("Error replacing image descriptions: %s", str(e))
            raise
    
    def putBackImageDesc(self, split_sentences, image_dict):
        """Restore image descriptions from placeholders."""
        try:
            restored_count = 0
            
            for idx, sent in enumerate(split_sentences):
                if "__IMAGE_BLOCK__" in sent:
                    for img_block_ph, img_desc in image_dict.items():
                        if img_block_ph in sent:
                            sent = sent.replace(img_block_ph, img_desc)
                            restored_count += 1
                    split_sentences[idx] = sent
            
            self.logger.info("Restored %d image descriptions", restored_count)
            return split_sentences
            
        except Exception as e:
            self.logger.error("Error restoring image descriptions: %s", str(e))
            raise
    
    def makeSentences(self, raw_page_data):
        """Convert raw page data into sentences, handling image blocks."""
        try:
            self.logger.info("Converting page data to sentences")
            
            img_flag = False
            image_dicts = {}
            
            if "<image>" in raw_page_data:
                img_flag = True
                raw_page_data, image_dicts = self.replaceImageDesc(raw_page_data)

            split_sentences = self.splitText(raw_page_data)
            
            if img_flag:
                split_sentences = self.putBackImageDesc(split_sentences, image_dicts)
            
            self.logger.info("Successfully created %d sentences", len(split_sentences))
            return split_sentences
            
        except Exception as e:
            self.logger.error("Error making sentences: %s", str(e))
            raise
    
    def combineSentList(self, pages):
        """Combine sentence lists from multiple pages."""
        try:
            final_list = []
            for page in pages:
                if isinstance(page, list):
                    final_list.extend(page)
                else:
                    self.logger.warning("Non-list page encountered, skipping")
            
            self.logger.info("Combined %d pages into %d sentences", len(pages), len(final_list))
            return final_list
            
        except Exception as e:
            self.logger.error("Error combining sentence lists: %s", str(e))
            raise
    
    def combineSentences(self, sents_to_consider):
        """Combine sentences with periods."""
        try:
            if not sents_to_consider:
                self.logger.warning("Empty sentence list provided for combination")
                return ""
            
            combined = ".".join(sents_to_consider)
            self.logger.debug("Combined %d sentences into text of length %d", len(sents_to_consider), len(combined))
            return combined
            
        except Exception as e:
            self.logger.error("Error combining sentences: %s", str(e))
            raise
    
    def get_embeddings(self, content):
        """Get embeddings for content with error handling."""
        try:
            if isinstance(content, str):
                if not content.strip():
                    self.logger.warning("Empty string provided for embedding")
                    return None
                embedding = self.encoder.get_embeddings([content])[0]
                self.logger.debug("Generated embedding for single string")
                return embedding
            elif isinstance(content, list):
                if not content:
                    self.logger.warning("Empty list provided for embedding")
                    return []
                embeddings = self.encoder.get_embeddings(content)
                self.logger.debug("Generated embeddings for %d items", len(content))
                return embeddings
            else:
                self.logger.error("Invalid content type for embedding: %s", type(content))
                raise ValueError(f"Invalid content type: {type(content)}")
                
        except Exception as e:
            self.logger.error("Error generating embeddings: %s", str(e))
            raise
    
    def getTokens(self, sent):
        """Get token count for a sentence."""
        try:
            if not sent:
                return 0
            
            tokens = len(self.token_encoding_obj.encode(sent))
            self.logger.debug("Sentence has %d tokens", tokens)
            return tokens
            
        except Exception as e:
            self.logger.error("Error counting tokens: %s", str(e))
            raise
    
    def stayTogether(self, e1, e2, e3, E):
        """Determine if sentences should stay together based on embeddings."""
        try:
            e1_ = e1.reshape(1, -1)
            e2_ = e2.reshape(1, -1)
            e3_ = e3.reshape(1, -1)
            E_ = E.reshape(1, -1)
            
            cos1 = (cosine_similarity(e1_, E_) + cosine_similarity(e2_, E_)) / 2
            cos2 = cosine_similarity(e3_, E_)
            
            should_stay = cos2 > cos1
            self.logger.debug("Cosine similarity check: cos1=%.4f, cos2=%.4f, stay_together=%s", 
                            cos1[0][0], cos2[0][0], should_stay)
            return should_stay
            
        except Exception as e:
            self.logger.error("Error in stayTogether calculation: %s", str(e))
            raise
    
    def getPreviousChunkId(self, chunksList):
        """Get the ID of the previous chunk."""
        try:
            if len(chunksList) > 0:
                prev_id = chunksList[-1]['chunk_id']
                self.logger.debug("Previous chunk ID: %s", prev_id)
                return prev_id
            else:
                self.logger.debug("No previous chunks available")
                return ""
                
        except Exception as e:
            self.logger.error("Error getting previous chunk ID: %s", str(e))
            return ""
    
    def detectEntities(self, chunk):
        """Detect named entities in chunk text."""
        try:
            if not chunk.strip():
                self.logger.warning("Empty chunk provided for entity detection")
                return []
            
            doc = self.nlp(chunk)
            entities = [ent.text for ent in doc.ents if ent.text not in ["\\in", "###", "$", "|", "$ | $"]]
            
            self.logger.debug("Detected %d entities in chunk", len(entities))
            return entities
            
        except Exception as e:
            self.logger.error("Error detecting entities: %s", str(e))
            return []
    
    def getImageList(self, curr_chunk, image_data):
        """Extract image information from current chunk."""
        try:
            pattern = r"<image_id>.*?</image_id>"
            image_lst = []
            matches = re.findall(pattern, curr_chunk, re.DOTALL)
            
            for match in matches:
                try:
                    if match in image_data:
                        image_info = {
                            "image_id": image_data[match]['uid'],
                            "image_url": IMAGE_STORAGE_FOLDER_PATH + image_data[match]['uid'] + ".jpeg",
                            "image_type": image_data[match]['image_type'],
                            "image_description": image_data[match]['image_description'],
                        }
                        image_lst.append(image_info)
                    else:
                        self.logger.warning("Image data not found for match: %s", match)
                except Exception as e:
                    self.logger.error("Error processing image match %s: %s", match, str(e))
                    continue
            
            self.logger.debug("Found %d images in chunk", len(image_lst))
            return image_lst
            
        except Exception as e:
            self.logger.error("Error extracting image list: %s", str(e))
            return []
        
    def checkLast(self, chunks, curr_chunk):
        """Check if current chunk is the same as the last chunk."""
        try:
            if not chunks:
                return False
            
            is_last = chunks[-1]['content'] == curr_chunk
            self.logger.debug("Current chunk is last chunk: %s", is_last)
            return is_last
            
        except Exception as e:
            self.logger.error("Error checking last chunk: %s", str(e))
            return False
    
    def bigEChunks(self, page_sentences, image_data, language, file_info):
        """Main chunking algorithm using semantic embeddings."""
        try:
            self.logger.info("Starting bigEChunks algorithm for %d pages", len(page_sentences))
            
            chunks = []
            prev_sents = []
            chunk_idx = 0
            
            for page_no in range(0, len(page_sentences), self.page_combo):
                try:
                    self.logger.debug("Processing page group starting at %d", page_no)
                    
                    sents_to_consider = self.combineSentList(page_sentences[page_no: page_no + self.page_combo])
                    
                    if not sents_to_consider:
                        self.logger.warning("No sentences found for page group %d", page_no)
                        continue
                    
                    mid_point = int(len(sents_to_consider) / 2)
                    add_point = len(prev_sents)
                    next_prev_sents = sents_to_consider[mid_point:]
                    sents_to_consider = prev_sents + sents_to_consider
                    
                    pages_content = self.combineSentences(sents_to_consider)
                    page_embedding = self.get_embeddings(pages_content)
                    
                    if page_embedding is None:
                        self.logger.error("Failed to generate page embedding for page group %d", page_no)
                        continue
                    
                    prev_sents = next_prev_sents
                    curr_chunk = ""
                    
                    for sent_no in range(0, len(sents_to_consider), self.sentence_combo):
                        try:
                            if curr_chunk == "":
                                curr_chunk = ".".join(sents_to_consider[sent_no: sent_no + self.sentence_combo])
                                e1 = self.get_embeddings(curr_chunk)
                                if e1 is None:
                                    self.logger.error("Failed to generate embedding for initial chunk")
                                    break
                            else:
                                next_chunk = ".".join(sents_to_consider[sent_no: sent_no + self.sentence_combo])
                                e2 = self.get_embeddings(next_chunk)
                                
                                if e2 is None:
                                    self.logger.error("Failed to generate embedding for next chunk")
                                    break
                                
                                combo_chunk = ".".join([curr_chunk, next_chunk])
                                e3 = self.get_embeddings(combo_chunk)
                                
                                if e3 is None:
                                    self.logger.error("Failed to generate embedding for combo chunk")
                                    break
                                
                                if (self.stayTogether(e1=e1, e2=e2, e3=e3, E=page_embedding) and 
                                    self.getTokens(combo_chunk) <= self.max_chunk_length):
                                    e1 = e3
                                    curr_chunk = combo_chunk
                                    self.logger.debug("Merged chunks at position %d", sent_no)
                                else:
                                    # Create chunk
                                    chunk_id = str(uuid.uuid4())
                                    chunk_data = {
                                        "chunk_id": chunk_id,
                                        "file_info": file_info,
                                        "chunk_info": {
                                            "chunk_index": chunk_idx,
                                            "encoder": self.encoder.encoder_name,
                                            "language": language,
                                            "chunk_size": self.getTokens(curr_chunk),
                                            "chunk_type": TEXT_CHUNK_TYPE,
                                            "prev_chunk_id": self.getPreviousChunkId(chunks),
                                            "next_chunk_id": ""
                                        },
                                        "content": curr_chunk,
                                        "embedding": str(self.get_embeddings(curr_chunk).tolist()),
                                        "semantic_info": {
                                            "keywords": self.detectEntities(curr_chunk)
                                        },
                                        "position_info": {},
                                        "media_ref": {
                                            "images": self.getImageList(curr_chunk, image_data),
                                            "tables": [],
                                            "links": []
                                        },
                                        "created_at": "",
                                        "updated_at": ""
                                    }
                                    chunks.append(chunk_data)
                                    
                                    # Update previous chunk's next_chunk_id
                                    if len(chunks) > 1:
                                        chunks[-2]["chunk_info"]['next_chunk_id'] = chunk_id
                                    
                                    self.logger.debug("Created chunk %d with ID %s", chunk_idx, chunk_id)
                                    
                                    e1 = e2
                                    curr_chunk = next_chunk
                                    chunk_idx += 1
                            
                            if (sent_no + 1) >= (mid_point + add_point):
                                break
                                
                        except Exception as e:
                            self.logger.error("Error processing sentence group %d: %s", sent_no, str(e))
                            continue
                    
                    # Handle remaining chunk
                    if curr_chunk and not self.checkLast(chunks, curr_chunk):
                        try:
                            chunk_id = str(uuid.uuid4())
                            chunk_data = {
                                "chunk_id": chunk_id,
                                "file_info": file_info,
                                "chunk_info": {
                                    "chunk_index": chunk_idx,
                                    "encoder": self.encoder.encoder_name,
                                    "language": language,
                                    "chunk_size": self.getTokens(curr_chunk),
                                    "chunk_type": TEXT_CHUNK_TYPE,
                                    "prev_chunk_id": self.getPreviousChunkId(chunks),
                                    "next_chunk_id": ""
                                },
                                "content": curr_chunk,
                                "embedding": str(self.get_embeddings(curr_chunk).tolist()),
                                "semantic_info": {
                                    "keywords": self.detectEntities(curr_chunk)
                                },
                                "position_info": {},
                                "media_ref": {
                                    "images": self.getImageList(curr_chunk, image_data),
                                    "tables": [],
                                    "links": []
                                },
                                "created_at": "",
                                "updated_at": ""
                            }
                            chunks.append(chunk_data)
                            
                            if len(chunks) > 1:
                                chunks[-2]["chunk_info"]['next_chunk_id'] = chunk_id
                            
                            self.logger.debug("Created final chunk %d with ID %s", chunk_idx, chunk_id)
                            
                        except Exception as e:
                            self.logger.error("Error creating final chunk: %s", str(e))
                    
                except Exception as e:
                    self.logger.error("Error processing page group %d: %s", page_no, str(e))
                    continue
            
            self.logger.info("Successfully created %d chunks", len(chunks))
            return chunks
            
        except Exception as e:
            self.logger.error("Error in bigEChunks: %s", str(e))
            raise
            
    def getChunks(self, ocr_results, document_level_info, file_info):
        """Process OCR results and create chunks."""
        try:
            self.logger.info("Processing OCR results for %d pages", len(ocr_results))
            
            image_objs = {}
            all_sentences = []
            language = None  # Initialize language variable
            
            for page_idx, page in enumerate(ocr_results):
                try:
                    self.logger.debug("Processing page %d", page_idx)
                    
                    page_raw_data = page.get('markdown', '')
                    image_data = page.get('images', [])
                    file_id = page.get('file_chunk_id', '')
                    
                    # Get language from document level info
                    if file_id in document_level_info:
                        language = document_level_info[file_id].get("language", "en")
                    else:
                        self.logger.warning("File ID %s not found in document_level_info", file_id)
                        language = "en"  # Default language
                    
                    if image_data:
                        page_raw_data, image_objs_ = self.replaceImageTag(page_raw_data, image_data)
                        image_objs.update(image_objs_)
                    
                    page_sentences = self.makeSentences(page_raw_data)
                    all_sentences.append(page_sentences)
                    
                except Exception as e:
                    self.logger.error("Error processing page %d: %s", page_idx, str(e))
                    continue
            
            if not all_sentences:
                self.logger.warning("No sentences extracted from OCR results")
                return [], {}
            
            # Use the last valid language found, or default to "en"
            if language is None:
                language = "en"
                self.logger.warning("No language found in document_level_info, using default: %s", language)
            
            all_chunks = self.bigEChunks(
                page_sentences=all_sentences, 
                image_data=image_objs, 
                language=language, 
                file_info=file_info
            )
            
            self.logger.info("Successfully processed OCR results: %d chunks created", len(all_chunks))
            return all_chunks, image_objs
            
        except Exception as e:
            self.logger.error("Error in getChunks: %s", str(e))
            raise

    def findChunks(self, output_json_path):
        """Main method to find chunks from OCR results."""
        try:
            self.logger.info("Starting chunk finding process for file: %s", output_json_path)
            
            json_data = self.readFile(path=output_json_path)
            
            chunk_file_metadata = json_data.get('file_metadata', {})
            ocr_result = json_data.get('ocr_results', [])
            document_level_info = json_data.get('document_annotations', {})
            
            self.logger.info("Loaded JSON data: %d OCR results, %d document annotations", 
                           len(ocr_result), len(document_level_info))
            
            chunks = []
            image_objs = {}
            
            if ocr_result:
                chunks, image_objs = self.getChunks(ocr_result, document_level_info, chunk_file_metadata)
            else:
                self.logger.warning("No OCR results found in input file")
            
            self.logger.info("Chunk finding process completed: %d chunks, %d images", 
                           len(chunks), len(image_objs))
            
            return chunks, image_objs
            
        except Exception as e:
            self.logger.error("Error in findChunks: %s", str(e))
            raise