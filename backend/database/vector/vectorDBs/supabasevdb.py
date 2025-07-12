import os
from dotenv import load_dotenv
from config.settings import *
from supabase import create_client, Client
import uuid
import base64
import json
import numpy as np
from typing import List, Dict, Any, Optional, Union
from utils.logger import Logger

load_dotenv(dotenv_path='./././.env')

class SupabaseChunkVectorDB:
    
    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_secret_key = os.getenv("SUPABASE_SECRET_KEY")
        self.image_bucket = os.getenv("SUPABASE_IMAGE_BUCKET")
        self.upload_path = IMAGES_URL
        self.supabase_client = create_client(self.supabase_url, self.supabase_secret_key)
        self.logger = Logger(name="RAGLogger").get_logger()
        
    
    def get_image_extension(self, image_base64):
        first_part = image_base64.split('/')[1].split(';')[0]
        return first_part.strip()
        
    def upload_images(self, image_objs):
        image_dict = {}
        for img_obj in image_objs.values():
            # print("Image_obj: ", img_obj)
            image_id = img_obj['uid']
            image_base64_str = img_obj['image_base64'].split(',')[1]
            file_data = base64.b64decode(image_base64_str)
            image_ext = self.get_image_extension(img_obj['image_base64'].split(',')[0])
            response = self.supabase_client.storage.from_(self.image_bucket).upload(self.upload_path + '/' + image_id + '.' + image_ext, file_data, {
                "content-type": f"image/{image_ext}"
            })
            image_url = response.fullPath
            image_dict[image_id] = {
                'image_hash': image_id,
                'image_url': image_url
            }
        
        return image_dict
    
    def upload_chunk_batch(self, records_to_insert):
        try:
            result = self.supabase_client.table("chunks").insert(records_to_insert).execute()
            self.logger.info(f"Successfully uploaded {len(records_to_insert)} chunks")
            return result.data
        except Exception as e:
            self.logger.info(f"Error uploading chunks batch: {e}")
            raise
    
    def upload_chunks(self, chunks_list):
        records_to_insert = []
        batch_size = UPLOAD_BATCH
        for chunk in chunks_list:
            record = {
                "chunk_id": chunk['chunk_id'],
                "file_info": chunk['file_info'],
                "content": chunk['content'],
                "embedding": chunk['embedding'],
                "chunk_info": chunk['chunk_info'],
                "position_info": chunk["position_info"],
                "media_ref": chunk['media_ref'],
                "semantic_info": chunk["semantic_info"]
            }
            records_to_insert.append(record)
            if len(records_to_insert) >= batch_size:
                result = self.upload_chunk_batch(records_to_insert=records_to_insert)
                records_to_insert = []
        
        if len(records_to_insert) != 0:
            result = self.upload_chunk_batch(records_to_insert=records_to_insert)
            records_to_insert = []
    
    def format_embedding(self, embed):
        return str(embed.tolist())
    
    def retrieve_top_k(self, query, top_k = 5, 
                       similarity_threshold = 0.7, 
                       file_filter = {},
                       chunk_filter = {},
                       langauge_filter = None,
                       chunk_type = None,
                       encoder = None):
        pass