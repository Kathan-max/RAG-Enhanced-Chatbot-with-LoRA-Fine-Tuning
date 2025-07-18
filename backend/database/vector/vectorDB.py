from database.vector.vectorDBs.supabasevdb import SupabaseChunkVectorDB
from utils.logger import Logger
import re

class VectorDB:
    
    def __init__(self, **kwargs):
        self.supabase = SupabaseChunkVectorDB()
    
    def upload_images(self, image_objs):
        return self.supabase.upload_images(image_objs)
    
    def upload_chunks(self, chunks):
        self.supabase.upload_chunks(chunks_list=chunks)
    
    def retrieve_images(self, image_urls):
        return self.supabase.retrieve_images(image_urls=image_urls)
        
    def retrieve_chunks(self, query, top_k = 5, 
                        similarity_threshold = 0.7, 
                        file_filter = {},
                        chunk_filter = {},
                        langauge_filter = None,
                        chunk_type = None,
                        encoder = None):
        retrieved_chunks = self.supabase.retrieve_top_k(
            query=query,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            file_filter=file_filter,
            chunk_filter=chunk_filter,
            langauge_filter=langauge_filter,
            chunk_type=chunk_type,
            encoder=encoder
        )
        # # retrieving the images to send to the frontend
        # image_urls = set()
        # for obj in retrieved_chunks:
        #     if obj['media_ref']['images'] != []:
        #         for image_obj in obj['media_ref']['images']:
        #               image_urls.add(image_obj['image_url'])
        
        # image_base64_dict = {}
        # if len(image_urls) > 0:
        #     image_base64_dict = self.supabase.retrieve_images(image_urls=image_urls)
        
        # return retrieved_chunks, image_base64_dict
        return retrieved_chunks
    
    def getChunkByID(self, chunk_id):
        return self.supabase.getChunkByID(chunk_id=chunk_id)

    def getChunksByID(self, chunk_ids):
        return self.supabase.getChunksByIds(chunk_ids = chunk_ids)