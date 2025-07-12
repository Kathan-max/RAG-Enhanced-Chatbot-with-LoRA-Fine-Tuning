from database.vector.vectorDBs.supabasevdb import SupabaseChunkVectorDB
from utils.logger import Logger

class VectorDB:
    
    def __init__(self, **kwargs):
        self.supabase = SupabaseChunkVectorDB()
    
    def upload_images(self, image_objs):
        return self.supabase.upload_images(image_objs)
    
    def upload_chunks(self, chunks):
        self.supabase.upload_chunks(chunks_list=chunks)