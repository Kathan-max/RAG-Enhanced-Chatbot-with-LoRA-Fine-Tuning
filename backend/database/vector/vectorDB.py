from database.vector.vectorDBs.supabasevdb import SupabaseChunkVectorDB
from utils.logger import Logger
import re

class VectorDB:
    
    def __init__(self, **kwargs):
        self.supabase = SupabaseChunkVectorDB()
        self.logger = Logger(name="RAGLogger").get_logger()
    
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

    def get_analytics_data(self):
        """
        Get comprehensive analytics data for the vector database
        """
        try:
            # Get basic vector DB statistics
            vector_stats = self.supabase.get_vector_db_statistics()
            
            # Get search performance stats
            search_stats = self.supabase.get_search_performance_stats()
            
            # Combine all analytics data
            analytics_data = {
                'vector_database': vector_stats,
                'search_performance': search_stats,
                'system_info': {
                    'database_type': 'Supabase Vector DB',
                    'connection_status': 'connected' if self.supabase.supabase_client else 'disconnected',
                    'api_version': '1.0',
                }
            }
            
            self.logger.info("Retrieved comprehensive analytics data")
            return analytics_data
            
        except Exception as e:
            self.logger.error(f"Error retrieving analytics data: {e}")
            return {
                'vector_database': {
                    'total_vectors': 0,
                    'total_documents': 0,
                    'embedding_dimensions': 0,
                    'similarity_algorithm': 'Unknown',
                    'database_status': 'error',
                    'error': str(e)
                },
                'search_performance': {
                    'average_search_time_ms': 0,
                    'total_searches': 0,
                    'error': str(e)
                },
                'system_info': {
                    'database_type': 'Supabase Vector DB',
                    'connection_status': 'error',
                    'api_version': '1.0',
                    'error': str(e)
                }
            }
    
    def get_simple_stats(self):
        """
        Get simplified statistics for quick display
        """
        try:
            stats = self.supabase.get_vector_db_statistics()
            return {
                'total_vectors': stats.get('total_vectors', 0),
                'total_documents': stats.get('total_documents', 0),
                'embedding_dimensions': stats.get('embedding_dimensions', 0),
                'similarity_algorithm': stats.get('similarity_algorithm', 'Cosine Similarity'),
                'database_status': stats.get('database_status', 'unknown'),
                'storage_size_mb': stats.get('estimated_storage_mb', 0.0)
            }
        except Exception as e:
            self.logger.error(f"Error retrieving simple stats: {e}")
            return {
                'total_vectors': 0,
                'total_documents': 0,
                'embedding_dimensions': 0,
                'similarity_algorithm': 'Unknown',
                'database_status': 'error',
                'storage_size_mb': 0.0,
                'error': str(e)
            }