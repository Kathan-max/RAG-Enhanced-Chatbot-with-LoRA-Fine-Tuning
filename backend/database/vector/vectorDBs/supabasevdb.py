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
        if encoder:
            query_embedding = self.format_embedding(encoder.get_embeddings(sentences = [query])[0])
            try:
                result = self.supabase_client.rpc(
                    "similarity_search_chunks",
                    {
                        "query_embedding": query_embedding,
                        "match_threahold": similarity_threshold,
                        "match_count": top_k,
                        "file_filter": None,
                        "chunk_filter": None, 
                        "language_filter": langauge_filter,
                        "chunk_type_filter": chunk_type
                    }
                ).execute()
                return result.data
            
            except Exception as e:
                self.logger.error(f"Error performing the similarity search: {e}")
                raise
            
    
    def retrieve_images(self, image_urls):
        bucket_name = IMAGE_STORAGE_FOLDER_PATH.split('/')[0]
        folder_name = IMAGE_STORAGE_FOLDER_PATH.split('/')[1] 
        image_base64s = {}
        for image_url in image_urls:
            image_basename = os.path.basename(image_url)
            supabase_path = f"{folder_name}/{image_basename}"
            extension = image_basename.split('.')[1]
            image_bytes = self.supabase_client.storage.from_(bucket_name).download(supabase_path)
            image_base64s[image_basename.replace('.'+extension, '')] = base64.b64encode(image_bytes).decode('utf-8')
        
        return image_base64s
    
    def getChunkByID(self, chunk_id):
        try:
            result = self.supabase_client.rpc(
                "get_chunk_by_id",
                {
                    "input_chunk_id": str(chunk_id)
                }
            ).execute()
            return result.data
        except Exception as e:
            self.logger.error(f"Error occurred when fetching the chunk of id: {chunk_id}, error: {e}")
            raise
    
    def getChunksByIds(self, chunk_ids):
        try:
            result = self.supabase_client.rpc(
                "get_chunks_by_ids",
                {
                    "input_chunks_ids": chunk_ids
                }
            ).execute()
            return result.data
        except Exception as e:
            self.logger.error(f"Error occurred when fetching the chunk of ids: {str(chunk_ids)}, error: {e}")
            raise
    
    def get_vector_db_statistics(self):
        """
        Get comprehensive vector database statistics
        """
        try:
            stats = {}
            
            # Get total number of chunks/vectors
            total_chunks_result = self.supabase_client.table("chunks").select("chunk_id", count="exact").execute()
            stats['total_vectors'] = total_chunks_result.count if total_chunks_result.count is not None else 0
            
            # Get unique files count
            unique_files_result = self.supabase_client.rpc("get_unique_files_count").execute()
            stats['total_documents'] = unique_files_result.data[0]['count'] if unique_files_result.data else 0
            
            # Get embedding dimensions (assuming all embeddings have same dimension)
            if stats['total_vectors'] > 0:
                sample_embedding_result = self.supabase_client.table("chunks").select("embedding").limit(1).execute()
                if sample_embedding_result.data:
                    sample_embedding = sample_embedding_result.data[0]['embedding']
                    if isinstance(sample_embedding, str):
                        # Parse the string representation of the list
                        import ast
                        embedding_list = ast.literal_eval(sample_embedding)
                        stats['embedding_dimensions'] = len(embedding_list)
                    elif isinstance(sample_embedding, list):
                        stats['embedding_dimensions'] = len(sample_embedding)
                    else:
                        stats['embedding_dimensions'] = 0
                else:
                    stats['embedding_dimensions'] = 0
            else:
                stats['embedding_dimensions'] = 0
            
            # Get similarity algorithm info
            stats['similarity_algorithm'] = "Cosine Similarity"  # Default for most vector DBs
            stats['vector_index_type'] = "HNSW"  # Hierarchical Navigable Small World
            
            # Get chunk type distribution
            chunk_types_result = self.supabase_client.rpc("get_chunk_type_distribution").execute()
            if chunk_types_result.data:
                stats['chunk_type_distribution'] = {item['chunk_type']: item['count'] for item in chunk_types_result.data}
            else:
                stats['chunk_type_distribution'] = {}
            
            # Get language distribution
            language_dist_result = self.supabase_client.rpc("get_language_distribution").execute()
            if language_dist_result.data:
                stats['language_distribution'] = {item['language']: item['count'] for item in language_dist_result.data}
            else:
                stats['language_distribution'] = {}
            
            # Get file type distribution
            file_types_result = self.supabase_client.rpc("get_file_type_distribution").execute()
            if file_types_result.data:
                stats['file_type_distribution'] = {item['file_type']: item['count'] for item in file_types_result.data}
            else:
                stats['file_type_distribution'] = {}
            
            # Get storage size estimation (approximate)
            stats['estimated_storage_mb'] = round((stats['total_vectors'] * stats['embedding_dimensions'] * 4) / (1024 * 1024), 2)  # 4 bytes per float
            
            # Database health metrics
            stats['database_status'] = "healthy"
            stats['last_updated'] = self._get_last_update_timestamp()
            
            self.logger.info(f"Retrieved vector DB statistics: {stats['total_vectors']} vectors, {stats['total_documents']} documents")
            return stats
            
        except Exception as e:
            self.logger.error(f"Error retrieving vector DB statistics: {e}")
            # Return default stats in case of error
            return {
                'total_vectors': 0,
                'total_documents': 0,
                'embedding_dimensions': 0,
                'similarity_algorithm': 'Cosine Similarity',
                'vector_index_type': 'HNSW',
                'chunk_type_distribution': {},
                'language_distribution': {},
                'file_type_distribution': {},
                'estimated_storage_mb': 0.0,
                'database_status': 'error',
                'last_updated': None,
                'error': str(e)
            }
    
    def _get_last_update_timestamp(self):
        """Get the timestamp of the most recently added chunk"""
        try:
            result = self.supabase_client.table("chunks").select("created_at").order("created_at", desc=True).limit(1).execute()
            if result.data:
                return result.data[0]['created_at']
            return None
        except Exception as e:
            self.logger.error(f"Error getting last update timestamp: {e}")
            return None
    
    def get_search_performance_stats(self):
        """Get search performance statistics"""
        try:
            # This would require storing search metrics in a separate table
            # For now, return placeholder data
            return {
                'average_search_time_ms': 0,
                'total_searches': 0,
                'cache_hit_rate': 0,
                'most_common_queries': []
            }
        except Exception as e:
            self.logger.error(f"Error retrieving search performance stats: {e}")
            return {
                'average_search_time_ms': 0,
                'total_searches': 0,
                'cache_hit_rate': 0,
                'most_common_queries': [],
                'error': str(e)
            }