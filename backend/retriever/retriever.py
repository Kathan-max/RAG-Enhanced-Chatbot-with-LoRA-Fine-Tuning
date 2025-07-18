from database.vector.vectorDB import VectorDB
from config.settings import *
from llmservice.llmhelper import LLmHelper
from utils.logger import Logger
import copy
from retriever.reranker.reranker import Reranker

class Retriever:
    
    def __init__(self, generation_model = DEFAULT_GENERATION_MODEL):
        self.vectordb = VectorDB()
        self.llmhelper_ = LLmHelper(generation_model=generation_model)
        self.logger = Logger(name="RAGLogger").get_logger()
        self.reranker = Reranker()
        
    def fetchChunksByIds(self, chunk_ids):
        return self.vectordb.getChunksByID(chunk_ids=chunk_ids)

    def combineMediaRef(self, *media_refs):
        image_dict = {}
        for obj_ref in media_refs:
            if obj_ref.get('images'):
                for obj in obj_ref['images']:
                    image_dict[obj['image_id']] = obj
        return list(image_dict.values())
        
    def combineSemanticInfo(self, *semantic_objs):
        final_keys_words = []
        for sem_obj in semantic_objs:
            if sem_obj.get('keywords'):
                final_keys_words += sem_obj['keywords']
        return {'keywords': list(set(final_keys_words))}
    
    def combineTwoChunks(self, first_chunk, second_chunk, prev_flag):
        # combined_chunk = first_chunk.copy()
        combined_chunk = copy.deepcopy(first_chunk)
        if prev_flag:
            combined_chunk['content'] = second_chunk['content'] + " " + first_chunk['content']
        else:
            combined_chunk['content'] = first_chunk['content'] + " " + second_chunk['content']
            
        combined_chunk['media_ref']['images'] = self.combineMediaRef(first_chunk['media_ref'], second_chunk['media_ref'])
        combined_chunk['semantic_info'] = self.combineSemanticInfo(first_chunk['semantic_info'], second_chunk['semantic_info'])
        
        return combined_chunk

    def recursiveCombine(self, curr_chunk, curr_id, tree, ideal_tokens, chunk_look_up):
        
        if self.llmhelper_.getTokens(curr_chunk['content']) >= ideal_tokens:
            return curr_chunk
        
        # if "similarity" not in curr_chunk:
            # print("In the rec call not found in the chunk: ", curr_chunk['chunk_id'])
        new_curr_chunk = curr_chunk.copy()
        
        if curr_id in tree and 'prev' in tree[curr_id]:
            prev_id = tree[curr_id]['prev']
            if prev_id in chunk_look_up and chunk_look_up[prev_id] != "Dummy Data string":
                prev_node = chunk_look_up[prev_id]
                new_curr_chunk = self.combineTwoChunks(curr_chunk, prev_node, prev_flag=True)
                
                return self.recursiveCombine(curr_chunk=new_curr_chunk, curr_id=prev_id,
                                             tree=tree, ideal_tokens=ideal_tokens, chunk_look_up=chunk_look_up)
            
        if curr_id in tree and 'next' in tree[curr_id]:
            next_id = tree[curr_id]['next']
            if next_id in chunk_look_up and chunk_look_up[next_id] != "Dummy Data string":
                next_node = chunk_look_up[next_id]
                new_curr_chunk = self.combineTwoChunks(curr_chunk, next_node, prev_flag=False)
                return self.recursiveCombine(curr_chunk=new_curr_chunk, curr_id=next_id,
                                             tree = tree, ideal_tokens=ideal_tokens, chunk_look_up=chunk_look_up)

        return new_curr_chunk        
        
    def combineChunks(self, tree, root_node_ids, chunk_look_up, ideal_chunk_tokens):
        final_chunks = []
        processed_chunks = set()
        # print("Root ids: ", root_node_ids)
        for root_node in root_node_ids:
            if root_node in processed_chunks:
                continue
            if root_node in chunk_look_up and chunk_look_up[root_node] != "Dummy Data string":
                
                new_chunk = self.recursiveCombine(curr_chunk=chunk_look_up[root_node],
                                                  curr_id=root_node,
                                                  tree = tree,
                                                  ideal_tokens=ideal_chunk_tokens,
                                                  chunk_look_up=chunk_look_up)
                # if "similarity" not in new_chunk:
                #     print("Not found in the chunk: ", new_chunk['chunk_id'])
                final_chunks.append(new_chunk)
                processed_chunks.add(root_node)
        
        return final_chunks
                
                
    
    def fetchChains(self, retrieved_chunks, num_of_neighbors, top_k = TOP_K):
        chunk_look_up = {}
        tree = {}
        curr_chunks = retrieved_chunks
        root_ids = []
        
        ideal_chunk_tokens = self.llmhelper_.idealChunkTokens(top_k=top_k, chunks_coverage=TOTAL_CHUNKS_COVERAGE)
        
        for chunk in retrieved_chunks:
            root_ids.append(chunk['chunk_id'])
            chunk_look_up[chunk['chunk_id']] = chunk
        
        while(num_of_neighbors >= 0):
            chunks_to_fetch = []
            
            for chunk in curr_chunks:
                if chunk['chunk_id'] in tree:
                    continue
                
                tree[chunk['chunk_id']] = {}
                # chunk_look_up[chunk['chunk_id']] = chunk
                
                prev_c = chunk['chunk_info']['prev_chunk_id']
                if prev_c and (prev_c not in chunk_look_up):
                    tree[chunk['chunk_id']]['prev'] = prev_c
                    chunk_look_up[prev_c] = "Dummy Data string"
                    chunks_to_fetch.append(prev_c)
                
                next_c = chunk['chunk_info']['next_chunk_id']
                if next_c and (next_c not in chunk_look_up):
                    tree[chunk['chunk_id']]['next'] = next_c
                    chunk_look_up[next_c] = "Dummy Data string"
                    chunks_to_fetch.append(next_c)
            
            if chunks_to_fetch:
                # print("Ids to fetch: ", chunks_to_fetch)
                curr_chunks = self.fetchChunksByIds(chunk_ids= chunks_to_fetch)
                # print("Fetched total: ", len(curr_chunks), "Neighbors: ", num_of_neighbors)
                for chunk in curr_chunks:
                    if chunk['chunk_id'] in chunk_look_up:
                        chunk_look_up[chunk['chunk_id']] = chunk
            else:
                curr_chunks = []
            # curr_chunks = self.fetchChunksByIds(chunk_ids=chunks_to_fetch)
            num_of_neighbors -= 1
        
        return self.combineChunks(tree=tree, root_node_ids=root_ids, 
                                  chunk_look_up=chunk_look_up, ideal_chunk_tokens=ideal_chunk_tokens)
    
    def reRankChunks(self, query, retrieved_chunks, importance_dict):
        return self.reranker.rerank(query = query, retrieved_chunks=retrieved_chunks, importance_dict=importance_dict)
    
    def retrieveImages(self, retrieved_chunks):
        image_urls = set()
        for obj in retrieved_chunks:
            if obj['media_ref']['images'] != []:
                for image_obj in obj['media_ref']['images']:
                    image_urls.add(image_obj['image_url'])
        image_base64_dict = {}
        if len(image_urls)>0:
            image_base64_dict = self.vectordb.retrieve_images(image_urls=image_urls)
            
        return image_base64_dict
    
    def retrieveTopK(self, query, file_filter = {}, top_k = TOP_K,
                       chunk_filer = {}, language_filter = None, 
                       chunk_type = None, encoder = None, fetch_chains = False, num_of_neighbors = 0):
                
        retrieved_chunks = self.vectordb.retrieve_chunks(query=query,
                                                         top_k=top_k,
                                                         similarity_threshold=SIMILARITY_TH,
                                                         file_filter=file_filter,
                                                         langauge_filter=language_filter,
                                                         chunk_filter=chunk_filer,
                                                         chunk_type=chunk_type,
                                                         encoder=encoder)
        
        # original_chunks = retrieved_chunks.copy()
        
        if fetch_chains:
            retrieved_chunks = self.fetchChains(retrieved_chunks=retrieved_chunks, 
                                                num_of_neighbors=num_of_neighbors, 
                                                top_k=top_k)
        
        retrieved_images_dict = self.retrieveImages(retrieved_chunks = retrieved_chunks)
        reranked_chunks = self.reRankChunks(query = query, retrieved_chunks=retrieved_chunks, importance_dict = RERANKING_PARAMETERS_PERCENT)
        
        return reranked_chunks