import os
from utils.logger import Logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import spacy
from config.settings import *

class Reranker:
    
    def __init__(self):
        self.logger = Logger(name="RAGLogger").get_logger()
        self.vectorizer = TfidfVectorizer()
        self.scaler = MinMaxScaler()
        self.nlp = spacy.load("en_core_web_sm")
        
    def getTfIDFScore(self, query, chunks, percentage_val):
        corpus = [query] + [chunk['content'] for chunk in chunks]
        tfidf_mat = self.vectorizer.fit_transform(corpus)
        tfidf_sims = cosine_similarity(tfidf_mat[0:1], tfidf_mat[1:]).flatten()
        np_scores = np.array([tfidf_sims[i] for i in range(len(chunks))]).reshape(-1, 1)
        norm_scores = self.scaler.fit_transform(np_scores).flatten()
        
        for i in range(len(chunks)):
            chunks[i]['final_score'] += norm_scores[i]*(percentage_val/100)
        return chunks
    
    def getSimNormalizedScore(self, chunks, percentage_val):
        vector_sim_scores = np.array([chunk['similarity'] for chunk in chunks]).reshape(-1, 1)
        norm_vectors = self.scaler.fit_transform(vector_sim_scores).flatten()
        for i in range(len(chunks)):
            chunks[i]['final_score'] += norm_vectors[i]*(percentage_val/100)

        return chunks
    
    def getKWOverlapScore(self, query, chunks, percentage_val):
        doc = self.nlp(query)
        query_entities = set([ent.text.lower() for ent in doc.ents if ent.text not in ENTITIES_TO_IGNORE])
        for i, chunk in enumerate(chunks):
            chunk_kws = set([kw.lower() for kw in chunk['semantic_info']['keywords']])
            overlap = query_entities & chunk_kws
            chunks[i]['kw_scores'] = len(overlap)/len(chunk_kws) if chunk_kws else 0.0
        
        kw_scores_ = np.array([c['kw_scores'] for c in chunks]).reshape(-1, 1)
        norm_kw = self.scaler.fit_transform(kw_scores_).flatten()
        
        for i in range(len(chunks)):
            chunks[i]['final_score'] += norm_kw[i]*(percentage_val/100)
        
        return chunks
    
    def verify_percent_dic(self, importance_dict):
        sum_ = 0
        for i in importance_dict.values():
            sum_ += i
        if sum_ == 100 or sum_ > 99: # if the sum adds to 100
            return True
        return False
    
    def rerank(self, query, retrieved_chunks, importance_dict):
        print("Chunks before sorting: ", [chunk['chunk_id'] for chunk in retrieved_chunks])
        if not self.verify_percent_dic(importance_dict=importance_dict):
            self.logger.error("Provided percentages do not add up to 100 please check.")
            return retrieved_chunks    
        
        # initialize the key = "final_score" to all the chunks to make life easy
        for i in range(len(retrieved_chunks)):
            retrieved_chunks[i]['final_score'] = 0
        
        for key, value in importance_dict.items():
            if key == "tfidf":
                retrieved_chunks = self.getTfIDFScore(query=query, chunks=retrieved_chunks, percentage_val = value)
            elif key == "kw":    
                retrieved_chunks = self.getKWOverlapScore(query=query, chunks=retrieved_chunks, percentage_val = value)
            elif key == "vec_sim":
                retrieved_chunks = self.getSimNormalizedScore(chunks=retrieved_chunks, percentage_val = value)
        
        retrieved_chunks = sorted(retrieved_chunks, key=lambda x: x['final_score'], reverse=True)
        print("Chunks after sorting: ", [chunk['chunk_id'] for chunk in retrieved_chunks])
        return retrieved_chunks