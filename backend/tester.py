from database.vector.vectorDB import VectorDB
from embedding.encoder import Encoder
from retriever.retriever import Retriever
import json

vector_db = VectorDB()

query = """At each step the model is auto-regressive [10], consuming the previously generated symbols as additional input when generating the next..<image><image_id>2c851fc8-d71d-4b7a-9990-c7b827a55698</image_id>This graph illustrates the architecture of a Transformer model used for sequence-to-sequence tasks. The model consists of an encoder (left) and a decoder (right). The encoder processes the input sequence through multiple layers of multi-head attention and feed-forward neural networks, each followed by an add & norm step. Positional encoding is added to the input embeddings to incorporate the order of the sequence. The decoder processes the output sequence similarly, but it also includes masked multi-head attention to prevent it from attending to future positions. The final output probabilities are generated through a linear layer followed by a softmax layer. The diagram shows the flow of data through these components, highlighting the interactions between different parts of the model.</image>

Figure 1: The Transformer - model architecture..The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1, respectively..# 3.1 Encoder and Decoder Stacks 

Encoder: The encoder is composed of a stack of $N=6$ identical layers..Each layer has two sub-layers..The first is a multi-head self-attention mechanism, and the second is a simple, positionwise fully connected feed-forward network..We employ a residual connection [11] around each of the two sub-layers, followed by layer normalization [1]..That is, the output of each sub-layer is $\operatorname{LayerNorm}(x+\operatorname{Sublayer}(x))$, where $\operatorname{Sublayer}(x)$ is the function implemented by the sub-layer itself..To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension $d_{\text {model }}=512$..Decoder: The decoder is also composed of a stack of $N=6$ identical layers..In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack..Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization..We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions..This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position $i$ can depend only on the known outputs at positions less than $i$..### 3.2 Attention

An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors..The output is computed as a weighted sum"""

"""
query, top_k = 5, 
                        similarity_threshold = 0.7, 
                        file_filter = {},
                        chunk_filter = {},
                        langauge_filter = None,
                        chunk_type = None,
                        encoder = None
"""

# retrieved_chunks = vector_db.retrieve_chunks(query=query, top_k=5, encoder=Encoder())
# print(retrieved_chunks)

retriver = Retriever()

reranked_chunks, original_chunks  = retriver.retrieveTopK(query=query, top_k=7, encoder=Encoder(), fetch_chains=True, num_of_neighbors=3)

with open("original_data.json", 'w') as f:
    json.dump(original_chunks, f, indent=4)

with open("reranked_data.json", 'w') as f:
    json.dump(reranked_chunks, f, indent=4)
