from typing import List

import numpy as np

from torch.nn.functional import cosine_similarity
from src.postprocess.interface import IPostprocess, special_chars
from src.utility import get_device

from scipy import spatial

def get_embedding(wv, word):
    try:
        emb = wv.get_vector(word)
    except KeyError:
        emb = np.zeros(300)
    return emb

class EmbeddingDistancePostProcessor(IPostprocess):
    
    def set_embedding(self, embedding):
        self.embedding = embedding

    # == Cosine Similarity normalization strategy ==
    def recover_term(self, original_term: str, sent: List[str]) -> str:
        words = original_term.split(" ")
        new_words = []
        for word in words:
            cosine_sim = []
            for token in sent:
                if token in sent:
                    cosine_sim.append(2) 
                if token not in special_chars:
                    cosine_sim.append(self.get_cosine_similarity(word, token))
                else:
                    cosine_sim.append(-1)
            smallest_idx = cosine_sim.index(max(cosine_sim))
            new_words.append(sent[smallest_idx])
        new_term = " ".join(new_words)
        return new_term

    def get_cosine_similarity(self, word1, word2):
        cosine_similarity = lambda em1, em2 : 1 - spatial.distance.cosine(em1, em2)
        em1 = get_embedding(self.embedding, word1)
        em2 = get_embedding(self.embedding, word2)
        return cosine_similarity(em1, em2)
        

    
