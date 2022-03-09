from typing import List, Tuple

from torch.nn.functional import cosine_similarity

from src.postprocess.interface import IPostprocess

class EmbeddingDistancePostProcessor(IPostprocess):
    
    def set_embedding(self, tokenizer, embedding):
        self.embedding = embedding
        self.tokenizer = tokenizer

    # == Cosine Similarity normalization strategy ==
    def recover_term(self, original_term: str, sent: List[str]) -> str:
        words = original_term.split(" ")
        new_words = []
        for word in words:
            cosine_sim = []
            for token in sent:
                cosine_sim.append(self.get_cosine_similarity.eval(word, token))
            smallest_idx = cosine_sim.index(max(cosine_sim))
            new_words.append(sent[smallest_idx])
        new_term = " ".join(new_words)
        return new_term

    def get_cosine_similarity(self, word1, word2):
        token_id1 = self.tokenizer.encode(word1, return_tensors='pt')
        token_id2 = self.tokenizer.encode(word2, return_tensors='pt')

        em1 = self.embedding(token_id1).mean(axis=1)
        em2 = self.embedding(token_id2).mean(axis=1)

        return cosine_similarity(em1.reshape(1,-1), em2.reshape(1,-1)).item()
