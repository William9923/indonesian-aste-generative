# Postprocess Interface to check prediction and make normalization to non-correct aspect/sentiment term...
# Current Implementation: EditDistancePostProcessor, EmbeddingDistancePostProcessor
# Postprocess Interface:
from typing import Tuple, List

sentiment_word_list = ["positif", "negatif", "netral"]
class IPostprocess:
    def check_and_fix_preds(self, all_pairs:List[List[Tuple[str, str, str]]], sents: List[List[str]]) ->List[List[Tuple[str, str, str]]]:
        pass 
    def recover_term(self, original_term:str, sent:List[str]) -> str:
        pass  

