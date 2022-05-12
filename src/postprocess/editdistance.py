import editdistance
from typing import List

from src.postprocess.interface import IPostprocess, special_chars

class EditDistancePostProcessor(IPostprocess):
    # == Levenshtein distance normalization strategy ==
    def recover_term(self, original_term: str, sent: List[str]) -> str:
        words = original_term.split(" ")
        new_words = []
        for word in words:
            if word in special_chars: # make sure not to recover on punctuation (or any special char)...
                continue
            edit_dis = []
            for token in sent:
                edit_dis.append(editdistance.eval(word, token))
            smallest_idx = edit_dis.index(min(edit_dis))
            new_words.append(sent[smallest_idx])
        new_term = " ".join(new_words)
        return new_term
