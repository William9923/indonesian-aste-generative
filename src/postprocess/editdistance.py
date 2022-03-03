import editdistance

sentiment_word_list = ["positif", "negatif", "netral"]
class EditDistancePostProcessor:
    # == Levenshtein distance normalization strategy ==
    def recover_term(self, original_term, sent):
        words = original_term.split(" ")
        new_words = []
        for word in words:
            edit_dis = []
            for token in sent:
                edit_dis.append(editdistance.eval(word, token))
            smallest_idx = edit_dis.index(min(edit_dis))
            new_words.append(sent[smallest_idx])
        new_term = " ".join(new_words)
        return new_term

    # == Generalize strategy ==
    def fix_preds(self, all_pairs, sents):
        all_new_pairs = []

        for i, pairs in enumerate(all_pairs):
            new_pairs = []
            # -- [Handling failed to parse by system] --
            if pairs == []:
                all_new_pairs.append(pairs)
            else:
                for pair in pairs:
                    at, ot, polarity = pair

                    # --- [Recovering aspect term] ---
                    if at not in sents[i]:
                        new_at = self.recover_term(at, sents[i])
                    else:
                        new_at = at

                    # --- [Recovering polarity term] ---
                    if polarity not in sentiment_word_list:
                        new_sentiment = self.recover_term(polarity, sentiment_word_list)
                    else:
                        new_sentiment = polarity

                    # -- [Recovering opinion term]
                    if ot not in sents[i]:
                        new_ot = self.recover_term(ot, sents[i])
                    else:
                        new_ot = ot

                    new_pairs.append((new_at, new_ot, new_sentiment))
                all_new_pairs.append(new_pairs)
        return all_new_pairs