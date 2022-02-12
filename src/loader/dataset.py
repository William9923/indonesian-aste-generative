from torch.utils.data import Dataset
from copy import deepcopy

senttag2word = {"POS": "positif", "NEG": "negatif", "NEU": "netral"}


class HotelDataset(Dataset):
    """
    Construct PyTorch dataset object for Indonesian Hotel Review Text Dataset
    sents, labels, paradigm, tokenizer, max_length=128
    Args :
        sents -> List[str] : collenction of sentences as inputs
        labels -> List[str] : collection of targets depending on paradigm
        tokenizer -> Tokenizer: Transformer Tokenizer for T5
        max_length -> int: maximum word length per instance

    Typical usage example:
    dataset = HotelDataset(sents, labels, paradigm, tokenizer)

    or

    params = {
        "sents": sents,
        "labels": labels,
        "tokenizer" : tokenizer,
        "max_length" : 128,
    }
    dataset = HotelDataset(**params)
    """

    def __init__(self, sents, labels, tokenizer, configs):
        self.tokenizer = tokenizer
        self.sents = sents
        self.labels = labels
        self.configs = configs

        self.max_length = configs.get("loader").get("max_seq_length")

        self.inputs = []
        self.targets = []

        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()
        target_mask = self.targets[index]["attention_mask"].squeeze()

        return {
            "source_ids": source_ids,
            "source_mask": src_mask,
            "target_ids": target_ids,
            "target_mask": target_mask,
        }

    def _build(self):
        func = generate_extraction_style_target
        inputs = deepcopy(self.sents)
        targets = func(self.sents, self.labels)
        for i in range(len(inputs)):
            input_item = " ".join(inputs[i])
            tokenized_input = self.tokenizer.batch_encode_plus(
                [input_item],
                max_length=self.configs.get("loader").get("max_seq_length"),
                padding=self.configs.get("loader").get("padding"),
                truncation=self.configs.get("loader").get("truncation"),
                return_tensors="pt",
            )

            target_item = targets[i]
            tokenized_target = self.tokenizer.batch_encode_plus(
                [target_item],
                max_length=self.configs.get("loader").get("max_seq_length"),
                padding=self.configs.get("loader").get("padding"),
                truncation=self.configs.get("loader").get("truncation"),
                return_tensors="pt",
            )

            self.inputs.append(tokenized_input)
            self.targets.append(tokenized_target)


def generate_extraction_style_target(sents_e, labels):
    """Helper func for generating target for GAS extraction-style paradigm"""
    extracted_targets = []
    for i, label in enumerate(labels):
        all_tri = []
        for tri in label:
            if len(tri[0]) == 1:
                aspect = sents_e[i][tri[0][0]]
            else:
                start_idx, end_idx = tri[0][0], tri[0][-1]
                aspect = " ".join(sents_e[i][start_idx : end_idx + 1])
            if len(tri[1]) == 1:
                sentiment = sents_e[i][tri[1][0]]
            else:
                start_idx, end_idx = tri[1][0], tri[1][-1]
                sentiment = " ".join(sents_e[i][start_idx : end_idx + 1])
            polarity = senttag2word[tri[2]]
            all_tri.append((aspect, sentiment, polarity))
        label_strs = ["(" + ", ".join(l) + ")" for l in all_tri]
        extracted_targets.append("; ".join(label_strs))
    return extracted_targets
