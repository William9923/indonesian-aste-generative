from src.utility import get_device, extract
from src.generator.normalization import fix_preds


class T5Generator:
    def __init__(self, tokenizer, model, configs):
        self.device = get_device()
        self.model = model
        self.tokenizer = tokenizer
        self.configs = configs

    def generate(self, sents, fix=False):
        # --- [Preprocessing] ---
        splitted_sents = [txt.split(" ") for txt in sents]
        max_length = self.configs.get("loader").get("max_seq_length")
        # --- [Tokenization] ---
        batch = self.tokenizer.batch_encode_plus(
            sents,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # --- [Generating Triplets Opinion] ---
        self.model.eval()
        outs = self.model.generate(
            input_ids=batch["input_ids"].to(self.device),
            attention_mask=batch["attention_mask"].to(self.device),
            max_length=max_length,
        )

        # --- [Decoding] ---
        outputs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]

        # --- [Normalization, if needed] ---
        if fix:
            all_preds = []
            for out in outputs:
                all_preds.append(extract(out))
            outputs = fix_preds(all_preds, splitted_sents)

        return outputs
