from src.utility import get_device, extract


class Generator:
    def __init__(self, tokenizer, model, postprocessor, configs):
        self.device = get_device()
        self.model = model
        self.tokenizer = tokenizer
        self.postprocessor = postprocessor
        self.configs = configs

    def generate(self, sents, fix=False):
        pass


class T5Generator(Generator):
    def __init__(self, tokenizer, model, postprocessor, configs):
        super().__init__(tokenizer, model, postprocessor, configs)

    def generate(self, sents, fix=False):

        # --- [Preprocessing] ---
        splitted_sents = [txt.split(" ") for txt in sents]
        max_length = self.configs.get("loader").get("max_seq_length")

        # --- [Tokenization] ---
        batch = self.tokenizer.batch_encode_plus(
            sents,
            max_length=max_length,
            padding=self.configs.get("loader").get("padding"),
            truncation=self.configs.get("loader").get("truncation"),
            return_tensors="pt",
        )

        # --- [Generating Triplets Opinion] ---
        self.model.eval()
        # Method: Greedy searchx
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
            outputs = self.postprocessor.fix_preds(all_preds, splitted_sents)

        return outputs
