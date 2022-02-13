import os
from pprint import pprint

from transformers import T5Tokenizer

from args import init_args
from src.utility import get_config, set_seed
from src.constant import Path
from src.loader import Loader
from src.trainer import T5Trainer
from src.evaluation import Evaluator

if __name__ == "__main__":

    # 1. Initialize scripts argument + setup + dependencies
    args = init_args()
    config_path = args.config
    prefix = args.prefix
    # config_path = "resources/t5-config.yaml"
    configs = get_config(config_path)
    set_seed(configs["main"]["seed"])

    mode = configs.get("main").get("mode")

    tokenizer = None
    if configs["type"] == "t5":
        model_name = configs.get("main").get("pretrained")
        tokenizer = T5Tokenizer.from_pretrained(model_name)

    # 2. Preparing Dataset ...
    loader = Loader(tokenizer, configs)

    train_loader = loader.get_train_loader()
    train_dataset = loader.get_train_dataset()
    train_sents = train_dataset.get_sents()

    val_loader = loader.get_val_loader()
    val_dataset = loader.get_val_dataset()
    val_sents = val_dataset.get_sents()

    test_loader = loader.get_test_loader()
    test_dataset = loader.get_test_dataset()
    test_sents = test_dataset.get_sents()

    if mode == "train":
        print("\n".join(train_dataset.get_stats("Training")))
        print("\n".join(val_dataset.get_stats("Validation")))
    else:
        print("\n".join(test_dataset.get_stats("Testing")))

    # 3. Training (skip if do-eval only)
    if mode == "train":
        trainer = T5Trainer(
            tokenizer=tokenizer,
            train_loader=train_loader,
            val_loader=val_loader,
            prefix=prefix,
            configs=configs,
        )
        trainer.fit()
        trainer.save()

    load_trainer = T5Trainer(
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        prefix=prefix,
        configs=configs,
    )

    saved_path = os.path.join(Path.MODEL, prefix)
    load_trainer.load(saved_path)
    model = load_trainer.get_model()

    # 4. Evaluation ...
    evaluator = Evaluator(configs)
    if mode == "train":
        evaluator.export("train", prefix, tokenizer, model, train_loader, train_sents)
        evaluator.export("val", prefix, tokenizer, model, val_loader, val_sents)
    else:
        pprint(evaluator.evaluate(tokenizer, model, test_loader, test_sents))
        evaluator.export("test", prefix, tokenizer, model, test_loader, test_sents)

    # 5. Inference / Generate ... -> only be used for demo only
    # sents = [
    #     "pelayanan ramah , kamar nyaman dan fasilitas lengkap . hanya airnya showernya kurang panas .",
    #     "tidak terlalu jauh dari pusat kota .",
    #     "dengan harga terjangkau kita sudah mendapatkan fasilitas yang nyaman .",
    #     "kamar luas dan bersih . seprai bersih .",
    #     "seprai nya kurang bersih .",
    #     "kamarnya bersih dan rapi . saya kebetulan dapat yang di lantai dua .",
    # ]

    # generator = T5Generator(tokenizer, model, configs)
    # res = generator.generate(sents, fix=True)
    # pprint(res)
