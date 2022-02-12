import os

from transformers import T5Tokenizer

from args import init_args
from src.utility import get_config, set_seed
from src.constant import Path
from src.loader import Loader
from src.trainer import T5Trainer

if __name__ == "__main__":
    
    # 1. Initialize scripts argument + setup + dependencies
    args = init_args()
    config_path = args.config
    prefix = args.prefix
    # config_path = "resources/t5-config.yaml"
    configs = get_config(config_path)
    set_seed(configs["main"]["seed"])

    tokenizer = None
    if configs["type"] == "t5":
        model_name = configs.get("main").get("pretrained")
        tokenizer = T5Tokenizer.from_pretrained(model_name)

    # 2. Preparing Dataset ...
    loader = Loader(tokenizer, configs)
    train_loader = loader.get_train_loader()
    val_loader = loader.get_val_loader()

    # 3. Training ...
    trainer = T5Trainer(tokenizer=tokenizer, train_loader=train_loader, val_loader=val_loader, prefix=prefix, configs=configs)
    trainer.fit()
    trainer.save()
    
    load_trainer = T5Trainer(tokenizer=tokenizer, train_loader=train_loader, val_loader=val_loader, prefix=prefix, configs=configs)
    saved_path = os.path.join(Path.MODEL, prefix, Path.CHECKPOINT, "model-state-last.pt")
    load_trainer.load(saved_path)

    # 4. Evaluation ...

    # 5. Inference / Generate ...

    
    # TODO: split do-eval only or do train + eval