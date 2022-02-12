from args import init_args
from src.utility import get_config, set_seed
from src.loader import Loader

if __name__ == "__main__":
    
    # 1. Initialize scripts argument + setup...
    args = init_args()
    config_path = args.config
    prefix = args.prefix
    # config_path = "resources/t5-config.yaml"
    configs = get_config(config_path)
    set_seed(configs["main"]["seed"])

    # 2. Preparing Dataset ...
    loader = Loader(configs)
    train_dataset = loader.get_train_dataset()
    test_dataset = loader.get_test_dataset()
    val_dataset = loader.get_val_dataset()
    print("\n".join(train_dataset.get_stats("Train")))
    train_loader = loader.get_train_loader()
    test_loader = loader.get_test_loader()
    val_loader = loader.get_val_loader()
    
    # 3. Training ...

    # 4. Evaluation ...

    # 5. Inference / Generate ...

    
    # TODO: split do-eval only or do train + eval