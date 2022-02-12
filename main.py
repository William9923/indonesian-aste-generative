from args import init_args
from src.utility import get_config, set_seed
from pprint import pprint

if __name__ == "__main__":
    
    # 1. Initialize scripts argument + setup...
    args = init_args()
    config_path = args.config
    prefix = args.prefix
    configs = get_config(config_path)
    set_seed(configs["main"]["seed"])

    # 2. Preparing Dataset ...
 
    # 3. Training ...

    # 4. Evaluation ...

    # 5. Inference / Generate ...

    
    # TODO: split do-eval only or do train + eval