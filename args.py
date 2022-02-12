import argparse

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True, help="prefix for models")
    parser.add_argument("--config", required=True, help="config for models")
    return parser.parse_args()



