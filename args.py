import argparse

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True, help="prefix for models")
    parser.add_argument("--config", required=True, help="config for models")
    parser.add_argument("--do-train", action=argparse.BooleanOptionalAction, help="perform training")
    parser.add_argument("--do-test", action=argparse.BooleanOptionalAction, help="perform testing")
    return parser.parse_args()



