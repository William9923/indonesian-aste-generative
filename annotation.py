import pandas as pd
import os

def load(file, separator):
    sents, labels = [], []
    words = []
    for line in file:
        line = line.strip()
        if line != "":
            words, targets = line.split(separator)
            sents.append(words.split())
            labels.append(eval(targets))
    idx = range(1, len(sents) + 1)

    return pd.DataFrame({
        "idx": idx,
        "sents": sents,
        "labels" : [str(label) for label in labels],
    })

def process(path, filename):
    with open(path, 'r') as f:
        csv = load(f, "####")
        csv.to_csv(os.path.join("data", "annotation", filename), index=False)

if __name__ == "__main__":
    path = os.path.join("data", "processed", "unfilter")

    train_path = os.path.join(path, "train.txt")
    test_path = os.path.join(path, "test.txt")
    val_path = os.path.join(path, "dev.txt")

    process(train_path, "train.csv")
    process(test_path, "test.csv")
    process(val_path, "val.csv")

    

    
