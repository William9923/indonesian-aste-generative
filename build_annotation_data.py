import pandas as pd
import os

TSV_SEPERATOR = "\t"

BIOTAG_SEPERATOR = "//"
TOKEN_SEPERATOR = " "
ANNOTATION_SEPERATOR = "####"
LABEL_SEPERATOR = ", "

def load(path):
    df = pd.read_csv(path)

    idx = list(df.idx)
    sents = [eval(sent) for sent in list(df.sents)]
    labels = [eval(label) for label in list(df.labels)]

    assert len(idx) == len(sents) == len(labels)
    return idx, sents, labels 

def parse(file, separator):
    sents, labels = [], []
    words = []
    for line in file:
        line = line.strip()
        if line != "":
            words, targets = line.split(separator)
            sents.append(words.split())
            labels.append(eval(targets))
    return sents, labels

def combine(file, seperator, idx, annotated_labels):
    curr_sents, curr_labels = parse(file, seperator)
    for i in range(len(idx)):
        curr_labels[idx[i] - 1] = annotated_labels[i]
    return curr_sents, curr_labels 

def write_annotated(sents, labels, target_path):
    assert len(sents) == len(labels)
    with open(target_path, "w") as fout:
        for i in range(len(sents)):
            text, label = sents[i], labels[i]

            fout.write(" ".join(text) + ANNOTATION_SEPERATOR)
            label = [str(l) for l in label]
            fout.write("[" + str(LABEL_SEPERATOR.join(label)) + "]" + "\n")


if __name__ == "__main__":
    idx, _, labels = load(os.path.join("data", "annotation", "test-annotated.csv"))
    
    with open(os.path.join("data", "processed", "unfilter", "test.txt"), 'r') as f:
        sents, labels = combine(f, ANNOTATION_SEPERATOR, idx, labels)

    write_annotated(sents, labels, os.path.join("data", "processed", "implicit", "test.txt"))
    
