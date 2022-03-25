import os
import random

senttag2word = {"POS": "positif", "NEG": "negatif", "NEU": "netral"}

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

def generate_extraction_style_target(sents_e, labels):
    """Helper func for generating target for GAS extraction-style paradigm"""
    extracted_targets = []
    found = False
 
    for i, label in enumerate(labels):
     
        all_tri = []
        for tri in label:
            try:
                if len(tri[0]) == 1:
                    if tri[0][0] == -1: # implicit
                        aspect = "hotel" 
                    else:
                        aspect = sents_e[i][tri[0][0]]
                else:
                    start_idx, end_idx = tri[0][0], tri[0][-1]
                    aspect = " ".join(sents_e[i][start_idx : end_idx + 1])
                if len(tri[1]) == 1:
                    sentiment = sents_e[i][tri[1][0]]
                else:
                    start_idx, end_idx = tri[1][0], tri[1][-1]
                    sentiment = " ".join(sents_e[i][start_idx : end_idx + 1])
                polarity = senttag2word[tri[2]]
                all_tri.append((aspect, sentiment, polarity))
            
            except:
                print(sents_e[i])
                print(i)
                print(label)
                print(tri)
                found = True
        label_strs = ["(" + ", ".join(l) + ")" for l in all_tri]
        extracted_targets.append("; ".join(label_strs))

        if found:
            break
    return extracted_targets

if __name__ == "__main__":
    with open(os.path.join("data", "processed", "implicit", "dev.txt"), 'r') as f:
        sents, labels = parse(f, "####")

    print("Problematic...")
    targets = generate_extraction_style_target(sents, labels)
    assert len(sents) == len(targets)

    # View some example
    memory = set()
    print(targets[:5])

    while len(memory) < 50:
        idx = random.randint(0, len(targets))
        memory.add(idx)
    
    for idx in memory:
        print(f"Sentence: {' '.join(sents[idx])}")
        print(f"Targets: {targets[idx]}")

    print("Done...")