def parse(data_path, separator):
    sents, labels = [], []
    with open(data_path, "r", encoding="UTF-8") as fp:
        words = []
        for line in fp:
            line = line.strip()
            if line != "":
                words, targets = line.split(separator)
                sents.append(words.split())
                labels.append(eval(targets))
    return sents, labels
