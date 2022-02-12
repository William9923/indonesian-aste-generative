from src.utility import extract
from src.generator import fix_preds

# == Metrics ==
precision_fn = lambda n_tp, n_pred: float(n_tp) / float(n_pred) if n_pred != 0 else 0
recall_fn = lambda n_tp, n_gold: float(n_tp) / float(n_gold) if n_gold != 0 else 0
f1_fn = (
    lambda precision, recall: (2 * precision * recall) / (precision + recall)
    if precision != 0 or recall != 0
    else 0
)


def score(pred, gold):
    assert len(pred) == len(gold)
    n_tp, n_gold, n_pred = 0, 0, 0

    for i in range(len(pred)):
        n_gold += len(gold[i])
        n_pred += len(pred[i])

        for t in pred[i]:
            if t in gold[i]:
                n_tp += 1

    precision = precision_fn(n_tp, n_pred)
    recall = recall_fn(n_tp, n_gold)
    f1 = f1_fn(precision, recall)

    return {"precision": precision, "recall": recall, "f1": f1}


# == Evaluation ==
def evaluate(pred_seqs, gold_seqs, sents):
    assert len(pred_seqs) == len(gold_seqs)
    num_samples = len(gold_seqs)

    all_labels, all_preds = [], []

    for i in range(num_samples):
        gold_list = extract(gold_seqs[i])
        pred_list = extract(pred_seqs[i])

        all_labels.append(gold_list)
        all_preds.append(pred_list)

    raw_scores = score(all_preds, all_labels)

    all_fixed_preds = fix_preds(all_preds, sents)
    fixed_scores = score(all_fixed_preds, all_labels)

    return raw_scores, fixed_scores, all_labels, all_preds, all_fixed_preds
