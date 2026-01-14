import numpy as np

def best_threshold_by_f1(scores, y_true):
    scores = np.asarray(scores)
    y_true = np.asarray(y_true)

    candidates = np.percentile(scores, np.linspace(50, 99.9, 250))
    best_t, best_f1 = float(candidates[0]), -1.0

    for t in candidates:
        y_pred = (scores >= t).astype(int)

        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)

        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)

    return best_t, best_f1

def pr_at_threshold(scores, y_true, threshold: float):
    scores = np.asarray(scores)
    y_true = np.asarray(y_true)
    y_pred = (scores >= threshold).astype(int)

    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)

    return float(precision), float(recall)
