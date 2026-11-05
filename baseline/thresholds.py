"""Choose one threshold per class, using validation predictions only."""

import numpy as np
from sklearn.metrics import precision_recall_curve


def select_thresholds(labels, probabilities):
    thresholds = []
    for j in range(labels.shape[1]):
        if len(np.unique(labels[:, j])) != 2:
            raise ValueError("Threshold selection needs positive and negative validation labels")
        precision, recall, candidates = precision_recall_curve(labels[:, j], probabilities[:, j])
        f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
        best = np.flatnonzero(np.isclose(f1, f1.max(), rtol=0, atol=1e-12))
        # Prefer the threshold closest to 0.5, then the larger one, on equal F1.
        index = min(best, key=lambda i: (abs(float(candidates[i]) - 0.5), -float(candidates[i])))
        thresholds.append(float(candidates[index]))
    return thresholds
