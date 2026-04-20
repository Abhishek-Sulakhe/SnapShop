"""
Evaluation metrics for the Shopee product matching pipeline.

- micro-F1 : the competition's primary metric
- Threshold optimization : finds the best decision boundary per model
- Sample weighting : inverse group-size weighting for micro-F1
"""

import numpy as np
from collections import defaultdict


def compute_micro_f1(predictions, ground_truth):
    """Compute micro-averaged F1 score for product matching.

    Args
    ----
    predictions  : dict mapping posting_id -> set of predicted matching ids
    ground_truth : dict mapping posting_id -> set of true matching ids

    Returns
    -------
    float : micro-F1 score
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for posting_id in ground_truth:
        pred_set = predictions.get(posting_id, set())
        true_set = ground_truth[posting_id]

        tp = len(pred_set & true_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1


def find_optimal_threshold(scores, labels, thresholds=None):
    """Find the similarity threshold that maximizes binary F1.

    Used per-model before the ensembling step:
      1. Find best threshold for each model on local data
      2. Subtract threshold from predictions
      3. Sum across models
      4. Predict match if sum > 0

    Args
    ----
    scores     : (N,) similarity or probability scores
    labels     : (N,) binary ground truth (1=same group)
    thresholds : optional array of values to sweep

    Returns
    -------
    (best_threshold, best_f1)
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.95, 0.01)

    best_f1 = 0.0
    best_th = 0.5

    for th in thresholds:
        preds = (scores >= th).astype(np.float32)
        tp = np.sum(preds * labels)
        fp = np.sum(preds * (1 - labels))
        fn = np.sum((1 - preds) * labels)

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        if f1 > best_f1:
            best_f1 = f1
            best_th = th

    return best_th, best_f1


def compute_sample_weights(df, power=0.4):
    """Compute per-sample weights inversely proportional to label group size.

    Motivation: micro-F1 gives equal weight to every item.  Without
    weighting, the model over-fits large groups and under-fits small ones.

        weight_i = 1 / (size_of_group_i ** power)

    Competition used power = 0.4 as a sweet spot between uniform
    weighting (power=0) and full inverse (power=1).

    Returns weights normalized so that mean = 1 (no change in effective lr).
    """
    group_sizes = df['label_group'].map(df['label_group'].value_counts())
    weights = 1.0 / (group_sizes.values ** power)
    weights = weights / weights.mean()
    return weights.astype(np.float32)
