"""Binary classification metrics.

中文说明：
这个模块实现二分类任务常用指标：accuracy、F1、MCC、AUROC 和 AUPRC。
输入 y_prob 应为正类概率。
"""

from typing import List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


def compute_binary_classification_metrics(y_true: List[int], y_prob: List[float]):
    """Compute binary classification metrics from labels and positive probabilities.

    中文说明：
    根据真实标签 y_true 和正类概率 y_prob 计算二分类指标。
    默认使用 0.5 作为阈值把概率转换为预测标签。
    """
    y_true_array = np.asarray(y_true, dtype=int)
    y_prob_array = np.asarray(y_prob, dtype=float)
    y_pred_array = (y_prob_array >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true_array, y_pred_array),
        "f1": f1_score(y_true_array, y_pred_array, zero_division=0),
        "mcc": matthews_corrcoef(y_true_array, y_pred_array),
    }

    # AUROC/AUPRC require both positive and negative labels.
    # 中文：AUROC/AUPRC 需要同时存在正负样本，否则返回 NaN。
    if len(np.unique(y_true_array)) < 2:
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")
    else:
        metrics["auroc"] = roc_auc_score(y_true_array, y_prob_array)
        metrics["auprc"] = average_precision_score(y_true_array, y_prob_array)

    return metrics
