from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from omegaconf import DictConfig


def compute_label_metrics(labels, preds, cfg: DictConfig):
    """
    ラベルごとのメトリクスを計算する

    Parameters
    ----------
    labels : np.ndarray
        真のラベル
    preds : np.ndarray
        予測ラベル
    cfg : DictConfig
        エントリポイントでhydra.mainから取得する設定

    Returns
    -------
    dict
        ラベルごとのメトリクスを格納したdict
    """
    label_metrics = {}
    unique_labels = np.unique(labels)
    for label in unique_labels:
        label_precision = precision_score(
            labels,
            preds,
            labels=[label],
            average=cfg.evaluation.metric_average,
            zero_division=np.nan if cfg.evaluation.zero_division == "np.nan" else cfg.evaluation.zero_division,
        )
        label_recall = recall_score(
            labels,
            preds,
            labels=[label],
            average=cfg.evaluation.metric_average,
            zero_division=np.nan if cfg.evaluation.zero_division == "np.nan" else cfg.evaluation.zero_division,
        )
        label_f1 = f1_score(
            labels,
            preds,
            labels=[label],
            average=cfg.evaluation.metric_average,
            zero_division=np.nan if cfg.evaluation.zero_division == "np.nan" else cfg.evaluation.zero_division,
        )
        label_metrics[label] = {
            "precision": label_precision,
            "recall": label_recall,
            "f1": label_f1,
        }
    return label_metrics
