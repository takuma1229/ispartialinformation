from transformers.trainer_callback import TrainerCallback
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from omegaconf import DictConfig
import datetime
import pytz
import os


def init_wandb(cfg: DictConfig):
    """
    WandBを初期化する

    Parameters
    ----------
    cfg : DictConfig
        エントリポイントでhydra.mainから取得する設定

    Returns
    -------
    None
    """
    wandb.login(key=os.environ["WANDB_API_KEY"])
    # 現在の日本時間を取得（日付と時間）
    date, time = datetime.datetime.now(pytz.timezone("Asia/Tokyo")).strftime("%Y%m%d/%H%M%S").split("/")
    current_time = f"{date}_{time}"
    # WandBの設定
    wandb.init(project="discourse_relation", name=cfg.experiment_type + ":" + current_time)


class WandbMetricsCallback(TrainerCallback):
    """
    WandBにメトリクスをログするコールバック

    Args:
        trainer (transformers.Trainer): トレーナ
        cfg (DictConfig): エントリポイントでhydra.mainから取得する設定
    """

    def __init__(self, trainer, cfg: DictConfig):
        self.trainer = trainer
        self.cfg: DictConfig = cfg

    def on_epoch_end(self, args, state, control, **kwargs):
        """
        エポック終了時の処理

        Parameters
        ----------
        args : transformers.TrainingArguments
            トレーニング設定
        state : transformers.TrainerState
            トレーナの状態
        control : transformers.TrainerControl
            トレーナのコントロール
        """
        # Training metrics
        train_output = self.trainer.predict(self.trainer.train_dataset)
        train_preds = np.argmax(train_output.predictions, axis=1)
        train_labels = train_output.label_ids
        train_metrics = self.compute_metrics(train_preds, train_labels)

        # Validation metrics
        val_output = self.trainer.predict(self.trainer.eval_dataset)
        val_preds = np.argmax(val_output.predictions, axis=1)
        val_labels = val_output.label_ids
        val_metrics = self.compute_metrics(val_preds, val_labels)

        # Retrieve losses from log history
        train_loss = None
        val_loss = None
        for log in self.trainer.state.log_history:
            if "loss" in log:
                train_loss = log["loss"]
            if "eval_loss" in log:
                val_loss = log["eval_loss"]

        # Log metrics to WandB
        wandb.log(
            {
                "epoch": state.epoch,
                "train/f1": train_metrics["f1"],
                "train/accuracy": train_metrics["accuracy"],
                "train/precision": train_metrics["precision"],
                "train/recall": train_metrics["recall"],
                "val/f1": val_metrics["f1"],
                "val/accuracy": val_metrics["accuracy"],
                "val/precision": val_metrics["precision"],
                "val/recall": val_metrics["recall"],
                "train/loss": train_loss,  # 訓練時の損失をログ
                "val/loss": val_loss,  # 検証時の損失をログ
            }
        )

    def compute_metrics(self, preds, labels):
        """
        メトリクスを計算する

        Parameters
        ----------
        preds : np.ndarray
            予測ラベル
        labels : np.ndarray
            真のラベル

        Returns
        -------
        dict
            メトリクスを格納したdict
        """
        return {
            "f1": f1_score(
                labels,
                preds,
                average=self.cfg.evaluation.metric_average,
                zero_division=np.nan
                if self.cfg.evaluation.zero_division == "np.nan"
                else self.cfg.evaluation.zero_division,
            ),
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(
                labels,
                preds,
                average=self.cfg.evaluation.metric_average,
                zero_division=np.nan
                if self.cfg.evaluation.zero_division == "np.nan"
                else self.cfg.evaluation.zero_division,
            ),
            "recall": recall_score(
                labels,
                preds,
                average=self.cfg.evaluation.metric_average,
                zero_division=np.nan
                if self.cfg.evaluation.zero_division == "np.nan"
                else self.cfg.evaluation.zero_division,
            ),
        }
