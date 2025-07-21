from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import wandb
import numpy as np
import os
from log import WandbMetricsCallback, init_wandb
from process_data import create_dataset
from utils import load_label_mapping
from evaluation import compute_label_metrics
from omegaconf import DictConfig
from collections import defaultdict
from omegaconf import OmegaConf


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

NUM_LABELS: int = 5
# cfg.evaluation.metric_average: Literal["macro", "micro", "weighted"] = "weighted"
LABEL_MAPPING_PATH: str = "./data/id_to_label.json"


# トークナイザーとモデルのロード
def load_model_and_tokenizer(model_name: str, num_labels: int):
    """
    モデルとトークナイザをロードする

    Parameters
    ----------
    model_name : str
        モデル名
    num_labels : int
        ラベル数

    Returns
    -------
    transformers.AutoTokenizer, transformers.AutoModelForSequenceClassification
        トークナイザとモデル
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, word_tokenizer_type="mecab")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return tokenizer, model


# tokenizerへのspecial_tokenの追加
def add_special_tokens_to_tokenizer(tokenizer, model):
    """
    トークナイザにSpecial Tokenを追加する

    Parameters
    ----------
    tokenizer : transformers.AutoTokenizer
        トークナイザ
    model : transformers.AutoModelForSequenceClassification
        モデル

    Returns
    -------
    transformers.AutoTokenizer
        トークナイザ
    """
    with open("./data/pos_data/pos_dict.json", "r") as f:
        POS_DICT = json.load(f)
    additional_special_tokens = []
    for _, v in POS_DICT.items():
        additional_special_tokens.append(v)
    num_added_tokens = tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
    additional_special_tokens = tokenizer.additional_special_tokens
    # ==============↓これしないとモデルとトークナイザの語彙サイズ不整合でバグる！
    model.resize_token_embeddings(len(tokenizer))
    print(
        "num_added_tokens: {}, added_special_tokens: {}".format(num_added_tokens, ",".join(additional_special_tokens))
    )
    return tokenizer


def my_compute_metrics(cfg) -> callable:
    """
    Trainerクラスのcompute_metrics属性に、クラス内でシグネイチャの形式が指定されているような関数を渡しつつ、その関数の定義内でcfgに参照するために、cfgを参照したうえで定義される関数を返す関数。cfgがグローバルな状態で扱えるならこれは必要ないのだが、hydraを使う場合cfgはあくまでlocalなので、このような関数を定義する必要がある。

    Parameters
    ----------
    cfg : DictConfig
        エントリポイントでhydra.mainから取得する設定

    Returns
    -------
    callable
        compute_metrics属性に渡す関数
    """

    def compute_metrics(pred):
        """
        メトリクスを計算する

        Parameters
        ----------
        pred : transformers.trainer_utils.EvalPrediction
            予測結果


        Returns
        -------
        dict
            各種メトリクスを格納したdict
        """
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        accuracy = accuracy_score(labels, preds)
        precision = precision_score(
            labels,
            preds,
            average=cfg.evaluation.metric_average,
            zero_division=np.nan if cfg.evaluation.zero_division == "np.nan" else cfg.evaluation.zero_division,
        )
        recall = recall_score(
            labels,
            preds,
            average=cfg.evaluation.metric_average,
            zero_division=np.nan if cfg.evaluation.zero_division == "np.nan" else cfg.evaluation.zero_division,
        )
        f1 = f1_score(labels, preds, average=cfg.evaluation.metric_average)
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return compute_metrics


def define_trainer(
    model,
    args: TrainingArguments,
    train_dataset,
    valid_dataset,
    tokenizer,
    compute_metrics,
    cfg: DictConfig,
) -> Trainer:
    """
    トレーナを定義する

    Parameters
    ----------
    model : transformers.AutoModelForSequenceClassification
        モデル
    args : transformers.TrainingArguments
        トレーニング設定
    train_dataset : CustomDataset
        訓練データセット
    valid_dataset : CustomDataset
        検証データセット
    tokenizer : transformers.AutoTokenizer
        トークナイザ
    compute_metrics : Callable
        メトリクス計算用の関数
    """
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    wandb_metrics_callback = WandbMetricsCallback(trainer, cfg)
    trainer.add_callback(wandb_metrics_callback)

    # EarlyStoppingCallbackを追加
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=cfg.training.early_stopping_patience,  # 例えば3エポック
        early_stopping_threshold=cfg.training.early_stopping_threshold,  # 例えば0.001のスコア差
    )
    trainer.add_callback(early_stopping_callback)
    return trainer


def train(trainer: Trainer) -> Trainer:
    """
    モデルのトレーニングを実行する

    Parameters
    ----------
    trainer : Trainer
        トレーナ

    Returns
    -------
    Trainer
        トレーナ
    """
    trainer.train()
    # モデルの評価
    results = trainer.evaluate()
    print(results)
    return trainer


def test(trainer: Trainer, test_dataset, cfg: DictConfig):
    """
    テストデータに対する評価を実行する

    Parameters
    ----------
    trainer : Trainer
        トレーナ
    test_dataset : CustomDataset
        テストデータセット
    cfg : DictConfig
        エントリポイントでhydra.mainから取得する設定


    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        テストデータに対するメトリクス。全ラベル対象のmetricsと、ラベルごとのmetrics
    """
    # テストデータに対して予測を実行
    test_results = trainer.predict(test_dataset)
    predictions = test_results.predictions.argmax(-1)
    true_labels = test_results.label_ids

    # ラベルマッピングの読み込み
    label_mapping_path = LABEL_MAPPING_PATH  # ダミーパス
    label_mapping = load_label_mapping(label_mapping_path)

    # メトリクスの計算
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(
        true_labels,
        predictions,
        average=cfg.evaluation.metric_average,
        zero_division=np.nan if cfg.evaluation.zero_division == "np.nan" else cfg.evaluation.zero_division,
    )
    recall = recall_score(
        true_labels,
        predictions,
        average=cfg.evaluation.metric_average,
        zero_division=np.nan if cfg.evaluation.zero_division == "np.nan" else cfg.evaluation.zero_division,
    )
    f1 = f1_score(
        true_labels,
        predictions,
        average=cfg.evaluation.metric_average,
        zero_division=np.nan if cfg.evaluation.zero_division == "np.nan" else cfg.evaluation.zero_division,
    )

    print("Test Results:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    label_metrics = compute_label_metrics(true_labels, predictions, cfg)  # ラベルごとのメトリクスを計算

    # 各ラベル別のスコアをログ
    columns = ["cfg.experiment_type", "label", "test/precision", "test/recall", "test/f1", "test/acc"]
    label_metrics_data = []
    for label, metrics in label_metrics.items():
        label_text = label_mapping[label]
        label_metrics_data.append(
            [cfg.experiment_type, label_text, metrics["precision"], metrics["recall"], metrics["f1"], metrics["f1"]]
        )
    label_metrics_table = wandb.Table(data=label_metrics_data, columns=columns)
    wandb.log({"label_metrics": label_metrics_table})

    # テストデータの予測結果をログ
    test_predictions_columns = [
        "experiment type",
        "id",
        "input_text",
        "connective",
        "true_label",
        "true_label_text",
        "predicted_label",
        "predicted_label_text",
    ]
    test_predictions_data = []
    for _, (id, text, connective, true_label, predicted_label) in enumerate(
        zip(
            [d["id"] for d in test_dataset.data],
            [d["text"] for d in test_dataset],
            [d["接続表現"] for d in test_dataset.data],
            true_labels,
            predictions,
            strict=False,
        )
    ):
        # input_text = tokenizer.decode(input_ids, skip_special_tokens=False)
        true_label_text = label_mapping[true_label.item()]
        predicted_label_text = label_mapping[predicted_label]

        assert len(cfg.experiment_type) > 2  # debug

        test_predictions_data.append(
            [
                cfg.experiment_type,
                id,
                text,
                connective,
                true_label,
                true_label_text,
                predicted_label,
                predicted_label_text,
            ]
        )
    print(f"test_predictions_data: {test_predictions_data}")
    test_predictions_table = wandb.Table(data=test_predictions_data, columns=test_predictions_columns)
    print(f"test_predictions_table: {test_predictions_table.get_dataframe()}")
    wandb.log({"test_predictions": test_predictions_table})

    columns = [
        "experiment_type",
        "test/acc",
        "test/precision",
        "test/recall",
        "test/f1",
    ]
    metrics_data = [cfg.experiment_type, accuracy, precision, recall, f1]
    table = wandb.Table(data=[metrics_data], columns=columns)
    wandb.log({"test-metrics": table})

    # connective ごとのメトリクスを計算

    connective_metrics = defaultdict(
        lambda: {"y_true": [], "y_pred": []}
    )  # connective ごとに y_true, y_pred を収集するための辞書

    # test_predictions_data から connective ごとにラベルを収集
    for row in test_predictions_data:
        _, _, _, connective, true_label, _, predicted_label, _ = row
        connective_metrics[connective]["y_true"].append(true_label)
        connective_metrics[connective]["y_pred"].append(predicted_label)

    # connective ごとの集計結果を WandB にまとめて記録するための準備
    connective_table = wandb.Table(columns=["experiment_type", "connective", "accuracy", "precision", "recall", "f1"])

    for conn, labels_dict in connective_metrics.items():
        y_true = labels_dict["y_true"]
        y_pred = labels_dict["y_pred"]

        # マルチクラスかバイナリかに応じて average パラメータを調整する
        # 必要に応じて average="macro"/"micro"/"weighted" を使い分ける
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(
            y_true,
            y_pred,
            average=cfg.evaluation.metric_average,
            zero_division=np.nan if cfg.evaluation.zero_division == "np.nan" else cfg.evaluation.zero_division,
        )
        recall = recall_score(
            y_true,
            y_pred,
            average=cfg.evaluation.metric_average,
            zero_division=np.nan if cfg.evaluation.zero_division == "np.nan" else cfg.evaluation.zero_division,
        )
        f1 = f1_score(
            y_true,
            y_pred,
            average=cfg.evaluation.metric_average,
            zero_division=np.nan if cfg.evaluation.zero_division == "np.nan" else cfg.evaluation.zero_division,
        )

        connective_table.add_data(cfg.experiment_type, conn, accuracy, precision, recall, f1)

    print(f"connective_table: {connective_table.get_dataframe()}")

    # connective ごとのメトリクスをログ
    wandb.log({"test_connective_metrics": connective_table})

    # exclude_countをログ
    exclude_count_table = wandb.Table(columns=["Experiment_type", "Value"])
    exclude_count_table.add_data(cfg.experiment_type, test_dataset.exclude_count)
    wandb.log({"Exclude_counts": exclude_count_table})

    print(f"test_dataset.exclude_count in {cfg.experiment_type}: {test_dataset.exclude_count}")

    # negation_position_dictをログ
    print(f"test_dataset.negation_position_dict: {test_dataset.negation_position_dict}")
    wandb.log({"negation_position_dict": test_dataset.negation_position_dict})

    return (table.get_dataframe(), label_metrics_table.get_dataframe())


# @hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    fine-tuningを行う。

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        テストデータに対するメトリクス。全ラベル対象のmetricsと、ラベルごとのmetrics
    """
    # WandBの初期化
    init_wandb(cfg)

    tokenizer, model = load_model_and_tokenizer(cfg.model_name, cfg.num_labels)
    tokenizer = add_special_tokens_to_tokenizer(tokenizer, model)
    # tokenizerのテスト
    assert tokenizer.tokenize("<INTERJECTION><AUXILIARY-SYMBOL><INTERJECTION><NOUN>") == [
        "<INTERJECTION>",
        "<AUXILIARY-SYMBOL>",
        "<INTERJECTION>",
        "<NOUN>",
    ]

    train_data = create_dataset(
        "./data/pos_data/train.json",
        tokenizer,
        max_len=512,
        cfg=cfg,
    )
    valid_data = create_dataset(
        "./data/pos_data/valid.json",
        tokenizer,
        max_len=512,
        cfg=cfg,
    )
    test_data = create_dataset(
        "./data/pos_data/test.json",
        tokenizer,
        max_len=512,
        cfg=cfg,
    )

    for i in range(5):
        input_ids = train_data[i]["input_ids"]
        input_text = tokenizer.decode(input_ids, skip_special_tokens=False)
        print("input_text: {}".format(input_text))

    # train_dataloader = create_dataloader(train_data, batch_size=16)
    # valid_dataloader = create_dataloader(valid_data, batch_size=16)
    # test_dataloader = create_dataloader(test_data, batch_size=16)

    # トレーニング設定
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=cfg.training.epochs,  # early_stoppingがかからない場合の最大epoch数
        per_device_train_batch_size=cfg.training.batch_size,
        per_device_eval_batch_size=cfg.training.batch_size,
        warmup_steps=500,
        weight_decay=cfg.training.weight_decay,
        logging_dir="./logs",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        report_to="wandb",  # WandBにログを報告する
        # lr_scheduler_type : 学習率のスケジューラの種類
        # (linearはデフォルトだが、学習率の制御をここで行っていることを明示するために一応指定)
        lr_scheduler_type="linear",
    )

    trainer = define_trainer(model, training_args, train_data, valid_data, tokenizer, my_compute_metrics(cfg), cfg)
    trained_trainer = train(trainer)
    all_label_metrics, by_label_metrics = test(trained_trainer, test_dataset=test_data, cfg=cfg)

    # WandBのランを終了
    wandb.finish()

    return all_label_metrics, by_label_metrics


if __name__ == "__main__":
    # グローバル設定を取得
    cfg = OmegaConf.load("./conf/config.yaml")
    main()
    # for cfg.experiment_type in [
    #     "normal",
    #     "ordinal_forms",
    #     "ordinal_forms_shuffle",
    #     "pos_sequence",
    #     "former-only",
    #     "latter-only",
    # ]:
    #     # for cfg.experiment_type in
    #     # ["ordinal_forms", "ordinal_forms_shuffle", "pos_sequence", 'former-only', 'latter-only']:
    #     cfg.experiment_type = cfg.experiment_type
    #     main()
