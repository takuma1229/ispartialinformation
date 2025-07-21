from fine_tuning import main as fine_tuning_main
import pandas as pd
from scipy.stats import norm
import numpy as np
from omegaconf import OmegaConf
import argparse
import os


def calculate_confidence_interval(scores: list[float]) -> dict:
    """
    結果のスコアから95%信頼区間を計算する。

    Args
    ----
    scores :list[float]
        スコアのリスト

    Returns
    -------
    ci :dict
        95%信頼区間を格納した辞書
    """
    mean = sum(scores) / len(scores)
    std = np.std(scores)
    standard_error = std / np.sqrt(len(scores))
    bottom, up = norm.interval(0.95, loc=mean, scale=standard_error)  # 小標本なのでt分布を使う。よってinterval
    return {"mean": mean, "bottom": bottom, "up": up}


def get_ci(df_list: list[pd.DataFrame], data_type: str) -> pd.DataFrame:
    """
    複数の実験結果から各metricsの95%信頼区間を計算する。

    Args
    ----
    df_list (list[pd.DataFrame])
        複数の実験結果を格納したDataFrameのリスト
    data_type: st
        全ラベルに対するmetricsか、ラベルごとのmetricsかを指定する。
        "all_label" or "by_label"

    Returns
    -------
    df (pd.DataFrame)
        各metricsの95%信頼区間を計算したDataFrame
    """
    if data_type == "all_label":
        metrics = ["test/acc", "test/precision", "test/recall", "test/f1"]
        ci_list = []
        for metric in metrics:
            scores = [df[metric] for df in df_list]
            ci = calculate_confidence_interval(scores)
            ci_list.append(ci)
        return pd.DataFrame(ci_list, index=metrics)
    elif data_type == "by_label":
        metrics = ["test/acc", "test/precision", "test/recall", "test/f1"]
        all_labels = set()

        for df in df_list:
            all_labels.update(df["label"].dropna().unique())

        results = []

        for label in all_labels:
            for metric in metrics:
                scores = []
                for df in df_list:
                    label_data = df[df["label"] == label]
                    scores.extend(label_data[metric].values)

                ci = calculate_confidence_interval(scores)

                results.append(
                    {
                        "label": label,
                        "metric": metric,
                        "mean": ci["mean"],
                        "ci_lower": ci["bottom"],
                        "ci_upper": ci["up"],
                    }
                )

        result_df = pd.DataFrame(results)
        # result_df = result_df.pivot(index="label", columns="metric", values=["mean", "ci_lower", "ci_upper"])

        return result_df


def main() -> None:
    """
    複数回の実験を行って、95%信頼区間を計算し、記録する。

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_type", type=str)
    parser.add_argument(
        "--model_name",
        type=str,
    )
    parser.add_argument(
        "--is_debug",
        action="store_true",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load("./conf/config.yaml")

    # コマンドライン引数を辞書に変換し、設定を上書き
    cli_overrides = {k: v for k, v in vars(args).items() if v is not None}
    cfg = OmegaConf.merge(cfg, cli_overrides)

    print(cfg)

    all_label_metrics_df_list = []
    by_label_metrics_df_list = []

    for _ in range(10):
        all_label_metrics, by_label_metrics = fine_tuning_main(cfg)
        all_label_metrics_df_list.append(all_label_metrics)
        by_label_metrics_df_list.append(by_label_metrics)

    all_label_metrics_df = pd.concat(all_label_metrics_df_list, axis=0)
    by_label_metrics_df = pd.concat(by_label_metrics_df_list, axis=0)
    all_label_ci = get_ci(all_label_metrics_df_list, data_type="all_label")
    by_label_ci = get_ci(by_label_metrics_df_list, data_type="by_label")

    print(all_label_ci)

    # 保存するディレクトリのパスを取得

    output_dir = (
        f"results/fine_tuning_scores/{cfg.model_name.replace("/", "_")}/multiple_experiments/{cfg.experiment_type}/"
    )
    os.makedirs(output_dir, exist_ok=True)  # ディレクトリが存在しない場合は作成

    all_label_metrics_df.to_csv(os.path.join(output_dir, "all_label_metrics.csv"))
    all_label_ci.to_csv(os.path.join(output_dir, "all_label_ci.csv"))
    by_label_metrics_df.to_csv(os.path.join(output_dir, "by_label_metrics.csv"))
    by_label_ci.to_csv(os.path.join(output_dir, "by_label_ci.csv"))


if __name__ == "__main__":
    main()
