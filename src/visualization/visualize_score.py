import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import json


def visualize_metrics_for_experiments(result_path: str):
    """
    fine-tuningの結果を棒グラフとしてplotする

    - 各experimet_typeと各metricsに対する結果を格納したcsvファイルを読み込み、棒グラフをpngとして保存する。
    - 各metricsに対する結果は、"results/fine_tuning_scores/from_wandb.csv"のような形式の、
        wandbからDLするようなcsvを想定する。
    - 結果は、"results/fine_tuning_scores/figures"に保存される。
    - concessionとall_labelに分けて可視化する。result_pathに"concession"が含まれるかで判断する。

    Args:
        result_path (str): 結果のcsvファイルのパス

    Returns
    -------
        None
    """
    is_concession: bool = "concession" in result_path
    df = pd.read_csv(result_path)
    df = df.iloc[::-1]  # 順番を逆にする

    # メトリクスごとにグラフを作成
    metrics = ["test/acc", "test/precision", "test/recall", "test/f1"]
    for metric in metrics:
        plt.figure(figsize=(10, 8))  # 縦幅を拡大
        bars = plt.bar(df["experiment_type"], df[metric], color=plt.cm.tab20.colors)
        plt.title(f"Comparison of {metric}")
        plt.ylabel(metric)
        plt.xticks(rotation=90)
        plt.ylim(0, 1.1)

        # 棒グラフ上に値を表示
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.2f}", ha="center", va="bottom")

        plt.tight_layout()
        plt.show()
        if is_concession:
            save_path = f"results/fine_tuning_scores/figures/metrics/concession/{metric.replace('/', '_')}.png"
        else:
            save_path = f"results/fine_tuning_scores/figures/metrics/all_label/{metric.replace('/', '_')}.png"
        plt.savefig(save_path)


def format_multiple_experiment_data(metric_type: str) -> pd.DataFrame:
    """
    multiple_experimentsの結果を整形して、ciやmeanを一つのファイルにまとめる

    - まとめた結果は、"results/fine_tuning_scores/multiple_experiments/{metric_type}_ci.csv"に保存される。
    - by_labelの場合は、「逆接・譲歩」に対する結果をまとめる。

    Args
    ----
    metric_type: str
        "all_label" or "by_label"を指定する

    Returns
    -------
    result_df: pd.DataFrame
        整形したデータを格納したDataFrame。
    """
    file_pattern = (
        f"./results/fine_tuning_scores/tohoku-nlp_bert-base-japanese-v3/multiple_experiments/*/{metric_type}_ci.csv"
    )

    # CSVファイルをすべて取得
    csv_files = glob.glob(file_pattern)
    # * 部分 (experiment_type) を抽出するリストを作成
    experiment_types = [os.path.basename(os.path.dirname(path)) for path in csv_files]

    if metric_type == "all_label":
        dataframes = []

        for file in zip(csv_files, experiment_types, strict=False):
            df = pd.read_csv(file[0])
            print(f"df[0:1]: {df.iloc[0]['mean']}")

            f1_data = df[df.iloc[:, 0] == "test/f1"]  # 1列目がtest/f1の行を取得

            f1_data = f1_data[["mean", "bottom", "up"]]

            # 各カラムのデータを整形
            f1_data["experiment_type"] = file[1]
            f1_data["mean"] = float(
                str(f1_data["mean"]).split()[2].replace("\\nName:", "")
            )  # 最悪な処理なのだが、mutliple_experimentの記録コードがおかしいため。もう一回回すと1日くらいかかる.....。
            f1_data["bottom"] = f1_data["bottom"].str.extract(r"([0-9\.]+)").astype(float)
            f1_data["up"] = f1_data["up"].str.extract(r"([0-9\.]+)").astype(float)

            dataframes.append(f1_data)

        result_df = pd.concat(dataframes, ignore_index=True)
        result_df = result_df.reindex(columns=["experiment_type", "mean", "bottom", "up"])
        result_df.to_csv(
            "results/fine_tuning_scores/tohoku-nlp_bert-base-japanese-v3/multiple_experiments/multiple_exps_all_label_ci.csv",
            index=False,
        )
    elif metric_type == "by_label":
        dataframes = []
        for file in zip(csv_files, experiment_types, strict=False):
            df = pd.read_csv(file[0])
            f1_data = df[df.iloc[:, 1] == "逆接・譲歩"]  # 1列目が逆接・譲歩の行を取得
            f1_data = f1_data[df.iloc[:, 2] == "test/f1"]  # 2列目が逆接・譲歩の行を取得
            f1_data = f1_data[["mean", "ci_lower", "ci_upper"]]
            f1_data["experiment_type"] = file[1]
            dataframes.append(f1_data)

        result_df = pd.concat(dataframes, ignore_index=True)
        result_df = result_df.reindex(columns=["experiment_type", "mean", "ci_lower", "ci_upper"])
        result_df = result_df.rename(columns={"ci_lower": "bottom", "ci_upper": "up"})
        result_df.to_csv(
            "results/fine_tuning_scores/tohoku-nlp_bert-base-japanese-v3/multiple_experiments/multiple_exps_concession_ci.csv",
            index=False,
        )

    return result_df


def visualize_metrics_with_ci(result_path: str):
    """
    multiple_experimentsの結果を棒グラフとしてplotする

    - 各experimet_typeと各metricsに対する結果を格納したcsvファイルを読み込み、棒グラフをpngとして保存する。

    Args:
        result_path (str): 結果のcsvファイルのパス

    Returns
    -------
        None
    """
    file_name = os.path.basename(result_path)
    df = pd.read_csv(result_path)

    # experiment_typeの順番を指定
    order = [
        "normal",
        "ordinal_forms_shuffle",
        "latter-only",
        "former-only",
        # "exclude_koso",
        "exclude_connectives",
        "exclude_content_words",
        "exclude_function_words",
        "exclude_mo",
        # "exclude_particle",
        # "exclude_ha_and_ga",
        "exclude_negation",
        # "exclude_iru_aru_oku",
        "convert_content_words_to_dummy",
        "convert_function_words_to_dummy",
        "convert_all_words_to_dummy",
        # "convert_connectives_to_dummy",
    ]
    df = df[df["experiment_type"].isin(order)]  # orderに従ってdfのカラムを絞り込む
    with open("./data/column_rename.json") as f:
        column_rename_dict: dict = json.load(f)
    renamed_order = [column_rename_dict[key] for key in order]  # 元のカラム名の順番通りにカラム系を論文用に変換したlist
    assert len(order) == len(df["experiment_type"].unique()), "experiment_typeの数が並び替えorderの数と一致しません。"
    print(f"renamed_order: {renamed_order}")
    df["experiment_type"] = pd.Categorical(df["experiment_type"], categories=order, ordered=True)
    df = df.sort_values("experiment_type")  # 指定順に並べ替え
    df["experiment_type"] = renamed_order  # カラム名を論文用に変換

    plt.figure(figsize=(10, 7))

    colors = plt.cm.tab20(range(len(df)))  # カラーマップを使用して色を設定

    bars = plt.bar(
        df["experiment_type"],
        df["mean"],
        yerr=[df["mean"] - df["bottom"], df["up"] - df["mean"]],
        capsize=5,
        color=colors,
    )

    # 各棒グラフの上に値を表示（エラーバーと被らないように位置を調整）
    for bar, mean, up in zip(bars, df["mean"], df["up"], strict=False):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            up + 0.01,  # エラーバーの上に表示
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=15,
        )

    label = "all_label" if "all_label" in result_path else "concession"
    value_name = "F1-score" if label == "all_label" else "Accuracy"
    plt.xlabel("Experiment Type")
    plt.ylabel(f"Mean {value_name} (with 95% CI)")
    plt.title(f"Mean {value_name} with Confidence Intervals by Experiment Type for {label}")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 0.9)  # グラフのy軸の範囲を指定 (最大値に応じてよしなに切らず自分で指定する)
    plt.tight_layout()

    # グラフの表示
    plt.show()
    plt.savefig(
        f"results/fine_tuning_scores/tohoku-nlp_bert-base-japanese-v3/figures/metrics/{file_name.replace('.csv', '.png')}"
    )


if __name__ == "__main__":
    # visualize_metrics_for_experiments("results/fine_tuning_scores/scores_from_wandb.csv")
    # visualize_metrics_for_experiments("results/fine_tuning_scores/scores_for_concession_from_wandb.csv")
    format_multiple_experiment_data("all_label")
    format_multiple_experiment_data("by_label")
    visualize_metrics_with_ci(
        "results/fine_tuning_scores/tohoku-nlp_bert-base-japanese-v3/multiple_experiments/multiple_exps_all_label_ci.csv"
    )
    visualize_metrics_with_ci(
        "results/fine_tuning_scores/tohoku-nlp_bert-base-japanese-v3/multiple_experiments/multiple_exps_concession_ci.csv"
    )
