import matplotlib.pyplot as plt
import pandas as pd


def make_df(metrics_path: str, operation_count_path: str) -> pd.DataFrame:
    """
    metricsとoperation_countのcsvファイルを読み込み、experiment_typeでピボットしたDataFrameを返す

    Args
    ----
    metrics_path (str): metricsのcsvファイルのパス
    operation_count_path (str): operation_countのcsvファイルのパス

    Returns
    -------
    df (pd.DataFrame): metricsとoperation_countを結合したDataFrame
    """
    metrics_df = pd.read_csv(metrics_path)
    operation_count_df = pd.read_csv(operation_count_path)

    metrics_df = metrics_df.rename(columns={"experiment type": "experiment_type"})
    metrics_df = metrics_df[::-1]  # rowを反転

    operation_count_df.rename(columns={"Experiment_type": "experiment_type", "Value": "operation_count"}, inplace=True)
    operation_count_df = operation_count_df[::-1]  # rowを反転

    df = metrics_df.merge(operation_count_df, on="experiment_type", how="left")

    return df


def calculate_metric_decline(df, normal_metrics) -> pd.DataFrame:
    """
    各experiment_typeにおけるnormalとのmetricsの差分を計算し、DataFrameに追加する

    Args
    ----
    df (pd.DataFrame): metricsとoperation_countを結合したDataFrame
    normal_metrics (pd.Series): normalのmetrics

    Returns
    -------
    df (pd.DataFrame): metricsの差分を追加したDataFrame
    """
    metric_columns = [col for col in df.columns if col.startswith("mean")]
    for metric in metric_columns:
        decline_col = f"{metric}_decline"
        per_operation_col = f"{metric}_decline_per_operation"
        df[decline_col] = normal_metrics[metric] - df[metric]
        df[per_operation_col] = df[decline_col] / (df["operation_count"] / 2)
    return df


def plot_metric_decline(df, is_concession: bool):
    """
    experiment_typeごとのmetricsの差分をplotする

    Args
    ----
    df (pd.DataFrame): metricsの差分を追加したDataFrame

    Returns
    -------
    Nones
    """
    metric_columns = [col for col in df.columns if col.endswith("_decline_per_operation")]
    for metric in metric_columns:
        plt.figure(figsize=(10, 6))
        color_map = {
            "normal": "#1F77B4",
            "ordinal_forms_shuffle": "#AEC7E8",
            "latter-only": "#FF7F0F",
            "former-only": "#FFBB77",
            # "exclude_koso",
            "exclude_connectives": "#2AA02B",
            "exclude_content_words": "#98DF8A",
            "exclude_function_words": "#D62727",
            "exclude_mo": "#FF9896",
            # "exclude_particle",
            # "exclude_ha_and_ga",
            "exclude_negation": "#9367BD",
            # "exclude_iru_aru_oku",
            "convert_content_words_to_dummy": "#C5B0D4",
            "convert_function_words_to_dummy": "#8B554B",
            "convert_all_words_to_dummy": "#C49C94",
            # "convert_connectives_to_dummy",
        }
        # colors = plt.cm.tab20(range(len(df)))  # カラーマップを使用して色を設定

        df_to_plot = df[
            ~df["experiment_type"].isin(
                [
                    "normal",
                    "exclude_iru_aru_oku",
                    "exclude_ha_and_ga",
                    "exclude_koso",
                    "exclude_particle",
                    "convert_connectives_to_dummy",
                ]
            )
        ]  # "normal" などを除外
        print(df_to_plot)
        df_to_plot = df_to_plot.sort_values(by=metric, ascending=False)  # 値で降順にソート
        colors = df_to_plot["experiment_type"].map(color_map)  # experiment_type に基づいて対応する色を取得
        plt.bar(df_to_plot["experiment_type"], df_to_plot[metric], alpha=0.7, color=colors)
        plt.title(f"{metric} per operation count")
        plt.xlabel("Experiment Type")
        plt.ylabel(f"{metric}")
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yscale("log")  # y軸を対数スケールに
        plt.tight_layout()
        plt.show()
        if is_concession:
            save_path: str = f"results/fine_tuning_scores/figures/degrade/concession/{metric.replace('/', '_')}.png"
        else:
            save_path: str = f"results/fine_tuning_scores/figures/degrade/all_label/{metric.replace('/', '_')}.png"
        plt.savefig(save_path)
        print(f"{save_path} に画像を保存しました。")


def main(metric_path: str):
    """
    normalに対するmetric_declineを計算し、plotする

    - 逆接に対するmetricなのか、全ラベルに対するmetricなのかは、metric_pathによって判断する

    Args
    ----
    metric_path (str): metricsのcsvファイルのパス

    Returns
    -------
    None
    """
    is_concession: bool = True if "concession" in metric_path else False
    metrics_path = metric_path
    operation_count_path = "./data/statistics/operation_count.csv"
    df = make_df(metrics_path, operation_count_path)

    normal_metrics = df[df["experiment_type"] == "normal"].iloc[0]
    df_with_declines = calculate_metric_decline(df, normal_metrics).dropna()
    df_with_declines = df_with_declines[
        ~df_with_declines.isin([float("inf"), float("nan")]).any(axis=1)
    ]  # infやNaNが含まれる行は除外
    print(df_with_declines)

    plot_metric_decline(df_with_declines, is_concession)


if __name__ == "__main__":
    main("results/fine_tuning_scores/multiple_experiments/multiple_exps_all_label_ci.csv")
    main("results/fine_tuning_scores/multiple_experiments/multiple_exps_concession_ci.csv")
