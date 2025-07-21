# デコレータを定義するファイル
import time
from functools import wraps
import json


def load_label_mapping(label_mapping_path):
    """
    ラベルと番号の対応を読み込む

    Parameters
    ----------
    label_mapping_path : str
        ラベルと番号の対応を記述したJSONファイルのパス
    """
    with open(label_mapping_path, "r", encoding="utf-8") as f:
        label_mapping = json.load(f)
    return {int(k): v for k, v in label_mapping.items()}


def print_function_info(func):
    """
    関数名と引数を表示するデコレータ。別にライブラリとかもあるが、デコレータの練習として自作。

    Parameters
    ----------
    func : function
        デコレートする関数

    Returns
    -------
    wrapper : function
        デコレートされた関数
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Function: {func.__name__}")
        print(f"Args: {args}")
        print(f"Kwargs: {kwargs}")
        return func(*args, **kwargs)

    return wrapper


def execution_speed(func):
    """
    関数の実行時間を表示するデコレータ

    Parameters
    ----------
    func : function
        デコレートする関数

    Returns
    -------
    wrapper : function
        デコレートされた関数
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print("実行時間 : " + str(run_time) + "秒")

    return wrapper
