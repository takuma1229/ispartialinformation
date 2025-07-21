import pandas as pd
from bs4 import BeautifulSoup
import requests
import time
from collections import defaultdict
import MeCab
import json
import re


# 文と、文に対する各種情報を保持するクラス
class TextWithInfo:
    """
    文と、文に対する各種情報を保持するクラス。

    Attributes
    ----------
    text : str
        文。
    infos : list
        文に対する形態素解析を行った情報。
    wakati_list : list
        分かち書きされたテキストのリスト。
    pos_sequence : list
        品詞のリスト。
    pos_frequency : str
        品詞の出現頻度。
    ordinal_forms : list
        原型のリスト。
    """

    def __init__(self, text):
        self.pos_dict = json.load(
            open("./data/pos_data/pos_dict.json", "r")
        )  # atrributeの中で一番最初に読み込む必要があることに注意
        self.text = self.touten_to_upper_letter(text)
        self.infos = self.get_infos()
        self.wakati_list = self.text_to_wakati_list()
        self.pos_sequence = self.text_to_pos_sequence()
        self.pos_frequency = self.text_to_pos_frequency()
        self.ordinal_forms = self.get_ordinal_forms()

    def touten_to_upper_letter(self, text) -> str:
        """
        テキスト中の半角カンマを全角カンマに変換するメソッド。テキスト中に半角カンマ(,)が入っていると、mecabの出力処理のカンマとの間でごっちゃになるので、全角にする

        Parameters
        ----------
        text : str
            テキスト。

        Returns
        -------
        str
            半角カンマを全角カンマにしたテキスト
        """
        return text.replace(",", "，")

    def get_infos(self) -> list:
        """
        テキストに対して形態素解析を行い、その情報をリストにして返すメソッド

        Returns
        -------
        list
            形態素解析の情報
        """
        tagger = MeCab.Tagger()
        word_info_raw = tagger.parse(self.text)
        infos_list = []
        for infos in word_info_raw.split("\n"):
            info_list = re.split("\t|\\s|,", infos)
            infos_list.append(info_list)
        return infos_list

    def text_to_wakati_list(self) -> list[str]:
        """
        テキストを分かち書きしてlistにするメソッド

        Returns
        -------
        list
            分かち書きされたテキストのリスト
        """
        wakati_list = []
        for info in self.infos:
            if info[0] == "EOS":
                return wakati_list
            else:
                wakati_list.append(info[0])

    def text_to_pos_sequence(self) -> list[str]:
        """
        テキストを品詞のリストにするメソッド

        Returns
        -------
        list
            品詞のリスト
        """
        pos_sequence = []
        for info in self.infos:
            if info[0] == "EOS":
                return pos_sequence
            else:
                pos_sequence.append(self.pos_dict[info[1]])

        return pos_sequence

    def text_to_pos_frequency(self) -> str:
        """
        テキストの品詞の出現頻度を取得するメソッド

        Returns
        -------
        str
            品詞の出現頻度
        """
        pos_count_dict = defaultdict(int)
        for pos in self.pos_sequence:
            pos_count_dict[pos] += 1
        # valueの値で降順ソート（items()を使っているので返り値はtupleのlist）
        pos_count_list = sorted(pos_count_dict.items(), key=lambda x: x[1], reverse=True)
        # dictに戻す
        pos_count_dict.clear()
        pos_count_dict.update(pos_count_list)

        pos_frequency_str = ""
        for pos in pos_count_dict:
            pos_frequency_str += f"{pos}:{pos_count_dict[pos]} "
        return pos_frequency_str

    def get_ordinal_forms(self) -> list[str]:
        """
        テキストの原型のリストを取得するメソッド

        Returns
        -------
        list
            原型のリスト
        """
        ordinal_forms = []
        for info in self.infos:
            if len(info) < 2:
                continue
            if info[1] in ["動詞", "形容詞", "助動詞", "形状詞"]:
                # 活用がある品詞の場合、原型を追加
                ordinal_forms.append(info[8])
            else:
                # そうでない場合、単に表層形を追加すればよい
                ordinal_forms.append(info[0])
        return ordinal_forms


def process_loaded_data(data):
    """
    データに対して、各種情報を追加する関数。

    Parameters
    ----------
    data : list
        ロードしたjsonのデータ

    Returns
    -------
    list
        データに各種情報を追加したデータ
    """
    for item in data:
        sentence = item["前件"] + item["接続表現"] + item["後件"]
        text_with_info = TextWithInfo(sentence)
        item["wakati"] = text_with_info.wakati_list
        item["pos_sequence"] = text_with_info.pos_sequence
        item["pos_frequency"] = text_with_info.pos_frequency
        item["ordinal_forms"] = text_with_info.ordinal_forms
    return data


def data_to_json(data, phase: str):
    """
    データをjson形式で保存する関数。

    Parameters
    ----------
    data : list
        データ。
    phase : str
        train, valid, testなどのフェーズ。

    Returns
    -------
    None
    """
    with open(f"./data/pos_data/{phase}.json", mode="wt", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def make_pos_data():
    """
    データを形態素解析して、各種情報を追加して保存する関数。

    Returns
    -------
    None
    """
    with open("./data/train.json") as f:
        train = json.load(f)
    with open("./data/valid.json") as f:
        valid = json.load(f)
    with open("./data/test.json") as f:
        test = json.load(f)

    train_data = process_loaded_data(train)
    data_to_json(train_data, "train")
    val_data = process_loaded_data(valid)
    data_to_json(val_data, "valid")
    test_data = process_loaded_data(test)
    data_to_json(test_data, "test")


def get_tree(df: pd.DataFrame):
    """
    各データに対して、idを用いてWebからtreeを取得し、データに追加する関数。

    Parameters
    ----------
    df : pd.DataFrame
        データフレーム。

    Returns
    -------
    pd.DataFrame
        treeを追加したデータフレーム。
    """
    for index, row in df.iterrows():
        id: str = row["id"]
        print(f"ID: {id}")
        # source: str = fetch_url_source(
        #     f"https://oncoj.orinst.ox.ac.uk/cgi-bin/analysis.sh?file={id}&db=Kainoki&mode=source"
        # )
        source = f"https://oncoj.orinst.ox.ac.uk/cgi-bin/analysis.sh?file={id}&db=Kainoki"
        tree: str = extract_pre_content(source)
        print(f"ID: {id}")
        print(f"souce: {source}")
        print(f"tree: {tree}")
        df.at[index, "tree"] = tree
        time.sleep(2)
    return df


def fetch_url_source(url):
    """
    指定されたURLのHTMLソースを取得する関数。

    Args:
        url (str): ソースを取得する対象のURL。

    Returns
    -------
        str: URLから取得したHTMLソース。

    Raises
    ------
        ValueError: 無効なURLが指定された場合。
        requests.exceptions.RequestException: HTTPリクエストエラーが発生した場合。
    """
    if not url.startswith(("http://", "https://")):
        raise ValueError("有効なURLを指定してください。例: 'http://example.com'")

    try:
        response = requests.get(url)
        response.raise_for_status()  # HTTPエラーがあれば例外を発生させる
        return response.text
    except requests.exceptions.RequestException as e:
        raise requests.exceptions.RequestException(f"URLの取得中にエラーが発生しました: {e}") from e


def extract_pre_content(html_source):
    """
    HTMLソースから<pre>タグ内の内容を抽出する。

    Args:
        html_source (str): HTMLソース。

    Returns
    -------
        str: <pre>タグ内のテキスト内容。
    """
    soup = BeautifulSoup(html_source, "html.parser")
    pre_tag = soup.find("pre")
    if pre_tag:
        print("pre_tag.get_text(strip=True)")
        print(pre_tag.get_text(strip=False))
        return pre_tag.get_text(strip=True)
    else:
        raise ValueError("HTMLソースに<pre>タグが見つかりませんでした。")


def add_tree_to_pos_data():
    """
    ./data/pos_data/*.jsonに対してtreeを付与してして保存する関数。

    Returns
    -------
    None
    """
    data_path = "./data/pos_data/"
    for file in ["train.json", "dev.json", "test.json"]:
        df = pd.read_json(data_path + file)
        df = get_tree(df)
        df.to_json(data_path + file, orient="records", lines=True)


def test():
    """
    TextWithInfoクラスの各種データ付与のテスト。

    Returns
    -------
    None
    """
    assert TextWithInfo("歩き疲れ、喉も渇ききってしまった。").ordinal_forms == [
        "歩き疲れる",
        "、",
        "喉",
        "も",
        "乾く",
        "切る",
        "て",
        "仕舞う",
        "た",
        "。",
    ]

    assert TextWithInfo("さわやかな秋の風に包まれながら食べるお弁当は最高です。").ordinal_forms == [
        "爽やか",
        "だ",
        "秋",
        "の",
        "風",
        "だ",
        "包む",
        "れる",
        "ながら",
        "食べる",
        "お",
        "弁当",
        "は",
        "最高",
        "です",
        "。",
    ]
    assert TextWithInfo("吾輩は猫である。名前はまだない。").pos_sequence == [
        "<PRONOUN>",
        "<PARTICLE>",
        "<NOUN>",
        "<AUXILIARY-VERB>",
        "<VERB>",
        "<AUXILIARY-SYMBOL>",
        "<NOUN>",
        "<PARTICLE>",
        "<ADVERB>",
        "<ADJECTIVE>",
        "<AUXILIARY-SYMBOL>",
    ]
    assert (
        TextWithInfo("吾輩は猫である。名前はまだない。").pos_frequency
        == "<PARTICLE>:2 <NOUN>:2 <AUXILIARY-SYMBOL>:2 <PRONOUN>:1 <AUXILIARY-VERB>:1 <VERB>:1 <ADVERB>:1 <ADJECTIVE>:1"
    )
    assert TextWithInfo("本日の降水量は、高いところで1,000,000mmでした。").pos_sequence == [
        "<NOUN>",
        "<PARTICLE>",
        "<NOUN>",
        "<NOUN>",
        "<PARTICLE>",
        "<AUXILIARY-SYMBOL>",
        "<ADJECTIVE>",
        "<NOUN>",
        "<PARTICLE>",
        "<NOUN>",
        "<AUXILIARY-SYMBOL>",
        "<NOUN>",
        "<AUXILIARY-SYMBOL>",
        "<NOUN>",
        "<NOUN>",
        "<AUXILIARY-VERB>",
        "<AUXILIARY-VERB>",
        "<AUXILIARY-SYMBOL>",
    ]
    assert (
        TextWithInfo("本日の降水量は、高いところで1,000,000mmでした。").pos_frequency
        == "<NOUN>:8 <AUXILIARY-SYMBOL>:4 <PARTICLE>:3 <AUXILIARY-VERB>:2 <ADJECTIVE>:1 "
    )


if __name__ == "__main__":
    test()
    make_pos_data()
