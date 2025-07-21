import torch
from torch.utils.data import Dataset
import random
import json
from omegaconf import DictConfig
from typing import List, Dict, Any


class CustomDataset(Dataset):
    """
    データセットのクラス

    Args:
        data (list[dict]): データ
        tokenizer (transformers.AutoTokenizer): トークナイザ
        max_len (int): 最大トークン数
        cfg (DictConfig): 設定
    """

    def __init__(self, data, tokenizer, max_len, cfg: DictConfig):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.cfg: DictConfig = cfg
        self.experiment_type = cfg.experiment_type
        self.exclude_count = 0
        self.dummy_words_dict = {
            "<NOUN>": "ミョガパス",
            "<PRONOUN>": "彼女",
            "<ADJECTIVAL-NOUN>": "さもらか",
            "<PRENOUN-ADJECTIVAL>": "この",
            "<ADVERB>": "もさらく",
            "<CONJUNCTION>": "でありく",
            "<INTERJECTION>": "わあ",
            "<VERB>": "たゆねる",
            "<ADJECTIVE>": "もさらい",
            "<AUXILIARY-VERB>": "だ",
            "<PARTICLE>": "が",
            "<PREFIX>": "ふら",
            "<SUFFIX>": "ぼね",
            "<SYMBOL>": "。",
            "<AUXILIARY-SYMBOL>": "―",
            "<BLANK>": " ",
        }
        self.negation_position_dict = {"arg1": 0, "arg2": 0}  # negationの位置をカウント

    def __len__(self):
        """
        データ数を返す

        Returns
        -------
        int: データ数
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        データを取得

        Parameters
        ----------
        index (int): インデックス

        Returns
        -------
        dict: データ
        """
        item: dict = self.data[index]
        if self.cfg.experiment_type == "normal":
            text = f"{item['前件']}{item['接続表現']}{item['後件']}"
        elif self.cfg.experiment_type == "ordinal_forms":
            text = "".join(item["ordinal_forms"])
        elif self.cfg.experiment_type == "ordinal_forms_shuffle":
            shuffled_list: list[str] = random.sample(item["ordinal_forms"], len(item["ordinal_forms"]))
            text = "".join(shuffled_list)
        elif self.cfg.experiment_type == "pos_sequence":
            if not item["pos_sequence"]:
                raise ValueError("Empty pos_sequence encountered")
            text = "".join(item["pos_sequence"])
        elif self.cfg.experiment_type == "former-only":
            connective_idx = item["wakati"].index(item["接続表現"])
            self.exclude_count += len(item["wakati"]) - (connective_idx + 1)
            text = item["前件"]
        elif self.cfg.experiment_type == "latter-only":
            connective_idx = item["wakati"].index(item["接続表現"])
            self.exclude_count += connective_idx
            text = item["後件"]
        elif self.cfg.experiment_type == "convert_all_words_to_dummy":
            text = self.convert_all_words_to_dummy(item)
        elif self.cfg.experiment_type == "exclude_mo":
            text = self.exclude_mo(item)
        elif self.cfg.experiment_type == "exclude_koso":
            text = self.exclude_koso(item)
        elif self.cfg.experiment_type == "exclude_negation":
            text = self.exclude_negation(item)
        elif self.cfg.experiment_type == "exclude_function_words":
            text = self.exclude_function_words(item)
        elif self.cfg.experiment_type == "exclude_content_words":
            text = self.exclude_content_words(item)
        elif self.cfg.experiment_type == "exclude_particle":
            text = self.exclude_particle(item)
        elif self.cfg.experiment_type == "convert_content_words_to_dummy":
            text = self.convert_content_words_to_dummy(item)
        elif self.cfg.experiment_type == "convert_function_words_to_dummy":
            text = self.convert_function_words_to_dummy(item)
        elif self.cfg.experiment_type == "exclude_ha_and_ga":
            text = self.exclude_ha_and_ga(item)
        elif self.cfg.experiment_type == "exclude_connectives":
            text = self.exclude_connectives(item)
        elif self.cfg.experiment_type == "convert_connectives_to_dummy":
            text = self.convert_connectives_to_dummy(item)
        elif self.cfg.experiment_type == "exclude_iru_aru_oku":
            text = self.exclude_iru_aru_oku(item)
        else:
            raise ValueError("Invalid experiment type")

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        label = torch.tensor(item["label"], dtype=torch.long)
        if label >= self.cfg.num_labels:
            raise ValueError(f"Label {label} is out of range for num_labels={self.cfg.num_labels}")

        return {
            "input_ids": inputs["input_ids"].flatten(),
            "attention_mask": inputs["attention_mask"].flatten(),
            "labels": torch.tensor(item["label"], dtype=torch.long),
            "id": item["id"],  # idを取得
            "text": text,  # モデルへの入力テキストを追加
        }

    def convert_all_words_to_dummy(self, item: dict) -> str:
        """
        元の文から全ての単語をダミーに変換するメソッド。

        Parameters
        ----------
        item : dict
            データの辞書。CustomDatasetの__getitem__におけるitemの形式を想定。

        Returns
        -------
        str
            全ての単語をダミーに変換した文。
        """
        wakati_sentence: list = item["wakati"]
        for idx, pos in enumerate(item["pos_sequence"]):
            wakati_sentence[idx] = self.dummy_words_dict[pos]
            self.exclude_count += 1

        return "".join(wakati_sentence)

    def exclude_mo(self, item: dict) -> str:
        """
        元の文から「ながらも」の「も」と「つつも」の「も」を除外するメソッド。

        Parameters
        ----------
        item : dict
            データの辞書。CustomDatasetの__getitem__におけるitemの形式を想定。

        Returns
        -------
        str
            「ながらも」の「も」と「つつも」の「も」を除外した文。
        """
        if "も" in item["ordinal_forms"]:
            mo_idx = item["ordinal_forms"].index("も")
            if item["ordinal_forms"][mo_idx - 1] == "ながら" and item["pos_sequence"][mo_idx] == "<PARTICLE>":
                self.exclude_count += 1
                return "".join(item["ordinal_forms"][: mo_idx - 1]) + "".join(item["ordinal_forms"][mo_idx + 1 :])
            elif item["ordinal_forms"][mo_idx - 1] == "つつ" and item["pos_sequence"][mo_idx] == "<PARTICLE>":
                self.exclude_count += 1
                return "".join(item["ordinal_forms"][: mo_idx - 1]) + "".join(item["ordinal_forms"][mo_idx + 1 :])

        return "".join(item["ordinal_forms"])  # 「ながらも」「つつも」がない場合はそのまま返す

    def exclude_koso(self, item: dict) -> str:
        """
        元の文から「こそ」を除外するメソッド。

        Parameters
        ----------
        item : dict
            データの辞書。CustomDatasetの__getitem__におけるitemの形式を想定。

        Returns
        -------
        str
            「こそ」を除外した文。
        """
        if "こそ" in item["ordinal_forms"]:
            koso_idx = item["ordinal_forms"].index("こそ")
            if item["pos_sequence"][koso_idx] == "<PARTICLE>":
                self.exclude_count += 1
                return "".join(item["ordinal_forms"][:koso_idx]) + "".join(item["ordinal_forms"][koso_idx + 1 :])

        return "".join(item["ordinal_forms"])  # 「こそ」がない場合はそのまま返す

    def exclude_negation(self, item: dict) -> str:
        """
        元の文からnegationを除外するメソッド。

        Parameters
        ----------
        item : dict
            データの辞書。CustomDatasetの__getitem__におけるitemの形式を想定。

        Returns
        -------
        str
            negationを除外した文。
        """
        # negationがどこにあるかをカウントしておく
        connective_idx = item["wakati"].index(item["接続表現"])

        wakati_sentence: list = item["wakati"]
        wakati_ordinal_form: list = item["ordinal_forms"]
        pop_index: list[int] = []
        if not isinstance(wakati_sentence, list):
            raise ValueError("wakati_sentenceがlistでありません: ", wakati_sentence)
        if not isinstance(wakati_ordinal_form, list):
            raise ValueError("wakati_ordinal_formsがlistでありません: ", wakati_ordinal_form)

        for negation in ["ない", "無い", "なし", "無し", "非", "不", "無", "未", "反", "異"]:
            if negation in wakati_ordinal_form:
                neg_idx = wakati_ordinal_form.index(negation)  # idxは原型のlistから取り出す必要がある
                if neg_idx < connective_idx:
                    self.negation_position_dict["arg1"] += 1
                else:
                    self.negation_position_dict["arg2"] += 1
                pop_index.append(neg_idx)
                self.exclude_count += 1

        wakati_sentence = [wakati_sentence[i] for i in range(len(wakati_sentence)) if i not in pop_index]
        return "".join(wakati_sentence)  # 取り除いた後のlistをjoinして返す

    def exclude_function_words(self, item: dict) -> str:
        """
        元の文から機能語を除外するメソッド。

        Parameters
        ----------
        item : dict
            データの辞書。CustomDatasetの__getitem__におけるitemの形式を想定。

        Returns
        -------
        str
            機能語を除外した文。
        """
        wakati_sentence: list = item["wakati"]
        pos_sequence: list = item["pos_sequence"]
        pop_index: list[int] = []

        for idx, pos in enumerate(pos_sequence):
            if pos not in ["<NOUN>", "<VERB>", "<ADJECTIVE>", "<ADVERB>"]:  # 内容語でなければ落とす
                pop_index.append(idx)
                self.exclude_count += 1

        wakati_sentence = [wakati_sentence[i] for i in range(len(wakati_sentence)) if i not in pop_index]

        return "".join(wakati_sentence)  # 取り除いた後のlistをjoinして返す

    def exclude_content_words(self, item: dict) -> str:
        """
        元の文から内容語を除外するメソッド。

        Parameters
        ----------
        item : dict
            データの辞書。CustomDatasetの__getitem__におけるitemの形式を想定。

        Returns
        -------
        str
            内容語を除外した文。
        """
        wakati_sentence: list = item["wakati"]
        pos_sequence: list = item["pos_sequence"]
        pop_index: list[int] = []

        for idx, pos in enumerate(pos_sequence):
            if pos in ["<NOUN>", "<VERB>", "<ADJECTIVE>", "<ADVERB>"]:
                pop_index.append(idx)
                self.exclude_count += 1

        wakati_sentence = [wakati_sentence[i] for i in range(len(wakati_sentence)) if i not in pop_index]
        return "".join(wakati_sentence)  # 取り除いた後のlistをjoinして返す

    def convert_content_words_to_dummy(self, item: dict) -> str:
        """
        内容語をダミーに変換するメソッド。統語的構造を保ったまま内容語のlexical informationを除去する目的。

        Parameters
        ----------
        item : dict
            データの辞書。CustomDatasetの__getitem__におけるitemの形式を想定。

        Returns
        -------
        str
            内容語をダミーに変換した文。
        """
        wakati_sentence: list = item["wakati"]
        pos_sequence: list = item["pos_sequence"]

        for idx, pos in enumerate(pos_sequence):
            if pos in ["<NOUN>", "<VERB>", "<ADJECTIVE>", "<ADVERB>"]:
                wakati_sentence[idx] = self.dummy_words_dict[pos]
                self.exclude_count += 1
        return "".join(wakati_sentence)

    def convert_function_words_to_dummy(self, item: dict) -> str:
        """
        機能語をダミーに変換するメソッド。

        Parameters
        ----------
        item : dict
            データの辞書。CustomDatasetの__getitem__におけるitemの形式を想定。

        Returns
        -------
        str
            機能語をダミーに変換した文。
        """
        wakati_sentence: list = item["wakati"]
        pos_sequence: list = item["pos_sequence"]

        for idx, pos in enumerate(pos_sequence):
            if pos not in ["<NOUN>", "<VERB>", "<ADJECTIVE>", "<ADVERB>"]:
                wakati_sentence[idx] = self.dummy_words_dict[pos]
                self.exclude_count += 1
        return "".join(wakati_sentence)

    def exclude_particle(self, item: dict) -> str:
        """
        元の文から助詞を除外するメソッド。

        Parameters
        ----------
        item : dict
            データの辞書。CustomDatasetの__getitem__におけるitemの形式を想定。

        Returns
        -------
        str
            助詞を除外した文。
        """
        wakati_sentence: list = item["wakati"]
        pos_sequence: list = item["pos_sequence"]
        pop_index: list[int] = []

        for idx, pos in enumerate(pos_sequence):
            if pos == "<PARTICLE>":
                pop_index.append(idx)
                self.exclude_count += 1

        wakati_sentence = [wakati_sentence[i] for i in range(len(wakati_sentence)) if i not in pop_index]
        return "".join(wakati_sentence)

    def exclude_ha_and_ga(self, item: dict) -> str:
        """
        元の文から助詞の「は」と「が」を除外するメソッド。

        Parameters
        ----------
        item : dict
            データの辞書。CustomDatasetの__getitem__におけるitemの形式を想定。

        Returns
        -------
        str
            助詞の「は」と「が」を除外した文。
        """
        wakati_sentence: list = item["wakati"]
        pos_sequence: list = item["pos_sequence"]
        pop_index: list[int] = []

        for idx, pos in enumerate(pos_sequence):
            if pos == "<PARTICLE>":
                if wakati_sentence[idx] in ["は", "が"]:
                    pop_index.append(idx)
                    self.exclude_count += 1

        wakati_sentence = [wakati_sentence[i] for i in range(len(wakati_sentence)) if i not in pop_index]
        return "".join(wakati_sentence)

    def exclude_connectives(self, item: dict) -> str:
        """
        元の文から接続詞を除外するメソッド。

        Parameters
        ----------
        item : dict
            データの辞書。CustomDatasetの__getitem__におけるitemの形式を想定。

        Returns
        -------
        str
            接続詞を除外した文。
        """
        wakati_sentence: list = item["wakati"]
        pop_index: list[int] = []

        for idx, token in enumerate(wakati_sentence):
            if token in ["ながら", "つつ"]:
                pop_index.append(idx)
                self.exclude_count += 1
                break
            if token in ["ところ"] and wakati_sentence[idx + 1] == "で":
                pop_index.append(idx)
                pop_index.append(idx + 1)
                self.exclude_count += 2
                # self.exclude_count += 1
                break

        wakati_sentence = [wakati_sentence[i] for i in range(len(wakati_sentence)) if i not in pop_index]
        return "".join(wakati_sentence)

    def convert_connectives_to_dummy(self, item: dict) -> str:
        """
        接続詞をダミーに変換するメソッド。統語的構造を保ったまま接続詞のlexical informationを除去する目的。

        Parameters
        ----------
        item : dict
            データの辞書。CustomDatasetの__getitem__におけるitemの形式を想定。

        Returns
        -------
        str
            接続詞をダミーに変換した文。
        """
        wakati_sentence: list = item["wakati"]

        for idx, token in enumerate(wakati_sentence):
            if token in ["ながら", "つつ"]:
                wakati_sentence[idx] = self.dummy_words_dict["<CONJUNCTION>"]
                self.exclude_count += 1
            if token in ["ところ"] and wakati_sentence[idx + 1] == "で":
                wakati_sentence[idx] = self.dummy_words_dict["<CONJUNCTION>"]
                wakati_sentence[idx + 1] = self.dummy_words_dict["<PARTICLE>"]
                self.exclude_count += 2

        return "".join(wakati_sentence)

    def exclude_iru_aru_oku(self, item: dict) -> str:
        """
        元の文から「いる」「ある」「おく」を除外するメソッド。

        Parameters
        ----------
        item : dict
            データの辞書。CustomDatasetの__getitem__におけるitemの形式を想定。

        Returns
        -------
        str
            「いる」「ある」「おく」を除外した文。
        """
        wakati_sentence: list = item["wakati"]
        wakati_ordinal_form: list = item["ordinal_forms"]
        pop_index: list[int] = []

        for idx, token in enumerate(wakati_ordinal_form):
            if token in [
                "いる",
                "ある",
                "おく",
                "居る",
                "有る",
                "置く",
                "在る",
            ]:
                pop_index.append(idx)
                self.exclude_count += 1

        wakati_sentence = [wakati_sentence[i] for i in range(len(wakati_sentence)) if i not in pop_index]
        return "".join(wakati_sentence)


class CustomDatasetForGenModel(Dataset):
    """
    OpenAI GPT 系 API 用データセット

    Args:
        data (list[dict]): 入力データ
        cfg (DictConfig): 実験設定
    """

    # ---------- 初期化 -------------------------------------------------
    def __init__(self, data: List[Dict[str, Any]], cfg: DictConfig):
        self.data = data
        self.cfg: DictConfig = cfg
        print(f"Config: \n{self.cfg}")  # 設定を表示

        # 既存実験用の属性はそのまま残す
        self.experiment_type = cfg.experiment_type
        self.exclude_count = 0
        self.dummy_words_dict = {
            "<NOUN>": "ミョガパス",
            "<PRONOUN>": "彼女",
            "<ADJECTIVAL-NOUN>": "さもらか",
            "<PRENOUN-ADJECTIVAL>": "この",
            "<ADVERB>": "もさらく",
            "<CONJUNCTION>": "でありく",
            "<INTERJECTION>": "わあ",
            "<VERB>": "たゆねる",
            "<ADJECTIVE>": "もさらい",
            "<AUXILIARY-VERB>": "だ",
            "<PARTICLE>": "が",
            "<PREFIX>": "ふら",
            "<SUFFIX>": "ぼね",
            "<SYMBOL>": "。",
            "<AUXILIARY-SYMBOL>": "―",
            "<BLANK>": " ",
        }
        self.negation_position_dict = {"arg1": 0, "arg2": 0}

    # ---------- 必須メソッド -------------------------------------------
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item: dict = self.data[index]

        # ---- 1. プロンプト文字列生成（元コードの条件分岐を踏襲） ----------
        if self.cfg.experiment_type == "normal":
            text = f"{item['前件']}{item['接続表現']}{item['後件']}"
        elif self.cfg.experiment_type == "ordinal_forms":
            text = "".join(item["ordinal_forms"])
        elif self.cfg.experiment_type == "ordinal_forms_shuffle":
            shuffled_list: List[str] = random.sample(item["ordinal_forms"], len(item["ordinal_forms"]))
            text = "".join(shuffled_list)
        elif self.cfg.experiment_type == "pos_sequence":
            if not item["pos_sequence"]:
                raise ValueError("Empty pos_sequence encountered")
            text = "".join(item["pos_sequence"])
        elif self.cfg.experiment_type == "former-only":
            connective_idx = item["wakati"].index(item["接続表現"])
            self.exclude_count += len(item["wakati"]) - (connective_idx + 1)
            text = item["前件"]
        elif self.cfg.experiment_type == "latter-only":
            connective_idx = item["wakati"].index(item["接続表現"])
            self.exclude_count += connective_idx
            text = item["後件"]
        elif self.cfg.experiment_type == "convert_all_words_to_dummy":
            text = self.convert_all_words_to_dummy(item)
        elif self.cfg.experiment_type == "exclude_mo":
            text = self.exclude_mo(item)
        elif self.cfg.experiment_type == "exclude_koso":
            text = self.exclude_koso(item)
        elif self.cfg.experiment_type == "exclude_negation":
            text = self.exclude_negation(item)
        elif self.cfg.experiment_type == "exclude_function_words":
            text = self.exclude_function_words(item)
        elif self.cfg.experiment_type == "exclude_content_words":
            text = self.exclude_content_words(item)
        elif self.cfg.experiment_type == "exclude_particle":
            text = self.exclude_particle(item)
        elif self.cfg.experiment_type == "convert_content_words_to_dummy":
            text = self.convert_content_words_to_dummy(item)
        elif self.cfg.experiment_type == "convert_function_words_to_dummy":
            text = self.convert_function_words_to_dummy(item)
        elif self.cfg.experiment_type == "exclude_ha_and_ga":
            text = self.exclude_ha_and_ga(item)
        elif self.cfg.experiment_type == "exclude_connectives":
            text = self.exclude_connectives(item)
        elif self.cfg.experiment_type == "convert_connectives_to_dummy":
            text = self.convert_connectives_to_dummy(item)
        elif self.cfg.experiment_type == "exclude_iru_aru_oku":
            text = self.exclude_iru_aru_oku(item)
        else:
            raise ValueError("Invalid experiment type")

        # ---- 2. ラベル整合性チェック ------------------------------------
        label = item["label"]
        if label >= self.cfg.num_labels:
            raise ValueError(f"Label {label} is out of range for num_labels={self.cfg.num_labels}")

        # ---- 3. GPT 用にそのまま返す -----------------------------------
        return {
            "prompt": text,  # ★ ここだけで十分
            "label": torch.tensor(label, dtype=torch.long),
            "id": item["id"],
        }

    def convert_all_words_to_dummy(self, item: dict) -> str:
        wakati = item["wakati"].copy()
        for idx, pos in enumerate(item["pos_sequence"]):
            wakati[idx] = self.dummy_words_dict[pos]
            self.exclude_count += 1
        return "".join(wakati)

    def exclude_mo(self, item: dict) -> str:
        if "も" in item["ordinal_forms"]:
            mo_idx = item["ordinal_forms"].index("も")
            if item["ordinal_forms"][mo_idx - 1] == "ながら" and item["pos_sequence"][mo_idx] == "<PARTICLE>":
                self.exclude_count += 1
                return "".join(item["ordinal_forms"][: mo_idx - 1]) + "".join(item["ordinal_forms"][mo_idx + 1 :])
            elif item["ordinal_forms"][mo_idx - 1] == "つつ" and item["pos_sequence"][mo_idx] == "<PARTICLE>":
                self.exclude_count += 1
                return "".join(item["ordinal_forms"][: mo_idx - 1]) + "".join(item["ordinal_forms"][mo_idx + 1 :])
        return "".join(item["ordinal_forms"])

    def exclude_koso(self, item: dict) -> str:
        if "こそ" in item["ordinal_forms"]:
            koso_idx = item["ordinal_forms"].index("こそ")
            if item["pos_sequence"][koso_idx] == "<PARTICLE>":
                self.exclude_count += 1
                return "".join(item["ordinal_forms"][:koso_idx]) + "".join(item["ordinal_forms"][koso_idx + 1 :])
        return "".join(item["ordinal_forms"])

    def exclude_negation(self, item: dict) -> str:
        connective_idx = item["wakati"].index(item["接続表現"])
        wakati_sentence = item["wakati"].copy()
        pop_index: List[int] = []
        for neg in ["ない", "無い", "なし", "無し", "非", "不", "無", "未", "反", "異"]:
            if neg in item["ordinal_forms"]:
                neg_idx = item["ordinal_forms"].index(neg)
                if neg_idx < connective_idx:
                    self.negation_position_dict["arg1"] += 1
                else:
                    self.negation_position_dict["arg2"] += 1
                pop_index.append(neg_idx)
                self.exclude_count += 1
        wakati_sentence = [w for i, w in enumerate(wakati_sentence) if i not in pop_index]
        return "".join(wakati_sentence)

    def exclude_function_words(self, item: dict) -> str:
        pop_index: List[int] = [
            idx
            for idx, pos in enumerate(item["pos_sequence"])
            if pos not in ["<NOUN>", "<VERB>", "<ADJECTIVE>", "<ADVERB>"]
        ]
        self.exclude_count += len(pop_index)
        return "".join([w for i, w in enumerate(item["wakati"]) if i not in pop_index])

    def exclude_content_words(self, item: dict) -> str:
        pop_index: List[int] = [
            idx
            for idx, pos in enumerate(item["pos_sequence"])
            if pos in ["<NOUN>", "<VERB>", "<ADJECTIVE>", "<ADVERB>"]
        ]
        self.exclude_count += len(pop_index)
        return "".join([w for i, w in enumerate(item["wakati"]) if i not in pop_index])

    def convert_content_words_to_dummy(self, item: dict) -> str:
        wakati = item["wakati"].copy()
        for idx, pos in enumerate(item["pos_sequence"]):
            if pos in ["<NOUN>", "<VERB>", "<ADJECTIVE>", "<ADVERB>"]:
                wakati[idx] = self.dummy_words_dict[pos]
                self.exclude_count += 1
        return "".join(wakati)

    def convert_function_words_to_dummy(self, item: dict) -> str:
        wakati = item["wakati"].copy()
        for idx, pos in enumerate(item["pos_sequence"]):
            if pos not in ["<NOUN>", "<VERB>", "<ADJECTIVE>", "<ADVERB>"]:
                wakati[idx] = self.dummy_words_dict[pos]
                self.exclude_count += 1
        return "".join(wakati)

    def exclude_particle(self, item: dict) -> str:
        pop_index = [idx for idx, pos in enumerate(item["pos_sequence"]) if pos == "<PARTICLE>"]
        self.exclude_count += len(pop_index)
        return "".join([w for i, w in enumerate(item["wakati"]) if i not in pop_index])

    def exclude_ha_and_ga(self, item: dict) -> str:
        pop_index = [
            idx
            for idx, (pos, token) in enumerate(zip(item["pos_sequence"], item["wakati"], strict=False))
            if pos == "<PARTICLE>" and token in ["は", "が"]
        ]
        self.exclude_count += len(pop_index)
        return "".join([w for i, w in enumerate(item["wakati"]) if i not in pop_index])

    def exclude_connectives(self, item: dict) -> str:
        wakati = item["wakati"].copy()
        pop_index: List[int] = []
        for idx, token in enumerate(wakati):
            if token in ["ながら", "つつ"]:
                pop_index.append(idx)
                self.exclude_count += 1
                break
            if token == "ところ" and idx + 1 < len(wakati) and wakati[idx + 1] == "で":
                pop_index.extend([idx, idx + 1])
                self.exclude_count += 2
                break
        return "".join([w for i, w in enumerate(wakati) if i not in pop_index])

    def convert_connectives_to_dummy(self, item: dict) -> str:
        wakati = item["wakati"].copy()
        for idx, token in enumerate(wakati):
            if token in ["ながら", "つつ"]:
                wakati[idx] = self.dummy_words_dict["<CONJUNCTION>"]
                self.exclude_count += 1
            if token == "ところ" and idx + 1 < len(wakati) and wakati[idx + 1] == "で":
                wakati[idx] = self.dummy_words_dict["<CONJUNCTION>"]
                wakati[idx + 1] = self.dummy_words_dict["<PARTICLE>"]
                self.exclude_count += 2
        return "".join(wakati)

    def exclude_iru_aru_oku(self, item: dict) -> str:
        pop_index = [
            idx
            for idx, token in enumerate(item["ordinal_forms"])
            if token
            in [
                "いる",
                "ある",
                "おく",
                "居る",
                "有る",
                "置く",
                "在る",
            ]
        ]
        self.exclude_count += len(pop_index)
        return "".join([w for i, w in enumerate(item["wakati"]) if i not in pop_index])


# データセットの作成
def create_dataset(
    data_path: str,
    tokenizer,
    max_len,
    cfg: DictConfig,
):
    """
    データセットを作成する

    Parameters
    ----------
    data_path : str
        データのパス
    tokenizer : transformers.AutoTokenizer
        トークナイザ
    max_len : int
        最大トークン数
    cfg : DictConfig
        設定

    Returns
    -------
    CustomDataset
        データセット
    """
    with open(data_path, "r") as f:
        data = json.load(f)

    if cfg.is_debug:
        print("!!!!!!!!!!!!!!!!!!!!!! warning. This is debug mode. !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        data = data[:10]  # デバッグ用にデータを10件に制限

    dataset = CustomDataset(data, tokenizer, max_len, cfg)
    return dataset


def create_dataset_for_gen_model(
    data_path: str,
    cfg: DictConfig,
):
    """
    JSON から `CustomDataset` を生成

    Parameters
    ----------
    data_path : str
        入力 JSON ファイル
    cfg : DictConfig
        実験設定

    Returns
    -------
    CustomDataset
        GPT 用データセット
    """
    with open(data_path, "r") as f:
        data = json.load(f)

    if cfg.is_debug:
        print("!!!!!!!!!!!!!!!!!!!!!! warning. This is debug mode. !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        data = data[:10]

    return CustomDatasetForGenModel(data, cfg)
