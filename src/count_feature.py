import pandas as pd

data_train = pd.read_json("data/pos_data/train.json")
data_valid = pd.read_json("data/pos_data/valid.json")
data_test = pd.read_json("data/pos_data/test.json")

data = pd.concat([data_train, data_valid, data_test], axis=0)

mo_count = 0

for _, row in data.iterrows():
    if "も" in row["ordinal_forms"]:
        mo_idx = row["ordinal_forms"].index("も")
        if row["ordinal_forms"][mo_idx - 1] == "ながら" and row["pos_sequence"][mo_idx - 1] == "<PARTICLE>":
            mo_count += 1

print(mo_count)
