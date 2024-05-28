import os
import sys

import json
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Any

LOOKUP_FILE = Path(__file__).parent.parent.parent / Path("references/3_class.json")


def load_data(filename) -> dict[str, Any]:
    with open(filename, "r") as fh:
        data = json.load(fh)

    features = data["features"]
    props = [_.get("properties") for _ in features]
    return {k: v for _ in props for k, v in _.items()}


def load_label_file(order):
    with open(LOOKUP_FILE, "r") as fh:
        lables = json.load(fh)

    swap = {v: k for k, v in lables.items()}
    new_labels = []
    for old in order:
        new_labels.append(swap.get(old))
    return new_labels


def init_confusion_matrix(array: list[float], labels: list[str]) -> pd.DataFrame:
    return pd.DataFrame(data=array, columns=labels, index=labels)


def add_producers(df: pd.DataFrame, array: list[str]) -> pd.DataFrame:
    df = df.reindex(columns=df.columns.tolist() + ["Producers"])
    pro = list(map(lambda x: round(x * 100, 2), array))
    df["Producers"] = pro
    return df


def add_consumers(df: pd.DataFrame, array: list[str]) -> pd.DataFrame:
    new_index = pd.Index(df.index.tolist() + ["Consumers"])
    df = df.reindex(new_index).fillna(value=np.nan)
    df.iloc[-1, 0:-1] = list(map(lambda x: (x * 100), array))
    return df


def add_overall(df: pd.DataFrame, data: float) -> pd.DataFrame:
    df = df.reindex(df.index.tolist() + ["Overall Accuracy"]).fillna(value=np.nan)
    df.iloc[-1, 0] = round(data * 100, 2)
    return df


def build_confusion_matrix(data: dict[str, Any], lables: str = None) -> pd.DataFrame:

    table = init_confusion_matrix(array=data.get("confusion_matrix"), labels=lables)
    table = add_producers(df=table, array=data.get("producers"))
    table = add_consumers(df=table, array=data.get("consumers"))
    table = add_overall(df=table, data=data.get("overall"))
    return table


def main(args: list[str]) -> int:
    if len(args) != 1:
        raise RuntimeError

    filename = args[0]
    if not filename.endswith(".geojson"):
        raise RuntimeError
    data = load_data(filename)
    new_labels = load_label_file(data.get("order"))
    cfm = build_confusion_matrix(data, new_labels)
    name = f'{os.path.basename(filename).split(".")[0]}.csv'
    outlocation = os.path.abspath(os.path.dirname(filename))
    cfm.to_csv(os.path.join(outlocation, name))

    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
