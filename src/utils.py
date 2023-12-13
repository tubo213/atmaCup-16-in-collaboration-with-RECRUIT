from pathlib import Path
from typing import Literal, TypeAlias

import pandas as pd
import polars as pl

DataFrame: TypeAlias = pl.DataFrame | pd.DataFrame
DfType: TypeAlias = Literal["pl", "pd"]


class Atma16Loader:
    def __init__(self, input_dir: Path):
        self.input_dir = input_dir
        self.csv_paths = input_dir.glob("*.csv")

    @staticmethod
    def _load_csv(path: Path, frame_type: DfType) -> DataFrame:
        return pl.read_csv(path) if frame_type == "pl" else pd.read_csv(path)

    @staticmethod
    def _load_parquet(path: Path, frame_type: DfType) -> DataFrame:
        return pl.read_parquet(path) if frame_type == "pl" else pd.read_parquet(path)

    def load_test_log(self, frame_type: DfType = "pl") -> DataFrame:
        return self._load_csv(self.input_dir / "test_log.csv", frame_type)

    def load_train_log(self, frame_type: DfType = "pl") -> DataFrame:
        return self._load_csv(self.input_dir / "train_log.csv", frame_type)

    def load_all_log(self, frame_type: DfType = "pl") -> DataFrame:
        df = pl.concat([self.load_train_log("pl"), self.load_test_log("pl")])
        return df if frame_type == "pl" else df.to_pandas()

    def load_ses2idx(self) -> tuple[dict[int, str], dict[str, int]]:
        idx2ses = dict(enumerate(self.load_all_log("pd")["session_id"].unique()))
        ses2idx = {k: idx for idx, k in idx2ses.items()}
        assert ses2idx["000007603d533d30453cc45d0f3d119f"] == 0
        assert idx2ses[0] == "000007603d533d30453cc45d0f3d119f"
        return idx2ses, ses2idx

    def load_train_label(self, frame_type: DfType = "pl") -> DataFrame:
        return self._load_csv(self.input_dir / "train_label.csv", frame_type)

    def load_yad(self, frame_type: DfType = "pl") -> DataFrame:
        return self._load_csv(self.input_dir / "yado.csv", frame_type)

    def load_image(self, frame_type: DfType = "pl") -> DataFrame:
        return self._load_parquet(self.input_dir / "image_embeddings.parquet", frame_type)

    def load_all_dfs(self, frame_type: DfType = "pl") -> dict[str, DataFrame]:
        return {
            path.stem: self._load_csv(path, frame_type) for path in self.input_dir.glob("*.csv")
        }

    def load_sample_submission(self, frame_type: DfType = "pl") -> DataFrame:
        return self._load_csv(self.input_dir / "sample_submission.csv", frame_type)


def apk(actual: int, predicted: list[int], k: int = 10):
    """
    Computes the average precision at k for a single actual value.

    Parameters:
    actual : int
        The actual value that is to be predicted
    predicted : list
        A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements

    Returns:
    float
        The average precision at k
    """
    if actual in predicted[:k]:
        return 1.0 / (predicted[:k].index(actual) + 1)
    return 0.0


def mapk(actual: list[int], predicted: list[list[int]], k=10):
    """
    Computes the mean average precision at k for lists of actual values and predicted values.

    Parameters:
    actual : list
        A list of actual values that are to be predicted
    predicted : list
        A list of lists of predicted elements (order does matter in the lists)
    k : int, optional
        The maximum number of predicted elements

    Returns:
    float
        The mean average precision at k
    """
    return sum(apk(a, p, k) for a, p in zip(actual, predicted)) / len(actual)


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k

        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
