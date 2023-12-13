from collections import defaultdict
from pathlib import Path
from typing import Dict

import hydra
import numpy as np
import polars as pl
import shirokumas as sk
import torch
from sklearn.model_selection import GroupKFold
from torch_geometric.data import Data
from tqdm import tqdm

from src.conf import PrepareDataConfig
from src.utils import Atma16Loader

NODE_FEATURE_COLS = [
    "wid_cd",  # cat
    "ken_cd",  # cat
    "lrg_cd",  # cat
    "sml_cd",  # cat
    "yad_type",  # numeric
    "wireless_lan_flg",  # numeric
    "onsen_flg",  # numeric
    "total_room_cnt",  # numeric
    "kd_stn_5min",  # numeric
    "kd_bch_5min",  # numeric
    "kd_slp_5min",  # numeric
    "kd_conv_walk_5min",  # numeric
    "counts",  # numeric
    # "wid_rank",  # numeric
    # "ken_rank",  # numeric
    # "lrg_rank",  # numeric
    # "sml_rank",  # numeric
]


def preprocess_yad(all_log_df: pl.DataFrame, yad_df: pl.DataFrame) -> pl.DataFrame:
    # yad_noの出現回数
    vc = all_log_df.get_column("yad_no").value_counts()
    yad_df = yad_df.join(vc, on="yad_no", how="left")

    # カテゴリ内でのcountのランキング
    yad_df = yad_df.with_columns(
        pl.col("counts").rank(method="min", descending=True).over("wid_cd").alias("wid_rank"),
        pl.col("counts").rank(method="min", descending=True).over("ken_cd").alias("ken_rank"),
        pl.col("counts").rank(method="min", descending=True).over("lrg_cd").alias("lrg_rank"),
        pl.col("counts").rank(method="min", descending=True).over("sml_cd").alias("sml_rank"),
    )

    # 欠損値を0埋め
    yad_df = yad_df.with_columns(
        pl.col(
            [
                "total_room_cnt",
                "kd_stn_5min",
                "kd_bch_5min",
                "kd_slp_5min",
                "kd_conv_walk_5min",
                "counts",
            ]
        ).fill_null(0)
    )

    # 欠損値を-1埋め
    yad_df = yad_df.with_columns(pl.col(["wireless_lan_flg"]).fill_null(-1))

    # 数値をlog1p変換
    yad_df = yad_df.with_columns(
        np.log1p(
            pl.col(
                [
                    "total_room_cnt",
                    "kd_stn_5min",
                    "kd_bch_5min",
                    "kd_slp_5min",
                    "kd_conv_walk_5min",
                    "counts",
                    "wid_rank",
                    "ken_rank",
                    "lrg_rank",
                    "sml_rank",
                ]
            )
        )
    )

    # cat列をlabel encoding
    cat_cols = ["wid_cd", "ken_cd", "lrg_cd", "sml_cd"]
    oe = sk.OrdinalEncoder(cols=cat_cols)
    yad_df = yad_df.with_columns(oe.fit_transform(yad_df))

    return yad_df


def create_node_features(df: pl.DataFrame, cols: list):
    node_features = df.select(pl.col(cols)).to_numpy()
    return torch.tensor(node_features, dtype=torch.float)


def create_edge_index(df: pl.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    edge_dict: Dict[tuple, int] = defaultdict(int)  # 2つのyado_noの遷移回数を記録するdict

    for _, gdf in tqdm(
        df.group_by("session_id"), total=df["session_id"].n_unique(), desc="create edge index"
    ):
        if gdf.shape[0] == 1:
            continue

        yad_nos = gdf["yad_no"].to_numpy()  # yado_noの遷移のlist
        for i in range(yad_nos.shape[0] - 1):
            edge_dict[(yad_nos[i], yad_nos[i + 1])] += 1  # 2つのyado_noの遷移が何回あったか

    edge_index = torch.tensor(list(edge_dict.keys()), dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(list(edge_dict.values()), dtype=torch.float).view(-1, 1)

    return edge_index, edge_attr


def build_graph(
    yad_df: pl.DataFrame,
    all_log_df: pl.DataFrame,
    node_feature_cols: list[str],
) -> Data:
    X = create_node_features(yad_df, node_feature_cols)
    edge_index, edge_attr = create_edge_index(all_log_df)
    # edge_attrを対数変換
    edge_attr = torch.log1p(edge_attr)
    return Data(
        x=X,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )


def train_test_split(train_log_df: pl.DataFrame, n_splits=5) -> pl.DataFrame:
    # session_idでGroupKFold
    kf = GroupKFold(n_splits=n_splits)
    # foldを列に追加
    train_log_df = train_log_df.with_columns(pl.lit(-1).alias("fold"))
    for fold, (_, test_idx) in enumerate(
        kf.split(train_log_df, groups=train_log_df["session_id"].to_numpy())
    ):
        train_log_df[test_idx, -1] = fold

    return train_log_df


@hydra.main(config_path="conf", config_name="prepare_data.yaml")
def main(cfg: PrepareDataConfig):
    input_dir = Path(cfg.dir.data_dir)
    processed_dir = Path(cfg.dir.processed_dir)
    processed_dir.mkdir(exist_ok=True, parents=True)

    dl = Atma16Loader(input_dir)
    all_log_df = dl.load_all_log()
    yad_df = dl.load_yad()
    test_log_df = dl.load_test_log()
    train_log_df = dl.load_train_log()
    train_label_df = dl.load_train_label()

    # yad_noを0indexに変換
    yad_df = yad_df.with_columns(
        pl.col("yad_no") - 1,
    )
    all_log_df = all_log_df.with_columns(
        pl.col("yad_no") - 1,
    )
    test_log_df = test_log_df.with_columns(
        pl.col("yad_no") - 1,
    )
    train_log_df = train_log_df.with_columns(
        pl.col("yad_no") - 1,
    )
    train_label_df = train_label_df.with_columns(
        pl.col("yad_no") - 1,
    )

    # yad_dfの前処理
    yad_df = preprocess_yad(all_log_df, yad_df)

    # グラフの構築
    G = build_graph(yad_df, all_log_df, NODE_FEATURE_COLS)

    # グラフの保存
    torch.save(G, processed_dir / "graph.pt")

    # train test split
    train_log_df = train_test_split(train_log_df, n_splits=cfg.n_splits)

    # 保存
    test_log_df.write_csv(processed_dir / "test_log.csv")
    yad_df.write_csv(processed_dir / "yad.csv")
    train_log_df.write_csv(processed_dir / "train_log.csv")
    train_label_df.write_csv(processed_dir / "train_label.csv")


if __name__ == "__main__":
    main()
