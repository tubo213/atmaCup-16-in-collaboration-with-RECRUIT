from pathlib import Path

import polars as pl
import torch
from pytorch_lightning import LightningDataModule
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import k_hop_subgraph
from tqdm import tqdm


# InMemoryDatasetを継承して、データセットを作成
# 一回作成すると、rootにキャッシュされる
# 再度作成する場合は、rootを削除する
# 参考: https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html#creating-in-memory-datasets # noqa
class YadDataset(InMemoryDataset):
    def __init__(self, root, G, log_df, label_df, k=3, name: str = "train_fold0"):
        self.G = G
        self.k = k
        self.log_df = log_df
        self.label_df = label_df
        self.name = name
        super(YadDataset, self).__init__(root, transform=None, pre_transform=None)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        # このメソッドは、処理済みデータファイルの名前のリストを返します
        return [f"{self.name}_data_k{self.k}.pt"]

    def process(self):
        # データの前処理と`Data`オブジェクトの作成
        seq_df = self.log_df.groupby("session_id").agg(pl.col("yad_no")).sort("session_id")
        seq_df = seq_df.join(
            self.label_df.select(pl.col("session_id"), pl.col("yad_no").alias("label")),
            on="session_id",
            how="left",
        )
        seq_list = seq_df.get_column("yad_no").to_list()
        label_list = seq_df.get_column("label").to_list()

        data_list = []
        for seq, label in tqdm(zip(seq_list, label_list), total=len(seq_list)):
            data_list.append(self.create_subgraph_data(seq, label))

        self.save(data_list, self.processed_paths[0])

    def create_subgraph_data(self, seq, label):
        # サブグラフデータの作成
        node_idx = torch.tensor(seq, dtype=torch.long)
        subset, subset_edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx=node_idx,
            num_hops=self.k,
            edge_index=self.G.edge_index,
            relabel_nodes=True,
            flow="target_to_source",
        )

        # edge特徴量
        edge_attr = self.G.edge_attr[edge_mask]  # (E, 1)
        # 訪問済みノードに対応するエッジのみ1
        connected = torch.isin(subset_edge_index, mapping).float().T  # (E, 2)
        # 訪問済みノードから訪問済みノードへのエッジのみ1
        seq_edge = connected.prod(dim=1).view(-1, 1)  # (E, 1)
        edge_attr = torch.cat([edge_attr, connected, seq_edge], dim=1)  # (E, 4)

        # ラベル
        y = (subset == label).float()

        # ノード特徴量
        x = self.G.x[subset]
        # 最後のノードは1, それ以外は0
        is_last = torch.zeros(x.shape[0], dtype=float)
        is_last[mapping[-1]] = 1.0
        # 訪問済みノードは1, 未訪問ノードは0
        is_visited = torch.zeros(x.shape[0], dtype=float)
        is_visited[mapping] = 1.0
        # 訪問順
        order_of_visit = torch.zeros((x.shape[0]), dtype=float)
        order_of_visit[mapping] = torch.arange(len(mapping), dtype=float) + 1.0
        # 奇数番目のノードは1, 偶数番目のノードは0
        is_odd = torch.zeros((x.shape[0])).long()
        is_odd[mapping] = torch.arange(1, len(mapping) + 1).long()
        is_odd = (is_odd % 2).float()

        x = torch.cat(
            [
                x,
                is_last.view(-1, 1),
                is_visited.view(-1, 1),
                torch.log1p(order_of_visit.view(-1, 1)),
                is_odd.view(-1, 1),
            ],
            dim=1,
        )

        return Data(
            x=x.float(),
            edge_index=subset_edge_index,
            edge_attr=edge_attr,
            y=y,
            subset_node_idx=subset,
            label=label,
        )


class YadDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.data_dir = Path(cfg.dir.data_dir)
        self.processed_dir = Path(cfg.dir.processed_dir)
        self.ds_root = str(self.processed_dir / "yad_dataset")
        self.log_df = pl.read_csv(self.processed_dir / "train_log.csv")
        self.label_df = pl.read_csv(self.processed_dir / "train_label.csv")
        self.G = torch.load(self.processed_dir / "graph.pt")

        # split
        self.train_log_df = self.log_df.filter(pl.col("fold") != cfg.fold)
        self.val_log_df = self.log_df.filter(pl.col("fold") == cfg.fold)

    def train_dataloader(self):
        name = f"train_fold{self.cfg.fold}"
        train_ds = YadDataset(
            self.ds_root, self.G, self.train_log_df, self.label_df, self.cfg.k, name
        )
        return DataLoader(
            train_ds,
            batch_size=self.cfg.dataset.batch_size,
            shuffle=True,
            num_workers=self.cfg.dataset.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        name = f"val_fold{self.cfg.fold}"
        val_ds = YadDataset(
            self.ds_root, self.G, self.val_log_df, self.label_df, self.cfg.k, name=name
        )
        return DataLoader(
            val_ds,
            batch_size=self.cfg.dataset.val_batch_size,
            shuffle=False,
            num_workers=self.cfg.dataset.num_workers,
            pin_memory=True,
            drop_last=False,
        )
