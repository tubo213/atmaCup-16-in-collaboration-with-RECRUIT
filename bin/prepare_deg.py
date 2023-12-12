from pathlib import Path

import hydra
import polars as pl
import torch
from torch_geometric.utils import degree
from tqdm import tqdm

from src.datamodule import YadDataset


def compute_deg(ds: YadDataset) -> torch.Tensor:
    # Compute the maximum in-degree in the training data.
    max_degree = -1
    for data in tqdm(ds, desc="max_degree"):
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))

    # Compute the in-degree histogram tensor
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in tqdm(ds, desc="deg"):
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())

    return deg


@hydra.main(config_path="conf", config_name="prepare_deg.yaml")
def main(cfg):
    processed_dir = Path(cfg.dir.processed_dir)
    processed_dir.mkdir(exist_ok=True, parents=True)
    output_dir = processed_dir / "deg"
    output_dir.mkdir(exist_ok=True, parents=True)

    train_log_df = pl.read_csv(processed_dir / "train_log.csv")
    train_label_df = pl.read_csv(processed_dir / "train_label.csv")
    G = torch.load(processed_dir / "graph.pt")

    ds = YadDataset(G, train_log_df, train_label_df, k=cfg.k)
    deg = compute_deg(ds)

    # Save the in-degree histogram tensor.
    torch.save(deg, output_dir / f"deg_k{cfg.k}.pt")


if __name__ == "__main__":
    main()
