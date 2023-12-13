from pathlib import Path
from typing import Literal, Optional

import hydra
import pandas as pd
import polars as pl
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from pytorch_lightning import seed_everything
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import unbatch
from tqdm import tqdm

from src.conf import InferenceConfig, TrainConfig
from src.datamodule import YadDataset
from src.model import YadGNN

PHASE_TYPE = Literal["val", "test"]


def load_train_cfg(output_dir: Path, exp_name: str, run_name: str) -> TrainConfig:
    path = (output_dir / exp_name / run_name / ".hydra").as_posix()
    GlobalHydra.instance().clear()  # clear the GlobalHydra object
    initialize(config_path=path)
    return compose("config")  # type: ignore


def load_log_df(
    processed_dir: Path, phase: PHASE_TYPE, fold: Optional[int] = None
) -> pl.DataFrame:
    if phase == "val":
        log_df = pl.read_csv(processed_dir / "train_log.csv")
        return log_df.filter(pl.col("fold") == fold)
    elif phase == "test":
        return pl.read_csv(processed_dir / "test_log.csv")
    else:
        raise ValueError(f"phase={phase} is not supported.")


def get_dataset_name(phase: PHASE_TYPE, fold: Optional[int] = None) -> str:
    if phase == "val":
        return f"val_fold{fold}"
    elif phase == "test":
        return "test"
    else:
        raise ValueError(f"phase={phase} is not supported.")


def get_dataloader(
    processed_dir: Path,
    G: Data,
    log_df: pl.DataFrame,
    k: int,
    batch_size: int,
    num_workers: int,
    name: str,
):
    ds = YadDataset(
        root=str(processed_dir / "yad_dataset"),
        G=G,
        log_df=log_df,
        label_df=None,
        k=k,
        name=name,
    )
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return dl


def inference(model: YadGNN, dl: DataLoader, device: torch.device, use_amp: bool) -> pl.DataFrame:
    model = model.to(device)
    model.eval()

    preds = []
    yad_ids = []
    session_ids = []
    for batch in tqdm(dl, desc="inference"):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_amp):
                batch = batch.to(device)
                pred = model(batch)
                pred_list = unbatch(pred, batch.batch)
                yad_id_list = unbatch(batch.subset_node_idx, batch.batch)
                for session_id, yad_id, pred_i in zip(batch.session_id, yad_id_list, pred_list):
                    preds.append(pred_i.detach().cpu().numpy().tolist())
                    yad_ids.append(yad_id.detach().cpu().numpy().tolist())
                    session_id_list = [session_id] * len(pred_i)
                    session_ids.append(session_id_list)

    # flatten
    preds = [item for sublist in preds for item in sublist]
    yad_ids = [item for sublist in yad_ids for item in sublist]
    session_ids = [item for sublist in session_ids for item in sublist]
    # make df
    df = pl.DataFrame({"session_id": session_ids, "yad_no": yad_ids, "score": preds})

    return df


def make_sub_df(oof_df: pl.DataFrame) -> pd.DataFrame:
    agg_oof_df = oof_df.group_by("session_id").agg(
        pl.col(["yad_no", "score"]).sort_by("score", descending=True).head(10)
    )
    sub_df = pd.DataFrame(
        index=agg_oof_df.get_column("session_id").to_list(),
        data=agg_oof_df.get_column("yad_no").to_list(),
    ).add_prefix("predict_")
    sub_df = sub_df.fillna("1")  # 1埋め
    # sort
    sub_df = sub_df.sort_index()

    return sub_df


@hydra.main(config_path="conf", config_name="inference")
def main(cfg: InferenceConfig):
    seed_everything(cfg.seed)
    processed_dir = Path(cfg.dir.processed_dir)
    # load train config
    train_output_dir = Path("../output") / "train"
    train_cfg = load_train_cfg(train_output_dir, cfg.train_exp_name, cfg.train_run_name)

    # load data
    G = torch.load(processed_dir / "graph.pt")
    log_df = load_log_df(processed_dir, cfg.phase, train_cfg.fold)

    # load model
    deg = torch.load(processed_dir / "deg" / f"deg_k{train_cfg.k}.pt")
    model = YadGNN(
        num_node_features=G.x.shape[1] - 4 + 5,
        num_edge_features=4,
        num_layers=train_cfg.model.num_layers,
        mid_dim=train_cfg.model.mid_dim,
        dropout_rate=train_cfg.model.dropout_rate,
        norm_type=train_cfg.model.norm_type,
        edge_dropout_rate=train_cfg.model.edge_dropout_rate,
        conv_type=train_cfg.model.conv_type,
        conv_params=dict(train_cfg.model.conv_params),
        deg=deg,
    )
    weight_path = (
        Path(cfg.dir.output_dir)
        / "train"
        / cfg.train_exp_name
        / cfg.train_run_name
        / "best_model.pth"
    )
    model.load_state_dict(torch.load(weight_path))

    # load dataset
    name = get_dataset_name(cfg.phase, train_cfg.fold)
    dl = get_dataloader(
        processed_dir,
        G,
        log_df,
        train_cfg.k,
        train_cfg.dataset.batch_size,
        train_cfg.dataset.num_workers,
        name,
    )

    # inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = inference(model, dl, device, cfg.use_amp)
    # yad_idを0indexにしているため、1足す
    df = df.with_columns(pl.col("yad_no") + 1)

    # save
    df.write_csv(f"oof_{cfg.phase}.csv")

    # make submission
    sub_df = make_sub_df(df)
    sub_df.to_csv(f"submission_{cfg.phase}.csv", index=False)


if __name__ == "__main__":
    main()
