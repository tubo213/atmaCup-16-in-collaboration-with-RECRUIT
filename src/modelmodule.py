import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Batch
from torch_geometric.utils import degree
from transformers import get_cosine_schedule_with_warmup

from src.conf import TrainConfig
from src.model import YadGNN
from src.utils import mapk


def pairwise_hinge_loss(y_pred: torch.Tensor, y_true: torch.Tensor, margin: float = 1.0):
    """
    y_pred: (n)
    y_true: (n)
    """
    weight = y_true.unsqueeze(0) > y_true.unsqueeze(1)  # (n, n)
    loss = F.relu(margin - (y_pred.unsqueeze(0) - y_pred.unsqueeze(1))) * weight

    return loss.sum() / (weight.sum() + 1e-6)


def compute_loss(logits: torch.Tensor, data: Batch) -> torch.Tensor:
    sizes = degree(data.batch, dtype=torch.long).tolist()

    logit_list: list[torch.Tensor] = logits.split(sizes)
    target_list: list[torch.Tensor] = data.y.split(sizes)

    loss = 0
    for y_pred, y_true in zip(logit_list, target_list):
        # bce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true)  # type: ignore
        loss += pairwise_hinge_loss(y_pred, y_true)
        # loss += bce_loss + 0.5 * hinge_loss

    return loss / data.num_graphs


def decode(logits: torch.Tensor, data: Batch) -> torch.Tensor:
    sizes = degree(data.batch, dtype=torch.long).tolist()

    logit_list = logits.split(sizes)
    subset_node_idx_list = data.subset_node_idx.split(sizes)

    preds = []
    for y_pred, subset_node_idx in zip(logit_list, subset_node_idx_list):
        prob = y_pred
        # 確率が大きい上位10個を取得
        arg_idx = torch.argsort(prob, descending=True)
        arg_topk = arg_idx[:10]
        topk_node_idx = subset_node_idx[arg_topk]
        # 10個未満の場合は0を追加
        if len(topk_node_idx) < 10:
            topk_node_idx = torch.cat(
                [
                    topk_node_idx,
                    torch.zeros(
                        10 - len(topk_node_idx), dtype=torch.long, device=topk_node_idx.device
                    ),
                ]
            )

        preds.append(topk_node_idx)

    return torch.cat(preds)


class PLYadModel(pl.LightningModule):
    def __init__(
        self, cfg: TrainConfig, num_node_features: int, num_edge_features: int, deg: torch.Tensor
    ):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model = YadGNN(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            num_layers=cfg.model.num_layers,
            mid_dim=cfg.model.mid_dim,
            dropout_rate=cfg.model.dropout_rate,
            edge_dropout_rate=cfg.model.edge_dropout_rate,
            conv_type=cfg.model.conv_type,
            conv_params=dict(cfg.model.conv_params),
            deg=deg,
        )
        self.validation_step_outputs: list = []
        self.__best_score = 0

    def forward(self, data: Batch) -> torch.Tensor:
        return self.model(data)

    def training_step(self, batch, batch_idx):
        logits = self.model(batch)
        loss = compute_loss(logits, batch)

        self.log(
            "train_loss",
            loss.detach().item(),
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            batch_size=batch.num_graphs,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self.model(batch)
        loss = compute_loss(logits, batch)
        preds = decode(logits, batch)
        self.validation_step_outputs.append(
            (
                batch.label.detach().cpu().numpy(),
                preds.detach().cpu().numpy(),
                loss.detach().cpu().numpy(),
            )
        )
        self.log(
            "val_loss",
            loss.detach().item(),
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            batch_size=batch.num_graphs,
        )

        return loss

    def on_validation_epoch_end(self):
        labels = np.concatenate([x[0] for x in self.validation_step_outputs]).reshape(-1)
        preds = np.concatenate([x[1] for x in self.validation_step_outputs]).reshape(-1, 10)

        score = mapk(labels.tolist(), preds.tolist(), k=10)
        self.log(
            "val_score",
            score,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )

        if score > self.__best_score:
            np.save("labels.npy", labels)
            np.save("preds.npy", preds)
            torch.save(self.model.state_dict(), "best_model.pth")
            print(f"Saved best model {self.__best_score} -> {score}")
            self.__best_score = score
            self.log("best_score", self.__best_score, on_step=False, on_epoch=True, logger=True)

        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.cfg.optimizer.lr)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_training_steps=self.trainer.max_steps, **self.cfg.scheduler
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
