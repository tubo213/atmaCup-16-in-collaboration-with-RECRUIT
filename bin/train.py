import logging
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers import WandbLogger

from src.datamodule import YadDataModule
from src.modelmodule import PLYadModel
from src.utils import flatten_dict

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s:%(name)s - %(message)s"
)
LOGGER = logging.getLogger(Path(__file__).name)


@hydra.main(config_path="conf", config_name="train", version_base="1.2")
def main(cfg):
    seed_everything(cfg.seed)

    # init lightning model
    datamodule = YadDataModule(cfg)
    deg = torch.load(datamodule.processed_dir / "deg" / f"deg_k{cfg.k}.pt")
    LOGGER.info("Set Up DataModule")
    model = PLYadModel(
        cfg, num_node_features=datamodule.G.x.shape[1] - 4 + 5, num_edge_features=4, deg=deg
    )
    # node: -4 + 4 # 4: category, 5: is_last, is_visited, order_of_visit, is_odd, visit_cnt
    # edge: 4: number_of_count, step in, step out, seq_edge

    # set callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath="checkpoints",
        verbose=True,
        monitor=cfg.trainer.monitor,
        mode=cfg.trainer.monitor_mode,
        save_top_k=1,
        save_last=False,
    )
    lr_monitor = LearningRateMonitor("epoch")
    progress_bar = RichProgressBar()
    model_summary = RichModelSummary(max_depth=2)

    # init experiment logger
    run_name = Path.cwd().name
    pl_logger = WandbLogger(
        group=cfg.exp_name,
        name=run_name,
        project="atmaCup-16-In-Collaboration-with-RECRUIT",
    )
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb_cfg = flatten_dict(wandb_cfg)  # 再帰的にdictを展開
    pl_logger.log_hyperparams(wandb_cfg)

    trainer = Trainer(
        # env
        default_root_dir=Path.cwd(),
        # num_nodes=cfg.training.num_gpus,
        accelerator=cfg.trainer.accelerator,
        precision=16 if cfg.trainer.use_amp else 32,
        # training
        fast_dev_run=cfg.trainer.debug,  # run only 1 train batch and 1 val batch
        max_epochs=cfg.trainer.epochs,
        max_steps=cfg.trainer.epochs * len(datamodule.train_dataloader()),
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        callbacks=[checkpoint_cb, lr_monitor, progress_bar, model_summary],
        logger=pl_logger,
        # resume_from_checkpoint=resume_from,
        num_sanity_val_steps=0,
        # sync_batchnorm=True
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
