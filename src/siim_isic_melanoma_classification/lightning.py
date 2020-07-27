import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

# from sklearn.metrics import accuracy_score, roc_auc_score
import torch
import torch.nn.functional as F
from torch import optim
import torch.distributed as dist
from pytorch_lightning import Trainer
from pytorch_lightning.core import LightningModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from pytorch_lightning.metrics.classification import AUROC

from .config import Config
from .util import clean_up, get_device, to_device, is_apex_available, is_tpu_available
from .comm import all_gather

import gc

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
except ImportError:
    pass


class LightningModelBase(LightningModule):
    def __init__(self, config, **kwargs):
        super().__init__()
        config_dict = asdict(config) if isinstance(config, Config) else config
        self.save_hyperparameters(config_dict)
        self.test_results = []

    def loss(self, y_hat, y):
        # y_smo = y.float() * (1 - label_smoothing) + 0.5 * label_smoothing
        # F.binary_cross_entropy_with_logits(y_hat, y_smo.type_as(y_hat),
        #                                           pos_weight=torch.tensor(pos_weight))
        return F.binary_cross_entropy_with_logits(y_hat, y.float())

    def step(self, batch):
        # return batch loss
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        return loss, y, y_hat.sigmoid()

    def training_step(self, batch, batch_idx):
        # hardware agnostic training
        loss, y, y_hat = self.step(batch)
        acc = (y_hat.round() == y).float().mean().item()
        self.logger.log_metrics({"loss": loss, "acc": acc})
        return {"loss": loss, "acc": acc}

    def validation_step(self, batch, batch_idx):
        loss, _, _ = self.step(batch)
        #return {"val_loss": loss, "y": y.detach(), "y_hat": y_hat.detach()}
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        #y = torch.cat([x["y"] for x in outputs])
        #y_hat = torch.cat([x["y_hat"] for x in outputs])

        #tmp = torch.cat([y_hat, y.float()], 1)
        #print(f"before tmp:{tmp.shape}")
        #tmp = all_gather(tmp)
        #print(f"after tmp:{tmp.shape}")

        #auc = (
        #    AUROC()(pred=tmp[:, 0], target=tmp[:, 1]) if y.float().mean() > 0 else 0.5
        #)  # skip sanity check
        #acc = (tmp[:, 0].round() == tmp[:, 1]).float().mean().item()
        #self.logger.log_metrics({"val_loss": avg_loss, "val_auc": auc, "val_acc": acc})
        #return {"val_loss": avg_loss, "val_auc": auc, "val_acc": acc}
        self.logger.log_metrics({"val_loss": avg_loss})
        return {"val_loss": avg_loss}

    def test_step(self, batch, batch_nb):
        x = batch
        y_hat = self(x).sigmoid()
        return {"y_hat": y_hat}

    def test_epoch_end(self, outputs):
        y_hat = torch.stack([x["y_hat"] for x in outputs]).flatten()
        self.test_results.append(y_hat.detach())
        return {}

    def configure_optimizers(self):
        """
        Return whatever optimizers and learning rate schedulers you want here.
        At least one optimizer is required.
        """
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10, eta_min=0.001
        )
        return [optimizer], [scheduler]

    def setup(self, stage):
        if is_tpu_available() and isinstance(self.logger, WandbLogger):
            self.logger._name = self.logger.experiment.name
            self.logger._offset = True
            self.logger._experiment = None
        clean_up()


class WorkaroundEarlyStopping(EarlyStopping):
    def _stop_distributed_training(self, trainer, pl_module):
        if trainer.use_ddp or trainer.use_ddp2:
            stop = torch.tensor(int(trainer.should_stop), device=pl_module.device)
            dist.all_reduce(stop, op=dist.reduce_op.SUM)
            dist.barrier()
            trainer.should_stop = stop == trainer.world_size

        if trainer.use_tpu:
            stop = torch.tensor(
                int(trainer.should_stop), device=pl_module.device, dtype=torch.int32
            )
            stop = xm.mesh_reduce("stop_signal", stop, lambda xs: torch.stack(xs).sum())
            torch_xla.core.xla_model.rendezvous(
                "pl.EarlyStoppingCallback.stop_distributed_training_check"
            )
            trainer.should_stop = int(stop.item()) == trainer.world_size


class Trainer(Trainer):
    def __init__(self, config: Config, **kwargs):
        if torch.cuda.is_available():
            kwargs["gpus"] = config.gpus
            kwargs["precision"] = config.precision if is_apex_available() else 32

        if is_tpu_available():
            kwargs["tpu_cores"] = config.tpus
            kwargs["precision"] = config.precision

        if "checkpoint_callback" not in kwargs:
            kwargs["checkpoint_callback"] = ModelCheckpoint(
                filepath=str(Path.cwd() / "{epoch}-{val_roc:.2f}"),
                verbose=True,
                #                monitor="val_roc",
                #                mode="max",
            )
        if "early_stop_callback" not in kwargs:
            kwargs["early_stop_callback"] = WorkaroundEarlyStopping(
                patience=config.early_stop_patience,
                verbose=True,
                #                monitor="val_roc",
                #                mode="max",
            )

        super().__init__(max_epochs=config.max_epochs, **kwargs)
        self._checkpoint_callback = kwargs["checkpoint_callback"]

    @property
    def best_model_path(self):
        return self._checkpoint_callback.best_model_path


class Classifier:
    def __init__(self, model, tta_epochs=1, device=None):
        self.tta_epochs = tta_epochs
        self.device = get_device() if device is None else device
        self.model = model.eval().to(self.device)

    def predict(self, data_loader):
        clean_up()

        with torch.no_grad():
            return (
                torch.stack(
                    [self._predict_once(data_loader) for _ in range(self.tta_epochs)]
                )
                .mean(0)
                .cpu()
                .numpy()
            )

    def _predict_once(self, data_loader):
        result = [torch.sigmoid(self.model(to_device(x))) for x in data_loader]
        return torch.cat(result)
