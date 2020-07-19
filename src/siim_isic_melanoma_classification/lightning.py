from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from pytorch_lightning import Trainer
from pytorch_lightning.core import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from .config import Config
from .util import clean_up, get_device, to_device

try:
    import apex

    has_apex = True
except ImportError:
    has_apex = False

from .config import Config


class LightningModelBase(LightningModule):
    def __init__(self, config, **kwargs):
        super().__init__()
        config_dict = asdict(config) if isinstance(config, Config) else config
        self.save_hyperparameters(config_dict)

    def loss(self, y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y.float())

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # forward pass
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.logger.log_metrics({"loss": loss})
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        x, y = batch
        y_hat = self(x)
        val_loss = self.loss(y_hat, y)
        y_pred = torch.round(torch.sigmoid(y_hat))
        n_correct_pred = torch.sum(y == y_pred).item()
        return {
            "val_loss": val_loss,
            "n_correct_pred": n_correct_pred,
            "n_pred": len(x),
        }

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = self.loss(y_hat, y)
        labels_hat = torch.argmax(y_hat, dim=1)
        n_correct_pred = torch.sum(y == labels_hat).item()
        return {
            "test_loss": test_loss,
            "n_correct_pred": n_correct_pred,
            "n_pred": len(x),
        }

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        val_acc = sum([x["n_correct_pred"] for x in outputs]) / sum(
            x["n_pred"] for x in outputs
        )
        self.logger.log_metrics({"val_loss": avg_loss})
        return {"val_loss": avg_loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        test_acc = sum([x["n_correct_pred"] for x in outputs]) / sum(
            x["n_pred"] for x in outputs
        )
        tensorboard_logs = {"test_loss": avg_loss, "test_acc": test_acc}
        return {"test_loss": avg_loss, "log": tensorboard_logs}

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


class Trainer(Trainer):
    def __init__(self, config: Config, **kwargs):
        if torch.cuda.is_available():
            kwargs["gpus"] = config.gpus
            kwargs["precision"] = config.precision if has_apex else 32

        if "checkpoint_callback" not in kwargs:
            kwargs["checkpoint_callback"] = ModelCheckpoint(
                filepath=str(Path.cwd()), verbose=True
            )
        if "early_stop_callback" not in kwargs:
            kwargs["early_stop_callback"] = EarlyStopping(
                patience=config.early_stop_patience, verbose=True
            )
        super().__init__(max_epochs=config.max_epochs, **kwargs)
        self._checkpoint_callback = kwargs["checkpoint_callback"]

    @property
    def best_model_path(self):
        return self._checkpoint_callback.best_model_path


class Classifier:
    def __init__(self, model, device=None):
        self.device = get_device() if device is None else device
        self.model = model.eval().to(self.device)

    def predict(self, data_loader):
        clean_up()

        result = []
        with torch.no_grad():
            for x in data_loader:
                logits = self.model(to_device(x))
                y_hat = torch.sigmoid(logits)
                result.append(y_hat.cpu().numpy())
        return np.hstack(result)
