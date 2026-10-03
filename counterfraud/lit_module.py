import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
from sklearn.metrics import f1_score

from counterfraud.losses import CompositeLoss
from counterfraud.model import CaTGNN


class CaTGNNLightningModule(pl.LightningModule):
    def __init__(
        self,
        cont_dim: int,
        cat_dims: list[int],
        emb_dim: int = 16,
        hidden_dim: int = 64,
        heads: int = 4,
        time_dim: int = 16,
        num_relations: int = 4,
        env_ratio: float = 0.2,
        top_k: int = 5,
        beta_alpha: float = 2.0,
        beta_beta: float = 2.0,
        mlp_hidden: int = 64,
        dropout: float = 0.1,
        base_loss: str = "focal",
        gamma: float = 1.0,
        warmup_epochs: int = 3,
        eta: float = 0.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        pos_weight: float | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = CaTGNN(
            cont_dim=cont_dim,
            cat_dims=cat_dims,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            heads=heads,
            time_dim=time_dim,
            num_relations=num_relations,
            env_ratio=env_ratio,
            top_k=top_k,
            beta_alpha=beta_alpha,
            beta_beta=beta_beta,
            mlp_hidden=mlp_hidden,
            dropout=dropout,
        )

        self.loss_fn = CompositeLoss(
            base_loss=base_loss,
            gamma=gamma,
            eta=eta,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            pos_weight=pos_weight,
        )

        self.val_auroc = torchmetrics.AUROC(task="binary")
        self.val_auprc = torchmetrics.AveragePrecision(task="binary")

        self.test_auroc = torchmetrics.AUROC(task="binary")
        self.test_auprc = torchmetrics.AveragePrecision(task="binary")

        self.val_probs: list[torch.Tensor] = []
        self.val_targets: list[torch.Tensor] = []
        self.test_probs: list[torch.Tensor] = []
        self.test_targets: list[torch.Tensor] = []

    def current_gamma(self) -> float:
        """Linear warmup for causal-invariant loss weight: gamma_t = gamma * (t / warmup)."""
        warmup = self.hparams.warmup_epochs
        if warmup <= 0 or self.current_epoch >= warmup:
            return self.hparams.gamma
        return self.hparams.gamma * (self.current_epoch / max(1, warmup))

    def shared_step(self, batch, intervene: bool):
        x_target_cont, x_target_cat, x_neighbor_cont, x_neighbor_cat, t_gap, rel_ids, valid_mask, y = batch

        out = self.model(
            x_target_cont=x_target_cont,
            x_target_cat=x_target_cat,
            x_neighbor_cont=x_neighbor_cont,
            x_neighbor_cat=x_neighbor_cat,
            t_gap=t_gap,
            valid_mask=valid_mask,
            rel_ids=rel_ids,
            intervene=intervene,
        )

        self.loss_fn.gamma = self.current_gamma() if intervene else self.hparams.gamma

        loss, components = self.loss_fn(
            out["logits"],
            y,
            logits_intervened=out.get("logits_intervened"),
            model_parameters=self.model.parameters(),
        )
        probs = torch.sigmoid(out["logits"])
        return loss, components, probs, y

    def training_step(self, batch, batch_idx):
        loss, components, _, y = self.shared_step(batch, intervene=True)
        self.log_dict(
            {f"train/{k}": v for k, v in components.items()},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=y.size(0),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, components, probs, y = self.shared_step(batch, intervene=False)
        y_int = y.long()
        self.val_auroc.update(probs, y_int)
        self.val_auprc.update(probs, y_int)
        self.val_probs.append(probs.detach().cpu())
        self.val_targets.append(y_int.detach().cpu())
        self.log_dict(
            {f"val/{k}": v for k, v in components.items()},
            on_epoch=True,
            batch_size=y.size(0),
        )
        return loss

    @staticmethod
    def _calibrated_f1(probs: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
        """Threshold search for optimal Macro-F1."""
        best_f1, best_thresh = 0.0, 0.5
        thresholds = np.linspace(0.02, 0.98, 97)
        for t in thresholds:
            score = float(f1_score(targets, (probs >= t).astype(int), average="macro", zero_division=0))
            if score > best_f1:
                best_f1 = score
                best_thresh = float(t)
        return best_f1, best_thresh

    def on_validation_epoch_end(self):
        auroc = self.val_auroc.compute()
        auprc = self.val_auprc.compute()
        self.log("val/auroc", auroc, prog_bar=True)
        self.log("val/auprc", auprc, prog_bar=True)

        if self.val_probs:
            probs = torch.cat(self.val_probs).numpy()
            targets = torch.cat(self.val_targets).numpy()
            best_f1, best_thresh = self._calibrated_f1(probs, targets)
            self.log("val/f1_macro", best_f1, prog_bar=True)
            self.log("val/threshold", best_thresh)

        self.val_auroc.reset()
        self.val_auprc.reset()
        self.val_probs.clear()
        self.val_targets.clear()

    def test_step(self, batch, batch_idx):
        loss, components, probs, y = self.shared_step(batch, intervene=False)
        y_int = y.long()
        self.test_auroc.update(probs, y_int)
        self.test_auprc.update(probs, y_int)
        self.test_probs.append(probs.detach().cpu())
        self.test_targets.append(y_int.detach().cpu())
        self.log_dict(
            {f"test/{k}": v for k, v in components.items()},
            on_epoch=True,
            batch_size=y.size(0),
        )
        return loss

    def on_test_epoch_end(self):
        auroc = self.test_auroc.compute()
        auprc = self.test_auprc.compute()
        self.log("test/auroc", auroc)
        self.log("test/auprc", auprc)

        if self.test_probs:
            probs = torch.cat(self.test_probs).numpy()
            targets = torch.cat(self.test_targets).numpy()
            best_f1, best_thresh = self._calibrated_f1(probs, targets)
            self.log("test/f1_macro", best_f1)
            self.log("test/threshold", best_thresh)

        self.test_auroc.reset()
        self.test_auprc.reset()
        self.test_probs.clear()
        self.test_targets.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-5,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }