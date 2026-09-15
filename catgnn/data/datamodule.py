import pytorch_lightning as pl
from torch.utils.data import DataLoader

from catgnn.data.dataset import SFFSDGraphDataset
from catgnn.data.features import build_features


class FraudGraphDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for S-FFSD fraud graph."""

    def __init__(
        self,
        data_path: str = "S-FFSD.csv",
        max_neighbors: int = 20,
        train_ratio: float = 0.7,
        batch_size: int = 128,
        num_workers: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_dict = None

    def setup(self, stage: str | None = None):
        if self.data_dict is None:
            self.data_dict = build_features(self.hparams.data_path)
            self.cont_dim = self.data_dict["cont_dim"]
            self.cat_dims = self.data_dict["cat_dims"]

            labeled_idx = self.data_dict["labeled_indices"]
            num_labeled = len(labeled_idx)
            train_end = int(num_labeled * self.hparams.train_ratio)

            self.train_idx = labeled_idx[:train_end]
            self.test_idx = labeled_idx[train_end:]

            train_labels = self.data_dict["Y"][self.train_idx]
            num_pos = train_labels.sum().item()
            num_neg = len(train_labels) - num_pos
            self.pos_weight = num_neg / max(num_pos, 1.0)

        self.train_ds = SFFSDGraphDataset(
            self.data_dict, self.train_idx, self.hparams.max_neighbors
        )
        self.test_ds = SFFSDGraphDataset(
            self.data_dict, self.test_idx, self.hparams.max_neighbors
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )
