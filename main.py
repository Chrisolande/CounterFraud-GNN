"""CaT-GNN: Causal Temporal Graph Neural Network for Fraud Detection.

Main execution entrypoint for training and evaluating CaT-GNN across single or
multiple random seeds with comprehensive metric aggregation (AUPRC, AUROC, Macro-F1).
"""

import argparse

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from catgnn.data import FraudGraphDataModule
from catgnn.lit_module import CaTGNNLightningModule


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CaT-GNN: Multi-Seed Training & Benchmark Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Dataset and Graph Parameters
    parser.add_argument("--data_path", type=str, default="S-FFSD.csv", help="Path to S-FFSD CSV dataset")
    parser.add_argument("--max_neighbors", type=int, default=20, help="Max historical neighbors per node")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Chronological train split ratio")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker processes")

    # Seeds and Training Configuration
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 100, 2024, 777, 888],
        help="List of random seeds to run for benchmark aggregation",
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Mini-batch size")
    parser.add_argument("--max_epochs", type=int, default=20, help="Maximum training epochs per seed")
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate for AdamW")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for regularization")
    parser.add_argument("--grad_clip_val", type=float, default=1.0, help="Gradient clipping maximum norm")

    # Architecture Hyperparameters
    parser.add_argument("--hidden_dim", type=int, default=64, help="GNN hidden representation dimension")
    parser.add_argument("--emb_dim", type=int, default=16, help="Categorical entity embedding dimension")
    parser.add_argument("--heads", type=int, default=4, help="Attention heads in TemporalAttentionLayer")
    parser.add_argument("--time_dim", type=int, default=16, help="Harmonic Time2Vec encoding dimension")
    parser.add_argument("--num_relations", type=int, default=4, help="Number of graph edge relations")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")

    # Causal & Invariant Loss Hyperparameters
    parser.add_argument("--env_ratio", type=float, default=0.2, help="Ratio of lowest-attention slots marked as environment")
    parser.add_argument("--top_k", type=int, default=5, help="Top-k causal candidate donors for Beta-mixup")
    parser.add_argument("--beta_alpha", type=float, default=2.0, help="Beta distribution alpha parameter")
    parser.add_argument("--beta_beta", type=float, default=2.0, help="Beta distribution beta parameter")
    parser.add_argument("--base_loss", type=str, default="focal", choices=["focal", "weighted_bce"], help="Base task loss")
    parser.add_argument("--gamma", type=float, default=1.0, help="Weight on causal-invariant consistency loss")
    parser.add_argument("--warmup_epochs", type=int, default=3, help="Warmup epochs to linearly scale gamma")
    parser.add_argument("--eta", type=float, default=0.0, help="Explicit L2 weight penalty")

    # Execution Options
    parser.add_argument("--export_csv", type=str, default="multi_seed_results.csv", help="CSV path for output summary")
    parser.add_argument("--fast_dev_run", action="store_true", help="Run 1 train/val/test batch for sanity check")
    return parser


def run_benchmark(args: argparse.Namespace) -> pd.DataFrame:
    """Run full benchmark across configured seeds and report mean ± std."""
    print("=" * 70)
    print(" CaT-GNN (Causal Temporal GNN) Benchmark Execution")
    print("=" * 70)
    print(f"Dataset Path      : {args.data_path}")
    print(f"Evaluation Seeds  : {args.seeds}")
    print(f"Max Epochs        : {args.max_epochs}")
    print(f"Batch Size        : {args.batch_size}")
    print(f"Hidden Dim / Heads: {args.hidden_dim} / {args.heads}")
    print(f"Causal Parameters : env_ratio={args.env_ratio}, top_k={args.top_k}, gamma={args.gamma}")
    print("=" * 70)

    # 1. Setup shared preprocessed DataModule
    datamodule = FraudGraphDataModule(
        data_path=args.data_path,
        max_neighbors=args.max_neighbors,
        train_ratio=args.train_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    datamodule.setup()
    print(
        f"Graph Prepared | Continuous Dim: {datamodule.cont_dim} | "
        f"Categorical Classes: {datamodule.cat_dims} | "
        f"Pos Weight: {datamodule.pos_weight:.2f} | "
        f"Train: {len(datamodule.train_ds):,} | Test: {len(datamodule.test_ds):,}"
    )

    records: list[dict[str, float | int]] = []

    for idx, seed in enumerate(args.seeds):
        print("\n" + "-" * 70)
        print(f" Running Trial [{idx + 1}/{len(args.seeds)}] with Seed = {seed}")
        print("-" * 70)

        pl.seed_everything(seed, workers=True)

        module = CaTGNNLightningModule(
            cont_dim=datamodule.cont_dim,
            cat_dims=datamodule.cat_dims,
            emb_dim=args.emb_dim,
            hidden_dim=args.hidden_dim,
            heads=args.heads,
            time_dim=args.time_dim,
            num_relations=args.num_relations,
            env_ratio=args.env_ratio,
            top_k=args.top_k,
            beta_alpha=args.beta_alpha,
            beta_beta=args.beta_beta,
            dropout=args.dropout,
            base_loss=args.base_loss,
            gamma=args.gamma,
            warmup_epochs=args.warmup_epochs,
            eta=args.eta,
            pos_weight=datamodule.pos_weight,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        callbacks = [
            ModelCheckpoint(monitor="val/auprc", mode="max", save_top_k=1, filename=f"catgnn-seed{seed}-{{epoch:02d}}-{{val/auprc:.4f}}"),
            EarlyStopping(monitor="val/auprc", mode="max", patience=8),
            LearningRateMonitor(logging_interval="epoch"),
        ]

        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            accelerator="auto",
            devices="auto",
            gradient_clip_val=args.grad_clip_val,
            callbacks=callbacks,
            enable_progress_bar=True,
            fast_dev_run=args.fast_dev_run,
            log_every_n_steps=10,
        )

        trainer.fit(module, datamodule=datamodule)
        test_results = trainer.test(
            module,
            datamodule=datamodule,
            ckpt_path="best" if not args.fast_dev_run else None,
        )[0]

        record = {
            "Seed": seed,
            "AUPRC": test_results.get("test/auprc", 0.0),
            "AUROC": test_results.get("test/auroc", 0.0),
            "Macro-F1": test_results.get("test/f1_macro", 0.0),
            "Threshold": test_results.get("test/threshold", 0.5),
        }
        records.append(record)
        print(f"Trial Seed {seed} Final Test: AUPRC = {record['AUPRC']:.4f} | AUROC = {record['AUROC']:.4f} | Macro-F1 = {record['Macro-F1']:.4f} (Threshold = {record['Threshold']:.3f})")

    df = pd.DataFrame(records)
    print("\n" + "=" * 70)
    print(" CaT-GNN MULTI-SEED EXPERIMENT SUMMARY")
    print("=" * 70)
    print(df.to_string(index=False))

    print("\nAGGREGATED BENCHMARK METRICS (Mean ± Std):")
    for metric in ["AUPRC", "AUROC", "Macro-F1"]:
        mean_val = df[metric].mean()
        std_val = df[metric].std() if len(df) > 1 else 0.0
        print(f" - {metric:10s}: {mean_val:.4f} ± {std_val:.4f}")
    print("=" * 70)

    if args.export_csv:
        df.to_csv(args.export_csv, index=False)
        print(f"Summary table exported to: {args.export_csv}")

    return df


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()