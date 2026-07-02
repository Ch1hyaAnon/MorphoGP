from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from morphogp.contourcluster import ContourClusterConfig, run_contourcluster


def plot_shapelet_interpretation(result: dict[str, object], output_dir: Path) -> None:
    profiles = result["profiles"]
    candidates = result["candidates"]
    reps = result["representative_indices"]
    matches = result["matches"]
    n_reps = len(reps)
    ncols = 2
    nrows = int(np.ceil(n_reps / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.2 * nrows), squeeze=False)
    for shapelet_cluster, candidate_index in enumerate(reps):
        ax = axes[shapelet_cluster // ncols][shapelet_cluster % ncols]
        candidate = candidates[candidate_index]
        best = (
            matches[matches["shapelet_cluster"] == shapelet_cluster]
            .sort_values("similarity", ascending=False)
            .iloc[0]
        )
        profile = next(p for p in profiles if p.profile_id == best["profile_id"])
        start = int(best["best_start"])
        stop = start + candidate.length
        ax.plot(profile.x_grid, profile.y_norm, color="0.25", linewidth=1.8)
        ax.plot(
            profile.x_grid[start:stop],
            profile.y_norm[start:stop],
            color="#D62728",
            linewidth=3.0,
        )
        ax.set_title(
            f"Shapelet {shapelet_cluster}: {best['profile_id']} "
            f"(sim={best['similarity']:.3f})"
        )
        ax.set_xlabel("Normalized cross-shore position")
        ax.set_ylabel("Normalized elevation")
        ax.grid(alpha=0.25)
    for idx in range(n_reps, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_shapelet_interpretation.png", dpi=220)
    fig.savefig(output_dir / "fig_shapelet_interpretation.pdf")
    plt.close(fig)


def plot_profile_clusters(result: dict[str, object], output_dir: Path) -> None:
    profiles = result["profiles"]
    labels = result["profile_labels"]
    unique = sorted(np.unique(labels))
    fig, axes = plt.subplots(len(unique), 1, figsize=(9, 2.8 * len(unique)), squeeze=False)
    for row, label in enumerate(unique):
        ax = axes[row][0]
        selected = [p for p, kappa in zip(profiles, labels) if kappa == label]
        ys = np.vstack([p.y_norm for p in selected])
        for p in selected:
            ax.plot(p.x_grid, p.y_norm, color="#4C78A8", alpha=0.18, linewidth=1.0)
        ax.fill_between(
            selected[0].x_grid,
            np.quantile(ys, 0.10, axis=0),
            np.quantile(ys, 0.90, axis=0),
            color="#4C78A8",
            alpha=0.16,
            linewidth=0,
        )
        ax.plot(selected[0].x_grid, ys.mean(axis=0), color="black", linewidth=2.2)
        ax.set_title(f"Morphology category kappa={label} (n={len(selected)})")
        ax.set_xlabel("Normalized cross-shore position")
        ax.set_ylabel("Normalized elevation")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_profile_clusters.png", dpi=220)
    fig.savefig(output_dir / "fig_profile_clusters.pdf")
    plt.close(fig)


def plot_embedding(output_dir: Path) -> None:
    frame = pd.read_csv(output_dir / "candidate_embedding_pca.csv")
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        frame["pc1"],
        frame["pc2"],
        c=frame["shapelet_cluster"],
        s=8,
        alpha=0.45,
        cmap="tab10",
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Shapelet embedding PCA")
    ax.grid(alpha=0.25)
    fig.colorbar(scatter, ax=ax, label="Shapelet cluster")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_embedding_umap_or_pca.png", dpi=220)
    fig.savefig(output_dir / "fig_embedding_umap_or_pca.pdf")
    plt.close(fig)


def write_run_summary(result: dict[str, object], output_dir: Path, mode: str) -> None:
    profile_metrics = result["profile_metrics"]
    train_metrics = result["train_metrics"]
    summary = [
        f"# ContourCluster run summary ({mode})",
        "",
        f"- valid profiles: {len(result['profiles'])}",
        f"- skipped profiles: {len(result['skipped'])}",
        f"- shapelet candidates: {len(result['candidates'])}",
        f"- representative shapelets: {len(result['representative_indices'])}",
        f"- selected profile K: {result['selected_k']}",
        f"- profile clustering: {train_metrics.get('profile_clustering', 'n/a')}",
        f"- final triplet loss: {train_metrics['triplet_loss_final']:.6f}",
        "",
        "## K selection",
        "",
        profile_metrics.to_markdown(index=False),
    ]
    (output_dir / "run_summary.md").write_text("\n".join(summary), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MorphoGP ContourCluster.")
    parser.add_argument("--data-dir", default="Datasets")
    parser.add_argument("--output-dir", default="results/contourcluster")
    parser.add_argument("--mode", choices=["baseline", "enhanced"], default="baseline")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--steps-per-epoch", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-shapelets", type=int, default=8)
    parser.add_argument("--profile-length", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=24)
    parser.add_argument("--profile-k", type=int, default=4)
    parser.add_argument(
        "--profile-clustering",
        choices=["kmeans", "balanced_kmeans"],
        default="balanced_kmeans",
    )
    parser.add_argument("--min-cluster-fraction", type=float, default=0.08)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)
    output_dir = Path(args.output_dir)
    config = ContourClusterConfig(
        profile_length=args.profile_length,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        n_shapelets=args.n_shapelets,
        d_model=args.d_model,
        embedding_dim=args.embedding_dim,
        profile_k=None if args.profile_k <= 0 else args.profile_k,
        profile_clustering=args.profile_clustering,
        min_cluster_fraction=args.min_cluster_fraction,
    )
    result = run_contourcluster(
        data_dir=args.data_dir,
        output_dir=output_dir,
        config=config,
        mode=args.mode,
        limit=args.limit,
        device=args.device,
    )
    plot_shapelet_interpretation(result, output_dir)
    plot_profile_clusters(result, output_dir)
    plot_embedding(output_dir)
    write_run_summary(result, output_dir, args.mode)
    print(f"saved outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
