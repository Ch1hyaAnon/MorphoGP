"""ContourCluster: morphology-driven profile clustering for MorphoGP.

The module implements the first stage of MorphoGP.  It learns local beach
shapelets, converts complete profiles into shapelet-similarity features, and
clusters complete profiles into morphology categories ``kappa`` for downstream
category-specific GP experts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F


TripletMode = Literal["baseline", "enhanced"]
ProfileClustering = Literal["kmeans", "balanced_kmeans"]


@dataclass(frozen=True)
class ContourClusterConfig:
    profile_length: int = 128
    candidate_lengths: tuple[int, ...] = (16, 24, 32)
    shapelet_points: int = 32
    min_profile_points: int = 5
    n_shapelets: int = 8
    k_range: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8)
    gamma: float = 3.0
    d_model: int = 48
    nhead: int = 4
    num_layers: int = 2
    embedding_dim: int = 32
    margin: float = 0.30
    batch_size: int = 256
    epochs: int = 8
    steps_per_epoch: int = 80
    learning_rate: float = 1e-3
    neighbor_pool: int = 12
    profile_k: int | None = 4
    profile_clustering: ProfileClustering = "balanced_kmeans"
    min_cluster_fraction: float = 0.08
    balanced_kmeans_max_iter: int = 30
    random_state: int = 42


@dataclass(frozen=True)
class Profile:
    profile_id: str
    x: np.ndarray
    y: np.ndarray
    x_grid: np.ndarray
    y_norm: np.ndarray


@dataclass(frozen=True)
class Candidate:
    candidate_id: int
    profile_index: int
    profile_id: str
    start: int
    length: int
    pattern: np.ndarray
    pattern_fixed: np.ndarray
    mean_slope: float
    slope_var: float
    mean_curvature: float
    curvature_sign_pattern: str
    center_position: float


class ShapeletTransformer(nn.Module):
    """Small Transformer encoder for fixed-length local profile fragments."""

    def __init__(self, config: ContourClusterConfig) -> None:
        super().__init__()
        self.input_proj = nn.Linear(2, config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.d_model * 2,
            batch_first=True,
            dropout=0.05,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, config.num_layers)
        self.out = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.embedding_dim),
        )

    def forward(self, pattern: torch.Tensor) -> torch.Tensor:
        batch, length = pattern.shape
        pos = torch.linspace(0.0, 1.0, length, device=pattern.device)
        pos = pos.expand(batch, length)
        x = torch.stack([pattern, pos], dim=-1)
        x = self.input_proj(x)
        x = self.encoder(x).mean(dim=1)
        return F.normalize(self.out(x), dim=1)


def minmax_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    min_value = np.nanmin(values)
    max_value = np.nanmax(values)
    scale = max_value - min_value
    if not np.isfinite(scale) or scale == 0:
        return np.zeros_like(values, dtype=float)
    return (values - min_value) / scale


def znorm(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    std = np.nanstd(values)
    if not np.isfinite(std) or std < eps:
        return values - np.nanmean(values)
    return (values - np.nanmean(values)) / std


def resample(values: np.ndarray, n_points: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == n_points:
        return values.copy()
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, n_points)
    return np.interp(target, source, values)


def load_profiles(
    data_dir: str | Path,
    config: ContourClusterConfig,
    limit: int | None = None,
) -> tuple[list[Profile], list[tuple[str, int]]]:
    profiles: list[Profile] = []
    skipped: list[tuple[str, int]] = []
    for path in sorted(Path(data_dir).glob("group_*.xlsx")):
        if limit is not None and len(profiles) >= limit:
            break
        frame = pd.read_excel(path)
        if "y" not in frame or ("x_norm" not in frame and "x" not in frame):
            skipped.append((path.stem, len(frame)))
            continue
        x_raw = frame["x_norm"].to_numpy() if "x_norm" in frame else frame["x"].to_numpy()
        y_raw = frame["y"].to_numpy()
        valid = np.isfinite(x_raw) & np.isfinite(y_raw)
        x_raw = np.asarray(x_raw[valid], dtype=float)
        y_raw = np.asarray(y_raw[valid], dtype=float)
        if len(x_raw) < config.min_profile_points:
            skipped.append((path.stem, len(x_raw)))
            continue
        order = np.argsort(x_raw)
        x_raw = x_raw[order]
        y_raw = y_raw[order]
        x_norm = minmax_normalize(x_raw)
        grid = np.linspace(0.0, 1.0, config.profile_length)
        y_interp = np.interp(grid, x_norm, y_raw)
        profiles.append(
            Profile(
                profile_id=path.stem,
                x=x_norm,
                y=y_raw,
                x_grid=grid,
                y_norm=minmax_normalize(y_interp),
            )
        )
    return profiles, skipped


def curvature_sign_pattern(curvature: np.ndarray) -> str:
    q1, q2, q3 = np.quantile(curvature, [0.25, 0.50, 0.75])
    return "".join("+" if q > 0 else "-" if q < 0 else "0" for q in (q1, q2, q3))


def extract_candidates(
    profiles: list[Profile],
    config: ContourClusterConfig,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    cid = 0
    for profile_index, profile in enumerate(profiles):
        slope = np.gradient(profile.y_norm, profile.x_grid)
        curvature = np.gradient(slope, profile.x_grid)
        for length in config.candidate_lengths:
            for start in range(0, config.profile_length - length + 1):
                stop = start + length
                pattern = profile.y_norm[start:stop].copy()
                segment_slope = slope[start:stop]
                segment_curvature = curvature[start:stop]
                candidates.append(
                    Candidate(
                        candidate_id=cid,
                        profile_index=profile_index,
                        profile_id=profile.profile_id,
                        start=start,
                        length=length,
                        pattern=pattern,
                        pattern_fixed=resample(pattern, config.shapelet_points),
                        mean_slope=float(np.mean(segment_slope)),
                        slope_var=float(np.var(segment_slope)),
                        mean_curvature=float(np.mean(segment_curvature)),
                        curvature_sign_pattern=curvature_sign_pattern(segment_curvature),
                        center_position=float(profile.x_grid[start:stop].mean()),
                    )
                )
                cid += 1
    return candidates


def candidate_geometry_matrix(candidates: list[Candidate]) -> np.ndarray:
    return np.array(
        [
            [
                c.mean_slope,
                c.slope_var,
                c.mean_curvature,
                c.center_position,
                c.length,
            ]
            for c in candidates
        ],
        dtype=float,
    )


def candidate_pattern_matrix(candidates: list[Candidate]) -> np.ndarray:
    return np.vstack([c.pattern_fixed for c in candidates]).astype(np.float32)


def augment_patterns(pattern: torch.Tensor) -> torch.Tensor:
    scale = torch.empty((pattern.shape[0], 1), device=pattern.device).uniform_(0.95, 1.05)
    noise = torch.randn_like(pattern) * 0.015
    shift = torch.randint(-2, 3, (pattern.shape[0],), device=pattern.device)
    augmented = pattern * scale + noise
    for row, offset in enumerate(shift.tolist()):
        augmented[row] = torch.roll(augmented[row], shifts=offset, dims=0)
    return augmented.clamp(0.0, 1.0)


def build_enhanced_neighbors(
    candidates: list[Candidate],
    config: ContourClusterConfig,
) -> tuple[np.ndarray, np.ndarray]:
    patterns = candidate_pattern_matrix(candidates)
    geom = StandardScaler().fit_transform(candidate_geometry_matrix(candidates))
    morph_features = np.hstack([StandardScaler().fit_transform(patterns), geom])
    nn_pos = NearestNeighbors(n_neighbors=config.neighbor_pool + 1).fit(morph_features)
    pos_neighbors = nn_pos.kneighbors(return_distance=False)[:, 1:]

    hard_pool_size = min(len(candidates), max(config.neighbor_pool * 6, config.neighbor_pool + 1))
    nn_shape = NearestNeighbors(n_neighbors=hard_pool_size).fit(patterns)
    shape_neighbors = nn_shape.kneighbors(return_distance=False)
    hard_neighbors = np.empty((len(candidates), config.neighbor_pool), dtype=int)
    for idx, pool in enumerate(shape_neighbors):
        pool = pool[pool != idx]
        if len(pool) == 0:
            hard_neighbors[idx] = idx
            continue
        geom_distance = np.linalg.norm(geom[pool] - geom[idx], axis=1)
        order = np.argsort(geom_distance)[::-1]
        selected = pool[order[: config.neighbor_pool]]
        if len(selected) < config.neighbor_pool:
            selected = np.resize(selected, config.neighbor_pool)
        hard_neighbors[idx] = selected
    return pos_neighbors, hard_neighbors


def sample_triplet_indices(
    n_candidates: int,
    batch_size: int,
    rng: np.random.Generator,
    mode: TripletMode,
    pos_neighbors: np.ndarray | None = None,
    hard_negative_neighbors: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    anchors = rng.integers(0, n_candidates, size=batch_size)
    use_augmentation = np.ones(batch_size, dtype=bool)
    positives = anchors.copy()
    if mode == "enhanced" and pos_neighbors is not None:
        use_augmentation = rng.random(batch_size) < 0.50
        neighbor_col = rng.integers(0, pos_neighbors.shape[1], size=batch_size)
        positives = pos_neighbors[anchors, neighbor_col]
        positives[use_augmentation] = anchors[use_augmentation]

    negatives = rng.integers(0, n_candidates, size=batch_size)
    if mode == "enhanced" and hard_negative_neighbors is not None:
        neighbor_col = rng.integers(0, hard_negative_neighbors.shape[1], size=batch_size)
        negatives = hard_negative_neighbors[anchors, neighbor_col]
    same = negatives == anchors
    while np.any(same):
        negatives[same] = rng.integers(0, n_candidates, size=int(np.sum(same)))
        same = negatives == anchors
    return anchors, positives, negatives, use_augmentation


def train_shapelet_encoder(
    candidates: list[Candidate],
    config: ContourClusterConfig,
    mode: TripletMode,
    device: str = "cpu",
) -> tuple[ShapeletTransformer, np.ndarray, dict[str, float]]:
    torch.manual_seed(config.random_state)
    rng = np.random.default_rng(config.random_state)
    model = ShapeletTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    pattern_tensor = torch.tensor(candidate_pattern_matrix(candidates), device=device)

    pos_neighbors = hard_neighbors = None
    if mode == "enhanced":
        pos_neighbors, hard_neighbors = build_enhanced_neighbors(candidates, config)

    losses: list[float] = []
    model.train()
    for _ in range(config.epochs):
        for _step in range(config.steps_per_epoch):
            a_idx, p_idx, n_idx, use_aug = sample_triplet_indices(
                len(candidates),
                config.batch_size,
                rng,
                mode,
                pos_neighbors,
                hard_neighbors,
            )
            anchors = pattern_tensor[a_idx]
            positives = pattern_tensor[p_idx].clone()
            if np.any(use_aug):
                positive_aug = augment_patterns(anchors[torch.tensor(use_aug, device=device)])
                positives[torch.tensor(use_aug, device=device)] = positive_aug
            negatives = pattern_tensor[n_idx]

            v_a = model(anchors)
            v_p = model(positives)
            v_n = model(negatives)
            loss = F.triplet_margin_loss(v_a, v_p, v_n, margin=config.margin, p=2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

    model.eval()
    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(candidates), 2048):
            batch = pattern_tensor[start : start + 2048]
            embeddings.append(model(batch).cpu().numpy())
    metrics = {
        "triplet_loss_final": float(np.mean(losses[-config.steps_per_epoch :])),
        "triplet_loss_mean": float(np.mean(losses)),
    }
    return model, np.vstack(embeddings), metrics


def cluster_shapelets(
    candidates: list[Candidate],
    embeddings: np.ndarray,
    config: ContourClusterConfig,
) -> tuple[np.ndarray, list[int]]:
    labels = KMeans(
        n_clusters=config.n_shapelets,
        n_init=20,
        random_state=config.random_state,
    ).fit_predict(embeddings)
    representative_indices: list[int] = []
    for label in sorted(np.unique(labels)):
        idx = np.flatnonzero(labels == label)
        centroid = embeddings[idx].mean(axis=0)
        nearest = idx[np.argmin(np.linalg.norm(embeddings[idx] - centroid, axis=1))]
        representative_indices.append(int(nearest))
    return labels, representative_indices


def softmin_similarity(distances: np.ndarray, gamma: float) -> tuple[float, int, float]:
    distances = np.asarray(distances, dtype=float)
    weights = np.exp(-gamma * distances)
    weights /= np.sum(weights)
    best_idx = int(np.argmax(weights))
    return float(weights[best_idx]), best_idx, float(distances[best_idx])


def match_shapelet_to_profile(
    shapelet: Candidate,
    profile: Profile,
    config: ContourClusterConfig,
) -> tuple[float, int, float]:
    distances: list[float] = []
    length = shapelet.length
    shapelet_norm = znorm(shapelet.pattern)
    for start in range(0, config.profile_length - length + 1):
        window = profile.y_norm[start : start + length]
        distances.append(float(np.linalg.norm(shapelet_norm - znorm(window))))
    similarity, best_start, best_distance = softmin_similarity(np.array(distances), config.gamma)
    return similarity, best_start, best_distance


def profile_slope_descriptors(profile: Profile) -> np.ndarray:
    slope = np.gradient(profile.y_norm, profile.x_grid)
    curvature = np.gradient(slope, profile.x_grid)
    return np.array(
        [
            np.mean(slope),
            np.std(slope),
            np.quantile(slope, 0.25),
            np.quantile(slope, 0.75),
            np.mean(curvature),
            np.std(curvature),
        ],
        dtype=float,
    )


def build_profile_features(
    profiles: list[Profile],
    candidates: list[Candidate],
    representative_indices: list[int],
    config: ContourClusterConfig,
) -> tuple[np.ndarray, pd.DataFrame]:
    rows: list[dict[str, float | int | str]] = []
    feature_rows: list[np.ndarray] = []
    representatives = [candidates[i] for i in representative_indices]
    for profile in profiles:
        similarities: list[float] = []
        best_positions: list[float] = []
        for shapelet_id, shapelet in enumerate(representatives):
            sim, best_start, best_distance = match_shapelet_to_profile(shapelet, profile, config)
            similarities.append(sim)
            best_positions.append(float(profile.x_grid[best_start : best_start + shapelet.length].mean()))
            rows.append(
                {
                    "profile_id": profile.profile_id,
                    "shapelet_cluster": shapelet_id,
                    "shapelet_candidate_id": shapelet.candidate_id,
                    "best_start": best_start,
                    "best_center_position": best_positions[-1],
                    "similarity": sim,
                    "distance": best_distance,
                }
            )
        feature_rows.append(
            np.concatenate(
                [
                    np.array(similarities, dtype=float),
                    np.array(best_positions, dtype=float),
                    profile_slope_descriptors(profile),
                ]
            )
        )
    return np.vstack(feature_rows), pd.DataFrame(rows)


def cluster_sizes(labels: np.ndarray) -> list[int]:
    return np.bincount(labels.astype(int)).astype(int).tolist()


def cluster_compactness(features: np.ndarray, labels: np.ndarray) -> float:
    distances: list[float] = []
    for label in np.unique(labels):
        cluster_features = features[labels == label]
        centroid = cluster_features.mean(axis=0)
        distances.append(float(np.mean(np.linalg.norm(cluster_features - centroid, axis=1))))
    return float(np.mean(distances))


def cluster_metric_row(features: np.ndarray, labels: np.ndarray, k: int) -> dict[str, float | int | str]:
    sizes = cluster_sizes(labels)
    min_size = int(min(sizes))
    max_size = int(max(sizes))
    imbalance_ratio = float(max_size / min_size) if min_size > 0 else float("inf")
    score = silhouette_score(features, labels) if len(np.unique(labels)) > 1 else np.nan
    return {
        "K": int(k),
        "silhouette": float(score),
        "compactness": cluster_compactness(features, labels),
        "cluster_sizes": ";".join(str(size) for size in sizes),
        "min_cluster_size": min_size,
        "max_cluster_size": max_size,
        "imbalance_ratio": imbalance_ratio,
    }


def balanced_capacities(n_samples: int, n_clusters: int) -> np.ndarray:
    base = n_samples // n_clusters
    remainder = n_samples % n_clusters
    return np.array(
        [base + (1 if cluster < remainder else 0) for cluster in range(n_clusters)],
        dtype=int,
    )


def assign_balanced_labels(features: np.ndarray, centers: np.ndarray) -> np.ndarray:
    n_samples, n_clusters = features.shape[0], centers.shape[0]
    capacities = balanced_capacities(n_samples, n_clusters)
    slots = np.repeat(np.arange(n_clusters), capacities)
    cost = np.linalg.norm(features[:, None, :] - centers[slots][None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)
    labels = np.empty(n_samples, dtype=int)
    labels[row_ind] = slots[col_ind]
    return labels


def balanced_kmeans(
    features: np.ndarray,
    n_clusters: int,
    random_state: int,
    max_iter: int,
) -> np.ndarray:
    if n_clusters >= len(features):
        raise ValueError("n_clusters must be smaller than the number of samples")
    initializer = KMeans(
        n_clusters=n_clusters,
        n_init=30,
        random_state=random_state,
    ).fit(features)
    centers = initializer.cluster_centers_
    labels: np.ndarray | None = None
    for _ in range(max_iter):
        new_labels = assign_balanced_labels(features, centers)
        new_centers = centers.copy()
        for cluster in range(n_clusters):
            new_centers[cluster] = features[new_labels == cluster].mean(axis=0)
        if labels is not None and np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        centers = new_centers
    if labels is None:
        raise RuntimeError("balanced_kmeans failed to assign labels")
    return labels


def cluster_profiles(
    scaled_features: np.ndarray,
    k: int,
    config: ContourClusterConfig,
) -> np.ndarray:
    if config.profile_clustering == "balanced_kmeans":
        return balanced_kmeans(
            scaled_features,
            n_clusters=k,
            random_state=config.random_state,
            max_iter=config.balanced_kmeans_max_iter,
        )
    return KMeans(
        n_clusters=k,
        n_init=30,
        random_state=config.random_state,
    ).fit_predict(scaled_features)


def choose_profile_k(
    features: np.ndarray,
    config: ContourClusterConfig,
) -> tuple[np.ndarray, int, pd.DataFrame]:
    scaled = StandardScaler().fit_transform(features)
    metric_rows: list[dict[str, float | int | str]] = []
    best_score = -np.inf
    best_labels: np.ndarray | None = None
    best_k = config.profile_k or config.k_range[0]
    min_allowed = max(1, int(np.ceil(config.min_cluster_fraction * len(features))))
    k_values = sorted(set(config.k_range) | ({config.profile_k} if config.profile_k is not None else set()))
    for k in k_values:
        if k >= len(features):
            continue
        labels = cluster_profiles(scaled, k, config)
        row = cluster_metric_row(scaled, labels, k)
        row["profile_clustering"] = config.profile_clustering
        row["passes_min_cluster_size"] = int(row["min_cluster_size"] >= min_allowed)
        metric_rows.append(row)
        if config.profile_k is not None and k != config.profile_k:
            continue
        score = float(row["silhouette"])
        if config.profile_k is None and row["min_cluster_size"] < min_allowed:
            continue
        if config.profile_k is not None or score > best_score:
            best_score = score
            best_labels = labels
            best_k = k
    if best_labels is None:
        valid_rows = [row for row in metric_rows if row["min_cluster_size"] > 0]
        if not valid_rows:
            raise ValueError("No valid K was available for profile clustering")
        fallback = max(valid_rows, key=lambda row: float(row["silhouette"]))
        best_k = int(fallback["K"])
        best_labels = cluster_profiles(scaled, best_k, config)
    return best_labels, best_k, pd.DataFrame(metric_rows)


def representative_shapelet_table(
    candidates: list[Candidate],
    representative_indices: list[int],
    matches: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for cluster_id, candidate_index in enumerate(representative_indices):
        c = candidates[candidate_index]
        cluster_matches = matches[matches["shapelet_cluster"] == cluster_id]
        best = cluster_matches.sort_values("similarity", ascending=False).head(3)
        rows.append(
            {
                "shapelet_cluster": cluster_id,
                "candidate_id": c.candidate_id,
                "source_profile_id": c.profile_id,
                "start": c.start,
                "length": c.length,
                "source_center_position": c.center_position,
                "mean_slope": c.mean_slope,
                "slope_variance": c.slope_var,
                "mean_curvature": c.mean_curvature,
                "curvature_sign_pattern": c.curvature_sign_pattern,
                "matched_profile_examples": ";".join(best["profile_id"].astype(str).tolist()),
                "mean_best_match_position": float(cluster_matches["best_center_position"].mean()),
            }
        )
    return pd.DataFrame(rows)


def profile_cluster_summary_table(
    profiles: list[Profile],
    labels: np.ndarray,
    profile_features: np.ndarray,
    n_shapelets: int,
) -> pd.DataFrame:
    scaled = StandardScaler().fit_transform(profile_features)
    rows: list[dict[str, float | int | str]] = []
    slope_names = [
        "profile_slope_mean",
        "profile_slope_std",
        "profile_slope_q25",
        "profile_slope_q75",
        "profile_curvature_mean",
        "profile_curvature_std",
    ]
    for label in sorted(np.unique(labels)):
        idx = np.flatnonzero(labels == label)
        centroid = scaled[idx].mean(axis=0)
        representative_idx = idx[np.argmin(np.linalg.norm(scaled[idx] - centroid, axis=1))]
        row: dict[str, float | int | str] = {
            "kappa": int(label),
            "n_profiles": int(len(idx)),
            "representative_profile_id": profiles[int(representative_idx)].profile_id,
        }
        for shapelet_id in range(n_shapelets):
            row[f"shapelet_{shapelet_id}_similarity_mean"] = float(
                profile_features[idx, shapelet_id].mean()
            )
            row[f"shapelet_{shapelet_id}_position_mean"] = float(
                profile_features[idx, n_shapelets + shapelet_id].mean()
            )
        slope_start = 2 * n_shapelets
        for offset, name in enumerate(slope_names):
            row[name] = float(profile_features[idx, slope_start + offset].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def embedding_pca(embeddings: np.ndarray) -> np.ndarray:
    return PCA(n_components=2, random_state=42).fit_transform(embeddings)


def run_contourcluster(
    data_dir: str | Path,
    output_dir: str | Path,
    config: ContourClusterConfig = ContourClusterConfig(),
    mode: TripletMode = "baseline",
    limit: int | None = None,
    device: str = "cpu",
) -> dict[str, object]:
    profiles, skipped = load_profiles(data_dir, config, limit=limit)
    if not profiles:
        raise ValueError("No valid profiles were loaded")
    candidates = extract_candidates(profiles, config)
    model, embeddings, train_metrics = train_shapelet_encoder(candidates, config, mode, device)
    shapelet_labels, representative_indices = cluster_shapelets(candidates, embeddings, config)
    profile_features, matches = build_profile_features(
        profiles, candidates, representative_indices, config
    )
    profile_labels, selected_k, profile_metrics = choose_profile_k(profile_features, config)
    reps = representative_shapelet_table(candidates, representative_indices, matches)
    profile_summary = profile_cluster_summary_table(
        profiles,
        profile_labels,
        profile_features,
        len(representative_indices),
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    profile_table = pd.DataFrame(
        {
            "profile_id": [p.profile_id for p in profiles],
            "kappa": profile_labels.astype(int),
        }
    )
    profile_table.to_csv(out / "profile_clusters.csv", index=False)
    reps.to_csv(out / "representative_shapelets.csv", index=False)
    profile_summary.to_csv(out / "profile_cluster_summary.csv", index=False)
    matches.to_csv(out / "shapelet_matches.csv", index=False)
    profile_metrics.to_csv(out / "profile_cluster_metrics.csv", index=False)
    pd.DataFrame(
        {
            "candidate_id": [c.candidate_id for c in candidates],
            "profile_id": [c.profile_id for c in candidates],
            "shapelet_cluster": shapelet_labels.astype(int),
        }
    ).to_csv(out / "candidate_shapelet_clusters.csv", index=False)
    pd.DataFrame(skipped, columns=["profile_id", "n_points"]).to_csv(
        out / "skipped_profiles.csv", index=False
    )
    pca = embedding_pca(embeddings)
    pd.DataFrame(
        {
            "candidate_id": [c.candidate_id for c in candidates],
            "pc1": pca[:, 0],
            "pc2": pca[:, 1],
            "shapelet_cluster": shapelet_labels.astype(int),
        }
    ).to_csv(out / "candidate_embedding_pca.csv", index=False)
    run_metrics = train_metrics | {
        "mode": mode,
        "selected_profile_K": selected_k,
        "profile_clustering": config.profile_clustering,
        "profile_k_requested": config.profile_k,
    }
    pd.DataFrame([run_metrics]).to_csv(out / "training_summary.csv", index=False)

    torch.save(model.state_dict(), out / "shapelet_encoder.pt")
    return {
        "profiles": profiles,
        "skipped": skipped,
        "candidates": candidates,
        "embeddings": embeddings,
        "shapelet_labels": shapelet_labels,
        "representative_indices": representative_indices,
        "profile_features": profile_features,
        "profile_labels": profile_labels,
        "selected_k": selected_k,
        "matches": matches,
        "representatives": reps,
        "profile_summary": profile_summary,
        "profile_metrics": profile_metrics,
        "train_metrics": run_metrics,
    }
