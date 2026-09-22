#!/usr/bin/env python3
"""Frozen seed-8888 common evaluator for the Genome Biology Table 1 audit.

This script is deliberately representation-only.  It never trains a model,
selects a checkpoint, or feeds target labels back into inference.  For Track 1
it applies exactly one shared path to both methods:

    GT-novel subset -> row L2 -> true K (benchmark only) ->
    sklearn KMeans(random_state=0, n_init=10) -> post-hoc scoring.

KMeans' ten restarts are selected internally by sklearn inertia.  Target labels
are not inspected until after ``fit_predict`` returns.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    roc_auc_score,
)
from threadpoolctl import threadpool_limits


DATASETS = {
    "Cao": {"shared": 6, "source_private": 4, "novel": 6},
    "Quake_10x": {"shared": 12, "source_private": 12, "novel": 12},
    "Quake_Smart-seq2": {"shared": 15, "source_private": 15, "novel": 15},
    "Wagner": {"shared": 5, "source_private": 4, "novel": 5},
    "Zeisel_2018": {"shared": 6, "source_private": 5, "novel": 6},
}

KMEANS_RANDOM_STATE = 0
KMEANS_N_INIT = 10
NUMERIC_THREADS = 1
LEIDEN_RESOLUTION = 1.0
LEIDEN_N_NEIGHBORS = 15
LEIDEN_RANDOM_STATE = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scolar-arrays-root", type=Path, required=True)
    p.add_argument("--strict-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def row_l2(z: np.ndarray) -> np.ndarray:
    z_t = torch.as_tensor(np.asarray(z), dtype=torch.float32)
    if z_t.ndim != 2:
        raise ValueError(f"Expected 2-D embeddings; got {tuple(z_t.shape)}")
    return torch.nn.functional.normalize(z_t, p=2, dim=1).numpy()


def dense_labels(y: np.ndarray) -> np.ndarray:
    _, inv = np.unique(np.asarray(y), return_inverse=True)
    return inv.astype(np.int64, copy=False)


def cluster_acc(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Implement the benchmark's global Hungarian accuracy semantics."""
    y_pred = np.asarray(y_pred, dtype=np.int64)
    y_true = np.asarray(y_true, dtype=np.int64)
    if y_pred.size != y_true.size or y_pred.size == 0:
        raise ValueError("Invalid global Hungarian inputs")
    d = int(max(y_pred.max(), y_true.max()) + 1)
    w = np.zeros((d, d), dtype=np.int64)
    np.add.at(w, (y_pred, y_true), 1)
    row, col = linear_sum_assignment(w.max() - w)
    return float(w[row, col].sum() / y_pred.size)


def aligned_scores(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> dict:
    """Exact production weighted/macro Hungarian implementation."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    d = int(max(y_pred.max(), y_true.max()) + 1)
    w = np.zeros((d, d), dtype=np.int64)
    np.add.at(w, (y_pred, y_true), 1)
    row, col = linear_sum_assignment(w.max() - w)
    mapping = dict(zip(row.tolist(), col.tolist()))
    mapped = np.asarray([mapping.get(int(v), int(v)) for v in y_pred], dtype=np.int64)
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (y_true, mapped), 1)
    per_class = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    return {
        "weighted": float(w[row, col].sum() / y_pred.size),
        "macro": float(per_class.mean()),
        "confusion": cm,
    }


def safe_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size < 2 or np.unique(y_true).size < 2 or np.unique(y_pred).size < 2:
        return {"ari": None, "ami": None, "nmi": None}
    return {
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "ami": float(adjusted_mutual_info_score(y_true, y_pred)),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
    }


def frozen_kmeans(z: np.ndarray, y_true_dense: np.ndarray, k: int) -> dict:
    # IMPORTANT: y_true_dense is not used anywhere before fit_predict returns.
    z_norm = row_l2(z)
    km = KMeans(
        n_clusters=int(k),
        init="k-means++",
        random_state=KMEANS_RANDOM_STATE,
        n_init=KMEANS_N_INIT,
    )
    # A fixed random_state is not enough for bitwise reproducibility when
    # OpenMP/BLAS reduction order varies.  Freeze the numeric thread count too.
    with threadpool_limits(limits=NUMERIC_THREADS):
        pred = km.fit_predict(z_norm)
    aligned = aligned_scores(y_true_dense, pred, k)
    return {
        "pred": pred,
        "inertia": float(km.inertia_),
        "weighted": aligned["weighted"],
        "macro": aligned["macro"],
        "ari": float(adjusted_rand_score(y_true_dense, pred)),
        "ami": float(adjusted_mutual_info_score(y_true_dense, pred)),
        "nmi": float(normalized_mutual_info_score(y_true_dense, pred)),
        "confusion": aligned["confusion"],
    }


def frozen_leiden(z: np.ndarray) -> np.ndarray:
    z_norm = row_l2(z)
    n = z_norm.shape[0]
    if n < 2:
        return np.zeros(n, dtype=np.int64)
    adata = sc.AnnData(z_norm)
    sc.pp.neighbors(
        adata,
        n_neighbors=max(2, min(LEIDEN_N_NEIGHBORS, n - 1)),
        use_rep="X",
        metric="euclidean",
        random_state=LEIDEN_RANDOM_STATE,
    )
    sc.tl.leiden(
        adata,
        resolution=LEIDEN_RESOLUTION,
        random_state=LEIDEN_RANDOM_STATE,
        flavor="igraph",
        n_iterations=2,
        directed=False,
    )
    return adata.obs["leiden"].astype(int).values


def load_scolar(path: Path, spec: dict) -> dict:
    a = np.load(path, allow_pickle=False)
    required = {
        "z", "y_raw", "gt_novel", "known_pred", "known_true",
        "novelty_score", "method_pred_novel", "n_known_classes",
    }
    missing = sorted(required - set(a.files))
    if missing:
        raise KeyError(f"{path}: missing {missing}")
    gt_novel = a["gt_novel"].astype(bool)
    if int(gt_novel.sum()) == 0 or int((~gt_novel).sum()) == 0:
        raise ValueError(f"{path}: empty known/novel benchmark subset")
    if np.unique(a["y_raw"][gt_novel]).size != spec["novel"]:
        raise ValueError(f"{path}: novel class cardinality mismatch")
    if int(a["n_known_classes"]) != spec["shared"] + spec["source_private"]:
        raise ValueError(f"{path}: source class cardinality mismatch")
    return {
        "z": a["z"],
        "y_all": a["y_raw"],
        "gt_novel": gt_novel,
        "known_pred": a["known_pred"].astype(np.int64),
        "known_true": a["known_true"].astype(np.int64),
        "novelty_score": a["novelty_score"],
        "pred_novel": a["method_pred_novel"].astype(bool),
        "n_known_classes": int(a["n_known_classes"]),
    }


def load_scbol(path: Path, spec: dict) -> dict:
    a = np.load(path, allow_pickle=False)
    required = {"z", "y_true", "source_pred", "source_max_logit", "method_pred"}
    missing = sorted(required - set(a.files))
    if missing:
        raise KeyError(f"{path}: missing {missing}")
    y = a["y_true"].astype(np.int64)
    source_classes = spec["shared"] + spec["source_private"]
    known = y < spec["shared"]
    gt_novel = y >= source_classes
    if np.any(~(known | gt_novel)):
        raise ValueError(f"{path}: target contains source-private labels")
    if np.unique(y[gt_novel]).size != spec["novel"]:
        raise ValueError(f"{path}: novel class cardinality mismatch")
    return {
        "z": a["z"],
        "y_all": y,
        "gt_novel": gt_novel,
        "known_pred": a["source_pred"][known].astype(np.int64),
        "known_true": y[known],
        # Larger values always mean "more novel" in this shared interface.
        "novelty_score": -a["source_max_logit"],
        "pred_novel": a["method_pred"].astype(np.int64) >= source_classes,
        "n_known_classes": source_classes,
    }


def evaluate(dataset: str, method: str, arrays: dict) -> tuple[dict, dict]:
    z = np.asarray(arrays["z"])
    y_all = np.asarray(arrays["y_all"])
    gt_novel = np.asarray(arrays["gt_novel"], dtype=bool)
    pred_novel = np.asarray(arrays["pred_novel"], dtype=bool)
    if not (len(z) == len(y_all) == len(gt_novel) == len(pred_novel)):
        raise ValueError(f"{dataset}/{method}: target array length mismatch")

    y_novel = dense_labels(y_all[gt_novel])
    k = int(np.unique(y_novel).size)
    track1 = frozen_kmeans(z[gt_novel], y_novel, k)
    known_pred = arrays["known_pred"]
    known_true = arrays["known_true"]
    known_acc = float(np.mean(known_pred == known_true))
    overall_j = cluster_acc(
        np.concatenate([known_pred, track1["pred"] + arrays["n_known_classes"]]),
        np.concatenate([known_true, y_novel + arrays["n_known_classes"]]),
    )
    auroc = float(roc_auc_score(gt_novel.astype(np.int8), arrays["novelty_score"]))

    tp = int(np.sum(gt_novel & pred_novel))
    fp = int(np.sum(~gt_novel & pred_novel))
    fn = int(np.sum(gt_novel & ~pred_novel))
    tn = int(np.sum(~gt_novel & ~pred_novel))
    # Match production's zero-safe detection semantics: declaring no novel
    # cells yields precision=recall=F1=0 when GT novel cells exist.
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    deployment = {
        "predicted_novel": int(pred_novel.sum()),
        "gt_novel": int(gt_novel.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "leiden_clusters": None,
        "all": {"ari": None, "ami": None, "nmi": None},
        "tp_only": {"ari": None, "ami": None, "nmi": None},
    }
    if int(pred_novel.sum()) >= 2:
        leiden = frozen_leiden(z[pred_novel])
        y_pred_subset = y_all[pred_novel]
        gt_pred_subset = gt_novel[pred_novel]
        deployment["leiden_clusters"] = int(np.unique(leiden).size)
        deployment["all"] = safe_scores(y_pred_subset, leiden)
        deployment["tp_only"] = safe_scores(
            y_pred_subset[gt_pred_subset], leiden[gt_pred_subset]
        )

    result = {
        "dataset": dataset,
        "method": method,
        "seed": 8888,
        "known": known_acc,
        "novel_weighted": track1["weighted"],
        "novel_macro": track1["macro"],
        "ari": track1["ari"],
        "ami": track1["ami"],
        "nmi": track1["nmi"],
        "overall_j": overall_j,
        "auroc": auroc,
        "k_novel_benchmark_only": k,
        "kmeans_inertia": track1["inertia"],
        "detection_f1": f1,
        "pred_novel": int(pred_novel.sum()),
        "gt_novel": int(gt_novel.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "leiden_clusters": deployment["leiden_clusters"],
        "leiden_ari_all": deployment["all"]["ari"],
        "leiden_ami_all": deployment["all"]["ami"],
        "leiden_nmi_all": deployment["all"]["nmi"],
        "leiden_ari_tp": deployment["tp_only"]["ari"],
        "leiden_ami_tp": deployment["tp_only"]["ami"],
        "leiden_nmi_tp": deployment["tp_only"]["nmi"],
    }
    detail = {
        "result": result,
        "track1_confusion_true_by_aligned_cluster": track1["confusion"].tolist(),
        "deployment": deployment,
    }
    return result, detail


def mean_rows(rows: list[dict]) -> list[dict]:
    fields = [
        "known", "novel_weighted", "novel_macro", "ari", "ami", "nmi",
        "overall_j", "auroc", "detection_f1", "pred_novel", "gt_novel",
        "tp", "fp", "fn", "leiden_clusters", "leiden_ari_all",
        "leiden_ami_all", "leiden_nmi_all", "leiden_ari_tp",
        "leiden_ami_tp", "leiden_nmi_tp",
    ]
    out = []
    for method in ("scOLAR", "scBOL"):
        subset = [r for r in rows if r["method"] == method]
        row = {"dataset": "Mean", "method": method, "seed": 8888}
        for field in fields:
            values = [r[field] for r in subset if r.get(field) is not None]
            # A cross-dataset Leiden mean is undefined when a method declared
            # no PredNovel cells on any constituent dataset.  Do not silently
            # average only the available subset and label it as a five-dataset mean.
            if field.startswith("leiden_") and len(values) != len(subset):
                row[field] = None
            else:
                row[field] = float(np.mean(values)) if values else None
        row["k_novel_benchmark_only"] = None
        row["kmeans_inertia"] = None
        out.append(row)
    return out


def format_pct(value) -> str:
    return "NA" if value is None else f"{100 * value:.2f}"


def format_metric(value) -> str:
    return "NA" if value is None else f"{value:.4f}"


def write_markdown(path: Path, rows: list[dict], means: list[dict]) -> None:
    display = rows + means
    lines = [
        "# Frozen common-evaluator comparison — seed 8888",
        "",
        "Track 1: GT-novel only → row L2 → true K (benchmark only) → one sklearn "
        "KMeans fit (`random_state=0`, `n_init=10`) → post-hoc Hungarian/ARI/AMI/NMI. "
        "The internal restarts are selected by sklearn inertia only.",
        "",
        "| Dataset | Method | Known % | Novel w. % | Novel macro % | ARI | AMI | NMI | OverallJ % | AUROC | Det. F1 | PredNovel | Leiden K | Leiden ARI all | Leiden ARI TP |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in display:
        lines.append(
            "| {dataset} | {method} | {known} | {nw} | {nm} | {ari} | {ami} | {nmi} | "
            "{oj} | {auroc} | {f1} | {pn} | {lk} | {la} | {lt} |".format(
                dataset=r["dataset"], method=r["method"],
                known=format_pct(r.get("known")), nw=format_pct(r.get("novel_weighted")),
                nm=format_pct(r.get("novel_macro")), ari=format_metric(r.get("ari")),
                ami=format_metric(r.get("ami")), nmi=format_metric(r.get("nmi")),
                oj=format_pct(r.get("overall_j")), auroc=format_metric(r.get("auroc")),
                f1=format_metric(r.get("detection_f1")), pn=("NA" if r.get("pred_novel") is None else f"{r['pred_novel']:.1f}" if r["dataset"] == "Mean" else str(r["pred_novel"])),
                lk=("NA" if r.get("leiden_clusters") is None else f"{r['leiden_clusters']:.1f}" if r["dataset"] == "Mean" else str(r["leiden_clusters"])),
                la=format_metric(r.get("leiden_ari_all")), lt=format_metric(r.get("leiden_ari_tp")),
            )
        )
    lines.extend([
        "",
        "Detection and Leiden are Track 2 (method-declared PredNovel), separate from Track 1. "
        "scBOL retains its released training-time oracle target-class cardinality; scOLAR's "
        "deployment path does not use K.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.set_num_threads(NUMERIC_THREADS)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve()
    rows = []
    details = {}
    inputs = []

    for dataset, spec in DATASETS.items():
        paths = {
            "scOLAR": args.scolar_arrays_root / dataset / "final_model.common_eval_arrays.npz",
            "scBOL": args.strict_root / "scBOL" / f"{dataset}_s8888" / "final_model.external_eval_arrays.npz",
        }
        for method, path in paths.items():
            path = path.resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            arrays = load_scolar(path, spec) if method == "scOLAR" else load_scbol(path, spec)
            result, detail = evaluate(dataset, method, arrays)
            rows.append(result)
            details[f"{dataset}/{method}"] = detail
            inputs.append({"dataset": dataset, "method": method, "path": str(path), "sha256": sha256(path)})
            print(
                f"{dataset:20s} {method:6s} Known={result['known']:.6f} "
                f"Novel={result['novel_weighted']:.6f} ARI={result['ari']:.6f} "
                f"OverallJ={result['overall_j']:.6f} AUROC={result['auroc']:.6f}"
            )

    means = mean_rows(rows)
    fieldnames = list(rows[0].keys())
    with (out / "common_evaluator_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
        w.writerows(means)

    comparisons = []
    metrics = ["known", "novel_weighted", "novel_macro", "ari", "ami", "nmi", "overall_j", "auroc", "detection_f1"]
    for dataset in [*DATASETS, "Mean"]:
        pool = rows if dataset != "Mean" else means
        a = next(r for r in pool if r["dataset"] == dataset and r["method"] == "scOLAR")
        b = next(r for r in pool if r["dataset"] == dataset and r["method"] == "scBOL")
        row = {"dataset": dataset, "delta_definition": "scOLAR_minus_scBOL"}
        for metric in metrics:
            row[metric] = None if a.get(metric) is None or b.get(metric) is None else a[metric] - b[metric]
        comparisons.append(row)
    with (out / "common_evaluator_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(comparisons[0].keys()))
        w.writeheader()
        w.writerows(comparisons)

    protocol = {
        "status": "FROZEN",
        "purpose": "Genome Biology Methodology P0 Final Table 1 audit; seed8888 fixed-final re-evaluation",
        "track1": {
            "input": "all and only GT-novel cells",
            "preprocessing": "row-wise L2 normalization",
            "k": "true novel-class count; benchmark-only",
            "estimator": "sklearn.cluster.KMeans",
            "init": "k-means++",
            "random_state": KMEANS_RANDOM_STATE,
            "n_init": KMEANS_N_INIT,
            "restart_selection": "sklearn internal minimum inertia only",
            "numeric_threads": NUMERIC_THREADS,
            "ground_truth_use": "K before fit; labels only for post-hoc scoring after fit_predict",
        },
        "track2": {
            "input": "method-declared PredNovel",
            "preprocessing": "row-wise L2 normalization",
            "clustering": "fixed Leiden; no true K",
            "resolution": LEIDEN_RESOLUTION,
            "n_neighbors": LEIDEN_N_NEIGHBORS,
            "metric": "euclidean",
            "random_state": LEIDEN_RANDOM_STATE,
            "flavor": "igraph",
            "directed": False,
            "n_iterations": 2,
        },
        "script": {"path": str(script_path), "sha256": sha256(script_path)},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "scanpy": sc.__version__,
            "torch": torch.__version__,
        },
        "inputs": inputs,
        "canonical_scbol_policy": "outer strict_seed8888 five-dataset matrix; never target-metric-selected",
        "prohibited": ["GT-selected restart", "GT-selected checkpoint", "legacy/oracle evaluator", "representation retraining"],
    }
    (out / "common_evaluator_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (out / "common_evaluator_details.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    write_markdown(out / "common_evaluator_table.md", rows, means)


if __name__ == "__main__":
    main()
