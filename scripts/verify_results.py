"""Recompute and validate every derived submitted CSV from run-level results."""

import argparse
import csv
import itertools
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("Cao", "Quake_10x", "Quake_Smart-seq2", "Wagner", "Zeisel_2018")
SEEDS = (101, 202, 303, 404)
ARMS = ("REAL_CL", "DEPTH_SHUFFLED_CL", "GENERIC_STAR")
KEY = ("joint_correct_known", "novelty_ap", "full_prednovel_ami")
ENDPOINTS = KEY + (
    "known_retention", "retained_known_accuracy", "novelty_auroc",
    "threshold_precision", "threshold_recall", "threshold_f1",
    "full_prednovel_ari", "full_prednovel_nmi", "lcc_yield",
    "assigned_cell_coverage", "lineage_U",
)
CONTRASTS = (
    ("REAL_CL", "DEPTH_SHUFFLED_CL"),
    ("REAL_CL", "GENERIC_STAR"),
    ("DEPTH_SHUFFLED_CL", "GENERIC_STAR"),
)


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value):
    if value is None or str(value).strip() == "":
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite numeric value: {value!r}")
    return result


def sample_std(values):
    return float(np.std(np.asarray(values, dtype=float), ddof=1)) if len(values) > 1 else None


def uncertainty(groups):
    arrays = [np.asarray(groups[d], dtype=float) for d in DATASETS if d in groups and groups[d]]
    n = len(arrays)
    if not n:
        return dict(effect=None, ci_low=None, ci_high=None, p_exact=None, n_datasets=0,
                    n_positive_datasets=None, n_negative_datasets=None, minimum_two_sided_p=None)
    means = np.asarray([values.mean() for values in arrays])
    effect = float(means.mean())
    rng = np.random.default_rng(9012026)
    choice = rng.integers(0, n, size=(10000, n))
    boot = np.zeros_like(choice, dtype=float)
    for index, values in enumerate(arrays):
        mask = choice == index
        count = int(mask.sum())
        boot[mask] = values[rng.integers(0, len(values), size=(count, len(values)))].mean(1)
    ci_low, ci_high = np.quantile(boot.mean(1), [.025, .975])
    flips = np.asarray(list(itertools.product([-1.0, 1.0], repeat=n)))
    statistics = np.abs((flips * means).mean(1))
    p_exact = float(np.mean(statistics >= abs(effect) - 1e-14))
    return dict(
        effect=effect, ci_low=float(ci_low), ci_high=float(ci_high), p_exact=p_exact,
        n_datasets=n, n_positive_datasets=int((means > 0).sum()),
        n_negative_datasets=int((means < 0).sum()), minimum_two_sided_p=float(2 / 2**n),
    )


def holm(rows, indices):
    order = sorted(indices, key=lambda index: rows[index]["p_exact"] if rows[index]["p_exact"] is not None else 1.0)
    running = 0.0
    for rank, index in enumerate(order):
        p_value = rows[index]["p_exact"]
        running = max(running, min(1.0, (len(order) - rank) * (p_value if p_value is not None else 1.0)))
        rows[index]["p_holm"] = running if p_value is not None else None


def recompute(run_rows):
    lookup = {(row["dataset"], int(row["seed"]), row["arm"]): row for row in run_rows}
    aggregate = []
    for dataset in DATASETS:
        for arm in ARMS:
            group = [lookup[(dataset, seed, arm)] for seed in SEEDS]
            item = {"dataset": dataset, "arm": arm}
            for endpoint in ENDPOINTS:
                values = [number(row.get(endpoint)) for row in group]
                values = [value for value in values if value is not None]
                item[f"{endpoint}_mean"] = float(np.mean(values)) if values else None
                item[f"{endpoint}_std"] = sample_std(values)
                item[f"{endpoint}_count"] = len(values)
            aggregate.append(item)

    dataset_effects = []
    summaries = []
    for first, second in CONTRASTS:
        contrast = f"{first}-{second}"
        for endpoint in ENDPOINTS:
            groups = {}
            controls = []
            for dataset in DATASETS:
                paired = []
                control_values = []
                for seed in SEEDS:
                    left = lookup[(dataset, seed, first)]
                    right = lookup[(dataset, seed, second)]
                    if left["split_sha256"] != right["split_sha256"]:
                        raise ValueError(f"split hash mismatch for {dataset}, seed {seed}, {contrast}")
                    left_value, right_value = number(left.get(endpoint)), number(right.get(endpoint))
                    if left_value is None or right_value is None:
                        continue
                    paired.append(left_value - right_value)
                    control_values.append(right_value)
                if paired:
                    groups[dataset] = paired
                    controls.append(float(np.mean(control_values)))
                dataset_effects.append(dict(
                    dataset=dataset, contrast=contrast, endpoint=endpoint,
                    n_paired_seeds=len(paired), effect=float(np.mean(paired)) if paired else None,
                    seed_sd=sample_std(paired),
                    status="FOUR_SEEDS_COMPLETE" if len(paired) == 4 else "PARTIAL_OR_NOT_EVALUABLE",
                ))
            item = dict(contrast=contrast, endpoint=endpoint, planned=(first == "REAL_CL"), **uncertainty(groups),
                        n_pairs=sum(map(len, groups.values())),
                        complete_5x4=(len(groups) == 5 and all(len(values) == 4 for values in groups.values())),
                        p_holm=None)
            control_mean = float(np.mean(controls)) if controls else None
            item["relative_effect"] = item["effect"] / control_mean if control_mean is not None and control_mean > 0 and item["effect"] is not None else None
            summaries.append(item)
    holm(summaries, [i for i, row in enumerate(summaries) if row["planned"] and row["endpoint"] in KEY])
    holm(summaries, [i for i, row in enumerate(summaries) if row["planned"] and row["endpoint"] == "lineage_U"])
    return aggregate, dataset_effects, summaries


def bool_value(value):
    if isinstance(value, bool):
        return value
    if str(value).strip().lower() in {"true", "1", "yes"}:
        return True
    if str(value).strip().lower() in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid Boolean: {value!r}")


def compare_table(name, expected, actual, keys):
    expected_by_key = {tuple(row[key] for key in keys): row for row in expected}
    actual_by_key = {tuple(row[key] for key in keys): row for row in actual}
    if expected_by_key.keys() != actual_by_key.keys():
        raise ValueError(f"{name} key mismatch")
    for key, expected_row in expected_by_key.items():
        actual_row = actual_by_key[key]
        if set(expected_row) != set(actual_row):
            raise ValueError(f"{name} columns differ at {key}")
        for column, expected_value in expected_row.items():
            actual_value = actual_row[column]
            if isinstance(expected_value, bool):
                if bool_value(actual_value) != expected_value:
                    raise ValueError(f"{name} {key} {column}: {actual_value!r} != {expected_value!r}")
            elif isinstance(expected_value, (int, float)) or expected_value is None:
                observed = number(actual_value)
                if expected_value is None:
                    if observed is not None:
                        raise ValueError(f"{name} {key} {column}: expected empty, got {actual_value!r}")
                elif isinstance(expected_value, int):
                    if observed is None or int(observed) != expected_value or observed != int(observed):
                        raise ValueError(f"{name} {key} {column}: {actual_value!r} != {expected_value}")
                elif observed is None or not math.isclose(observed, expected_value, rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError(f"{name} {key} {column}: {actual_value!r} != {expected_value!r}")
            elif actual_value != expected_value:
                raise ValueError(f"{name} {key} {column}: {actual_value!r} != {expected_value!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "supplementary_data")
    args = parser.parse_args()
    filenames = ("results_run_level.csv", "aggregate_by_dataset.csv", "paired_effects_by_dataset.csv", "paired_effects_and_CI.csv")
    tables = {name: read_rows(args.data_dir / name) for name in filenames}
    run_rows = tables["results_run_level.csv"]
    observed = {(row["dataset"], int(row["seed"]), row["arm"]) for row in run_rows}
    expected_matrix = {(dataset, seed, arm) for dataset in DATASETS for seed in SEEDS for arm in ARMS}
    if len(run_rows) != 60 or observed != expected_matrix:
        raise ValueError(f"run matrix mismatch: missing={sorted(expected_matrix-observed)}, extra={sorted(observed-expected_matrix)}")
    if any(row.get("status") != "COMPLETE" for row in run_rows):
        raise ValueError("not every run is marked COMPLETE")
    aggregate, paired, summaries = recompute(run_rows)
    compare_table("aggregate_by_dataset.csv", aggregate, tables["aggregate_by_dataset.csv"], ("dataset", "arm"))
    compare_table("paired_effects_by_dataset.csv", paired, tables["paired_effects_by_dataset.csv"], ("dataset", "contrast", "endpoint"))
    compare_table("paired_effects_and_CI.csv", summaries, tables["paired_effects_and_CI.csv"], ("contrast", "endpoint"))
    print("PASS: recomputed and matched all aggregate means/SDs, paired effects, hierarchical-bootstrap CIs, exact sign-flip tests, and Holm corrections")


if __name__ == "__main__":
    main()
