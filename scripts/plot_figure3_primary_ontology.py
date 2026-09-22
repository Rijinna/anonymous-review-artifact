"""Generate the primary ontology-intervention forest plot used as Figure 3.

Only two authority files are read:

* supplementary_data/paired_effects_by_dataset.csv
* supplementary_data/paired_effects_and_CI.csv

Dataset rows use the supplied matched-run mean paired effects. The Overall row
uses the supplied equal-dataset mean and hierarchical-bootstrap 95% interval.
No inferential quantity is recomputed by this script.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "matplotlib_config"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator


DATASETS = (
    "Cao",
    "Quake_10x",
    "Quake_Smart-seq2",
    "Wagner",
    "Zeisel_2018",
)
ENDPOINTS = (
    ("joint_correct_known", "A  Joint correct-known", 2),
    ("novelty_ap", "B  Novelty AP", 4),
    ("full_prednovel_ami", "C  Full-PredNovel AMI", 3),
)
CONTRASTS = (
    ("REAL_CL-DEPTH_SHUFFLED_CL", "Real - Shuffled", "#3B6E8F", "o", -0.13),
    ("REAL_CL-GENERIC_STAR", "Real - Star", "#A9653F", "s", 0.13),
)

DATASET_REQUIRED_COLUMNS = {
    "dataset",
    "contrast",
    "endpoint",
    "n_paired_seeds",
    "effect",
    "seed_sd",
    "status",
}
OVERALL_REQUIRED_COLUMNS = {
    "contrast",
    "endpoint",
    "planned",
    "effect",
    "ci_low",
    "ci_high",
    "n_datasets",
    "n_pairs",
    "complete_5x4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-csv",
        type=Path,
        default=ROOT / "supplementary_data" / "paired_effects_by_dataset.csv",
    )
    parser.add_argument(
        "--overall-csv",
        type=Path,
        default=ROOT / "supplementary_data" / "paired_effects_and_CI.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    return parser.parse_args()


def read_csv(path: Path, required: set[str]) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    if not path.is_file():
        raise FileNotFoundError(f"Authority CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = tuple(reader.fieldnames or ())
        missing = required.difference(columns)
        if missing:
            raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Authority CSV is empty: {path}")
    return rows, columns


def unique_row(rows: Iterable[Mapping[str, str]], *, source: str, **keys: str) -> Mapping[str, str]:
    matches = [row for row in rows if all(row[key] == value for key, value in keys.items())]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one row in {source} for {keys}, found {len(matches)}"
        )
    return matches[0]


def finite_float(row: Mapping[str, str], column: str, context: str) -> float:
    try:
        value = float(row[column])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {column!r} in {context}: {row.get(column)!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {column!r} in {context}: {value}")
    return value


def truthy(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def validate_and_collect(
    dataset_rows: list[dict[str, str]], overall_rows: list[dict[str, str]]
) -> dict[str, dict[str, dict[str, object]]]:
    collected: dict[str, dict[str, dict[str, object]]] = {}
    for endpoint, _, _ in ENDPOINTS:
        collected[endpoint] = {}
        for contrast, _, _, _, _ in CONTRASTS:
            values: list[float] = []
            selected_rows: list[Mapping[str, str]] = []
            for dataset in DATASETS:
                context = f"dataset={dataset}, contrast={contrast}, endpoint={endpoint}"
                row = unique_row(
                    dataset_rows,
                    source="paired_effects_by_dataset.csv",
                    dataset=dataset,
                    contrast=contrast,
                    endpoint=endpoint,
                )
                if int(row["n_paired_seeds"]) != 4:
                    raise ValueError(f"Expected four matched runs for {context}: {row}")
                if row["status"] != "FOUR_SEEDS_COMPLETE":
                    raise ValueError(f"Incomplete matched-run row for {context}: {row}")
                values.append(finite_float(row, "effect", context))
                selected_rows.append(row)

            context = f"overall contrast={contrast}, endpoint={endpoint}"
            overall = unique_row(
                overall_rows,
                source="paired_effects_and_CI.csv",
                contrast=contrast,
                endpoint=endpoint,
            )
            if not truthy(overall["planned"]):
                raise ValueError(f"Primary contrast is not marked planned for {context}: {overall}")
            if int(overall["n_datasets"]) != len(DATASETS) or int(overall["n_pairs"]) != 20:
                raise ValueError(f"Unexpected aggregation counts for {context}: {overall}")
            if not truthy(overall["complete_5x4"]):
                raise ValueError(f"Incomplete 5x4 summary for {context}: {overall}")

            effect = finite_float(overall, "effect", context)
            ci_low = finite_float(overall, "ci_low", context)
            ci_high = finite_float(overall, "ci_high", context)
            if not ci_low <= effect <= ci_high:
                raise ValueError(f"Overall effect lies outside its CI for {context}: {overall}")

            equal_dataset_mean = sum(values) / len(values)
            if not math.isclose(equal_dataset_mean, effect, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    f"Overall effect is not the equal-dataset mean for {context}: "
                    f"CSV={effect}, calculated check={equal_dataset_mean}"
                )

            collected[endpoint][contrast] = {
                "values": values,
                "dataset_rows": selected_rows,
                "overall": effect,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "overall_row": overall,
            }
    return collected


def padded_limits(values: Iterable[float]) -> tuple[float, float]:
    values = tuple(values)
    low = min((*values, 0.0))
    high = max((*values, 0.0))
    span = high - low
    if span <= 0:
        span = max(abs(low), abs(high), 1e-3)
    padding = 0.10 * span
    return low - padding, high + padding


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9.2,
            "axes.titlesize": 9.8,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.7,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "scOLAR-figure3-primary-ontology",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def plot(collected: dict[str, dict[str, dict[str, object]]], output_dir: Path) -> None:
    configure_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), sharey=False)
    fig.subplots_adjust(left=0.105, right=0.992, bottom=0.22, top=0.78, wspace=0.56)

    y_base = list(range(len(DATASETS) + 1))
    y_labels = [*DATASETS, "Overall"]

    for ax, (endpoint, title, decimals) in zip(axes, ENDPOINTS):
        bounds: list[float] = [0.0]
        for contrast, _, color, marker, offset in CONTRASTS:
            summary = collected[endpoint][contrast]
            dataset_values = summary["values"]
            overall = float(summary["overall"])
            ci_low = float(summary["ci_low"])
            ci_high = float(summary["ci_high"])
            bounds.extend([*dataset_values, ci_low, ci_high])

            dataset_y = [position + offset for position in y_base[:-1]]
            ax.scatter(
                dataset_values,
                dataset_y,
                s=27,
                marker=marker,
                facecolor=color,
                edgecolor="white",
                linewidth=0.45,
                zorder=4,
            )
            overall_y = y_base[-1] + offset
            ax.errorbar(
                overall,
                overall_y,
                xerr=[[overall - ci_low], [ci_high - overall]],
                fmt=marker,
                color=color,
                ecolor=color,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.55,
                markersize=6.2,
                elinewidth=1.25,
                capsize=2.5,
                capthick=1.0,
                zorder=5,
            )

        ax.axvline(0.0, color="#777777", linewidth=0.8, zorder=1)
        ax.axhline(len(DATASETS) - 0.5, color="#B8B8B8", linewidth=0.65, zorder=1)
        ax.set_xlim(*padded_limits(bounds))
        ax.set_ylim(len(DATASETS) + 0.55, -0.55)
        ax.set_yticks(y_base, labels=y_labels)
        ax.tick_params(axis="y", length=0, pad=3.0)
        ax.tick_params(axis="x", length=2.8, width=0.55, pad=2.5)
        if endpoint == "novelty_ap":
            ap_ticks = [-0.003, -0.002, -0.001, 0.000, 0.001]
            x_low, x_high = ax.get_xlim()
            if not all(x_low <= tick <= x_high for tick in ap_ticks):
                raise ValueError(
                    f"Required Novelty AP ticks fall outside the data-safe range: "
                    f"range=({x_low}, {x_high}), ticks={ap_ticks}"
                )
            ax.set_xticks(ap_ticks)
        else:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.{decimals}f}"))
        ax.grid(axis="x", color="#E7E7E7", linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", fontweight="semibold", pad=8)
        ax.set_xlabel("Effect (Real - control)", labelpad=5)

        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#8A8A8A")
        ax.spines["bottom"].set_linewidth(0.6)
        for tick_label in ax.get_yticklabels():
            if tick_label.get_text() == "Overall":
                tick_label.set_fontweight("semibold")

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="none",
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=6.4,
            label=label,
        )
        for _, label, color, marker, _ in CONTRASTS
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=2,
        frameon=False,
        handletextpad=0.5,
        columnspacing=2.0,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "figure3_primary_ontology"
    pdf_metadata = {
        "Title": "Primary ontology-intervention effects",
        "Author": "",
        "Subject": "Paired effects for the six prespecified primary ontology contrasts",
        "Keywords": "",
        "Creator": "scOLAR Figure 3 plotting script",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025, metadata=pdf_metadata)
    fig.savefig(
        stem.with_suffix(".svg"),
        bbox_inches="tight",
        pad_inches=0.025,
        metadata={
            "Title": "Primary ontology-intervention effects",
            "Description": "Dataset-level paired effects and supplied overall hierarchical-bootstrap intervals.",
            "Creator": "scOLAR Figure 3 plotting script",
            "Date": None,
        },
    )
    fig.savefig(
        stem.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.025,
        metadata={"Software": "scOLAR Figure 3 plotting script"},
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset_rows, dataset_columns = read_csv(args.dataset_csv, DATASET_REQUIRED_COLUMNS)
    overall_rows, overall_columns = read_csv(args.overall_csv, OVERALL_REQUIRED_COLUMNS)
    collected = validate_and_collect(dataset_rows, overall_rows)
    plot(collected, args.output_dir)
    print(f"paired_effects_by_dataset.csv columns: {', '.join(dataset_columns)}")
    print(f"paired_effects_and_CI.csv columns: {', '.join(overall_columns)}")
    print("Validated 30 dataset-level paired effects and 6 supplied Overall effects/95% CIs.")
    print("Generated figures/figure3_primary_ontology.{pdf,svg,png}")


if __name__ == "__main__":
    main()
