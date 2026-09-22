# scOLAR — anonymous ICLR 2027 supplementary code

This review snapshot contains the executable method, the frozen 60-run
ontology-control workflow, its exact protocol configuration, split checksums,
and the machine-readable results used for the reported tables. It contains no
manuscript source, author metadata, repository history, raw public datasets, or
trained checkpoints.

## Package layout

- `scolar/`: released scOLAR implementation. `train.py` is the standalone
  training entry point; `compile_ontology.py` builds `data/cl.pt`.
- `scolar/frozen/`: frozen implementation modules used by the 60-run
  ontology-control experiment.
- `scripts/`: split preparation, ontology views, blinded training/inference,
  separate evaluation, aggregation, result validation, and Figure 3 plotting.
- `configs/protocol_locked.json`: exact prespecified 60-run configuration.
- `metadata/benchmark_split_manifest.csv`: target counts and split SHA-256
  values for every dataset/seed, derived from the completed run records.
- `supplementary_data/`: the four submitted machine-readable result tables.

`configs/protocol_locked.json` is immutable because its SHA-256 is embedded in
all 60 submitted result rows. Historical deadlines and operational notes inside
that provenance record do not alter the scientific configuration.

## Environment

The recorded environment was Debian GNU/Linux 12, Python 3.9.25, PyTorch
2.8.0 with CUDA 12.8, Scanpy 1.10.3, and AnnData 0.10.8. Create it with:

```bash
conda env create -f environment.yml
conda activate scolar
```

`requirements.txt` records the pinned Python packages. PyTorch wheels are
platform-specific; when the pinned wheel is unavailable, install the PyTorch
2.8 build matching the local CUDA driver, then install the remaining pins.

## Public data and ontology

Follow `data/README.md`. Five public `.h5ad` files and Cell Ontology release
2025-12-17 must be downloaded. The large public inputs are not bundled. The
ontology source is checksum-pinned but omitted because the upstream OBO embeds
third-party contributor identifiers. Compile it before running experiments:

```bash
python scolar/compile_ontology.py --input data/cl.obo --output data/cl.pt
```

## Preprocessing and frozen splits

For each dataset, preparation reads `obs["cell_ontology_class"]`, reproduces
the fixed open-set class registry, makes mutually exclusive source/target cell
splits for seeds 101, 202, 303, and 404, and writes a blinded tensor file plus
a separate truth file. Expression-only normalization, log transformation, HVG
selection (2,000 genes), size factors, and gene standardization use the fixed
transductive source-plus-unlabeled-target protocol. Target annotations are not
available to training, calibration, checkpoint selection, Leiden, or LCC.

```bash
python scripts/prepare_inputs.py --dataset Cao
python scripts/ontology_adapter.py --dataset Cao
```

Generated split hashes should match
`metadata/benchmark_split_manifest.csv`. The snapshot does not include cell
identifiers from the public datasets; the deterministic preparation code and
the submitted hashes are the verification authority.

## One representative benchmark

After downloading `Cao.h5ad`, compiling the ontology, and running the two
preparation commands above:

```bash
python scripts/blind_train.py --dataset Cao --seed 101 --arm REAL_CL
python scripts/evaluate_run.py --run-dir runs/Cao/seed101/REAL_CL
```

Training intentionally refuses a silent CPU fallback and requires one CUDA
GPU. It runs the fixed 100-epoch schedule; this is not a cheap smoke test.

The standalone released implementation can also be inspected with:

```bash
python scolar/train.py --help
```

## Full benchmark suite

The complete matrix is five datasets × four seeds × three ontology arms = 60
runs. Inspect the exact commands without executing them:

```bash
python scripts/run_suite.py
```

After all public inputs are present, execute sequentially with:

```bash
python scripts/run_suite.py --execute
```

This runner is deliberately sequential. Parallel GPU scheduling was an
operational choice in the original execution and is not encoded here. A failed
or partial run is preserved rather than silently retried.

## Calibration, PredNovel grouping, and LCC

Calibration is reference-only. Twenty fixed exclusion draws mask
`m = min(10, K - 1)` of the `K` mapped source classes per draw; the threshold
is refreshed at epochs 60, 70, 80, 90, and
100, then recalculated after reloading the prespecified epoch-100 checkpoint.
The target labels never participate.

A target cell is `PredNovel` when its routing MLS—the maximum logit over the
full ontology-indexed classifier head—is below the calibrated threshold.
When a cell is accepted as known, its label is selected only among classifier
rows mapped to source classes. PredNovel cells are grouped without a supplied class
count using Leiden with 15 neighbours, resolution 1.0, two iterations,
undirected igraph flavour, and random state 0. LCC reuses that partition. For
eligible clusters (at least five cells, with at least ten PredNovel cells in
total), source probabilities are projected to ontology ancestors with
temperature 2.0; the top-three coverage must reach 0.40 and the selected
context must have depth at least two.

The aligned true-novel/true-K diagnostics in `evaluate_run.py` are explicitly
post-hoc benchmark diagnostics, not deployment behaviour.

## Ontology controls

Use `--arm REAL_CL`, `--arm DEPTH_SHUFFLED_CL`, or `--arm GENERIC_STAR` with
`scripts/blind_train.py`. Depth-shuffled mappings are deterministic within
exact-depth source strata and are materialized by `ontology_adapter.py`.
Generic-star retains canonical classifier rows while replacing biological
ancestry with a synthetic root and one leaf per source class. The complete
definitions and seeds are in `configs/protocol_locked.json`.

## Tables, figures, and supplied-result checks

After a newly completed 60-run matrix, regenerate run-level and aggregate
tables with:

```bash
python scripts/aggregate_reports.py
```

The aggregator deterministically discovers the expected 60 files at
`metrics/<dataset>/seed<seed>/<arm>/result.json`. It does not require a run
manifest, scheduler registry, PID state, or other operational artifacts.

Validate the bundled tables and regenerate the primary ontology-control figure:

```bash
python scripts/verify_results.py
python scripts/plot_figure3_primary_ontology.py
```

The plotting script validates all 30 dataset-level effects and six overall
effects/intervals before writing PDF, SVG, and PNG files under `figures/`.

## Compute and limitations

Each training run uses one GPU. The recorded hardware was an NVIDIA RTX 3090
or RTX 4090 on a 96-core, 251-GiB RAM host. The package does not claim a fixed
wall-clock time because comparable run times were not retained in the four
submitted tables. Full reproduction requires downloading roughly 5 GB of
public AnnData inputs, the public ontology, CUDA-capable hardware, and 60
100-epoch runs. Checkpoints are not required to inspect the method and are not
included. Split cell-ID lists and per-run operational JSON are also absent;
deterministic split generation, split hashes, final aggregate CSVs, and all
evaluation code are included.
