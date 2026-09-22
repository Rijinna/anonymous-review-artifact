"""Print or execute the frozen 60-run ontology-control benchmark workflow."""

import argparse
import subprocess
import sys

DATASETS = ("Cao", "Quake_10x", "Quake_Smart-seq2", "Wagner", "Zeisel_2018")
SEEDS = (101, 202, 303, 404)
ARMS = ("REAL_CL", "DEPTH_SHUFFLED_CL", "GENERIC_STAR")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="run commands sequentially; default is a dry-run command listing")
    parser.add_argument("--dataset", choices=DATASETS, help="restrict training/evaluation to one prepared dataset")
    parser.add_argument("--skip-preparation", action="store_true", help="reuse verified splits and ontology mappings")
    args = parser.parse_args()
    datasets = (args.dataset,) if args.dataset else DATASETS
    commands = []
    if not args.skip_preparation:
        commands.extend(([sys.executable, "scripts/prepare_inputs.py", "--dataset", d] for d in datasets))
        if args.dataset:
            commands.append([sys.executable, "scripts/ontology_adapter.py", "--dataset", args.dataset])
        else:
            commands.append([sys.executable, "scripts/ontology_adapter.py"])
    for dataset in datasets:
        for seed in SEEDS:
            for arm in ARMS:
                run_dir = f"runs/{dataset}/seed{seed}/{arm}"
                commands.append([sys.executable, "scripts/blind_train.py", "--dataset", dataset, "--seed", str(seed), "--arm", arm])
                commands.append([sys.executable, "scripts/evaluate_run.py", "--run-dir", run_dir])
    commands.append([sys.executable, "scripts/aggregate_reports.py"])
    for command in commands:
        print(" ".join(command), flush=True)
        if args.execute:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
