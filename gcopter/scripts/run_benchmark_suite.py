#!/usr/bin/env python3
import argparse
import json
import os
import platform
import socket
import subprocess
import time
from pathlib import Path


PRESETS = {
    "sparse": {
        "cylinder_ratio": 0.03,
        "circle_ratio": 0.00,
        "gate_ratio": 0.00,
        "ellip_ratio": 0.00,
        "poly_ratio": 0.003,
    },
    "medium": {
        "cylinder_ratio": 0.06,
        "circle_ratio": 0.00,
        "gate_ratio": 0.00,
        "ellip_ratio": 0.00,
        "poly_ratio": 0.006,
    },
    "dense": {
        "cylinder_ratio": 0.10,
        "circle_ratio": 0.01,
        "gate_ratio": 0.00,
        "ellip_ratio": 0.01,
        "poly_ratio": 0.010,
    },
}


def run(cmd, cwd=None):
    return subprocess.check_output(cmd, cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()


def git_info(repo):
    try:
        commit = run(["git", "rev-parse", "HEAD"], cwd=repo)
        dirty = run(["git", "status", "--short"], cwd=repo)
    except Exception:
        commit = "unknown"
        dirty = "unknown"
    return commit, dirty


def main():
    parser = argparse.ArgumentParser(description="Run the FIri benchmark suite over kr_param_map density presets.")
    parser.add_argument("--output-dir", default="/tmp/firi_benchmark")
    parser.add_argument("--densities", nargs="+", default=["sparse", "medium", "dense"], choices=sorted(PRESETS))
    parser.add_argument("--map-seeds-per-density", type=int, default=10)
    parser.add_argument("--master-seed", type=int, default=20260622)
    parser.add_argument("--benchmark-mode", default="all", choices=["all", "region", "mvie_replay", "corridor", "planning"])
    parser.add_argument("--point-seeds", type=int, default=50)
    parser.add_argument("--line-seeds-per-length", type=int, default=50)
    parser.add_argument("--planning-cases", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--resolution", type=float, default=0.25)
    parser.add_argument("--taskset-cpu", default=None, help="Optional CPU id for taskset pinning, e.g. 2.")
    parser.add_argument("--enable-visualization", action="store_true")
    parser.add_argument("--rviz", action="store_true")
    parser.add_argument("--manifest-in", default="")
    parser.add_argument("--manifest-out", default="")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    repo = Path(__file__).resolve().parents[2]
    commit, dirty = git_info(repo)
    runs = []

    for density_index, density in enumerate(args.densities):
        preset = PRESETS[density]
        for seed_index in range(args.map_seeds_per_density):
            map_seed = args.master_seed + density_index * 10000 + seed_index
            run_id = f"{density}_map{map_seed}_master{args.master_seed}"
            cmd = [
                "roslaunch",
                "gcopter",
                "firi_benchmark.launch",
                f"density_label:={density}",
                f"map_seed:={map_seed}",
                f"master_seed:={args.master_seed}",
                f"output_dir:={str(output_dir)}",
                f"benchmark_mode:={args.benchmark_mode}",
                f"run_id:={run_id}",
                f"resolution:={args.resolution}",
                f"point_seed_count:={args.point_seeds}",
                f"line_seed_count_per_length:={args.line_seeds_per_length}",
                f"planning_case_count:={args.planning_cases}",
                f"repeats:={args.repeats}",
                f"warmup_repeats:={args.warmup_repeats}",
                f"enable_visualization:={'true' if args.enable_visualization else 'false'}",
                f"rviz:={'true' if args.rviz else 'false'}",
            ]
            for key, value in preset.items():
                cmd.append(f"{key}:={value}")
            if args.taskset_cpu is not None:
                cmd = ["taskset", "-c", str(args.taskset_cpu)] + cmd

            start = time.time()
            print("Running:", " ".join(cmd), flush=True)
            subprocess.run(cmd, env=env, check=True)
            runs.append({
                "run_id": run_id,
                "density": density,
                "map_seed": map_seed,
                "elapsed_s": time.time() - start,
                "preset": preset,
            })

    metadata = {
        "git_commit": commit,
        "git_dirty_status": dirty,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "master_seed": args.master_seed,
        "thread_env": {k: env[k] for k in env if k.endswith("_NUM_THREADS") or k == "OMP_NUM_THREADS"},
        "densities": args.densities,
        "presets": {d: PRESETS[d] for d in args.densities},
        "runs": runs,
        "timestamp_unix": time.time(),
        "map_provider": "KumarRobotics/kr_param_map param_env::structure_map",
        "methods": [
            {"name": "firi_legacy", "constraint_builder": "FullFiri", "mvie_solver": "LegacyPenalty"},
            {"name": "firi_hom", "constraint_builder": "FullFiri", "mvie_solver": "HomGauge"},
        ],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
