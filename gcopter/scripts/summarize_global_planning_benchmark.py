#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def read_trials(root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(root.rglob("global_planning_trials.csv")):
        df = pd.read_csv(path)
        df["source_file"] = str(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def numeric(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def summarize_methods(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    time_cols = [
        "path_search_ms",
        "surface_extract_ms",
        "corridor_total_ms",
        "trajectory_setup_ms",
        "trajectory_optimize_ms",
        "end_to_end_ms",
    ]
    bool_cols = [
        "path_success",
        "corridor_success",
        "trajectory_setup_success",
        "trajectory_optimize_success",
        "trajectory_collision_free",
        "global_planning_success",
        "success",
    ]
    for (density, method), group in df.groupby(["density", "method"], dropna=False):
        row = {
            "density": density,
            "method": method,
            "rows": len(group),
            "cases": group["planning_case_id"].nunique(),
        }
        for col in bool_cols:
            values = pd.to_numeric(group[col], errors="coerce").fillna(0.0)
            row[f"{col}_rate"] = float(values.mean()) if len(values) else np.nan
        success_group = group[pd.to_numeric(group["global_planning_success"], errors="coerce").fillna(0) == 1]
        for col in time_cols:
            values = numeric(success_group, col).dropna()
            row[f"{col}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{col}_median"] = float(values.median()) if len(values) else np.nan
            row[f"{col}_p95"] = float(values.quantile(0.95)) if len(values) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_paired(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    time_cols = ["corridor_total_ms", "trajectory_optimize_ms", "end_to_end_ms"]
    pair_cols = ["density", "planning_case_id"]
    for time_col in time_cols:
        data = df[pair_cols + ["method", time_col, "global_planning_success"]].copy()
        data = data[pd.to_numeric(data["global_planning_success"], errors="coerce").fillna(0) == 1]
        data[time_col] = numeric(data, time_col)
        wide = data.pivot_table(index=pair_cols, columns="method", values=time_col, aggfunc="first")
        if "baseline_firi" not in wide.columns or "hom_mvie" not in wide.columns:
            continue
        wide = wide.dropna(subset=["baseline_firi", "hom_mvie"])
        wide["speedup_baseline_over_hom"] = wide["baseline_firi"] / wide["hom_mvie"]
        wide["abs_diff_ms"] = wide["baseline_firi"] - wide["hom_mvie"]
        for density, group in wide.groupby(level=0, dropna=False):
            speed = group["speedup_baseline_over_hom"].replace([np.inf, -np.inf], np.nan).dropna()
            diff = group["abs_diff_ms"].replace([np.inf, -np.inf], np.nan).dropna()
            rows.append({
                "density": density,
                "metric": time_col,
                "paired_cases": int(len(group)),
                "speedup_mean": float(speed.mean()) if len(speed) else np.nan,
                "speedup_median": float(speed.median()) if len(speed) else np.nan,
                "speedup_p25": float(speed.quantile(0.25)) if len(speed) else np.nan,
                "speedup_p75": float(speed.quantile(0.75)) if len(speed) else np.nan,
                "abs_diff_ms_mean": float(diff.mean()) if len(diff) else np.nan,
                "abs_diff_ms_median": float(diff.median()) if len(diff) else np.nan,
            })
    return pd.DataFrame(rows)


def write_markdown(method_summary: pd.DataFrame, paired_summary: pd.DataFrame, path: Path) -> None:
    def emit_table(f, df: pd.DataFrame, cols) -> None:
        table = df[cols].copy()
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
        for _, row in table.iterrows():
            values = []
            for col in cols:
                value = row[col]
                if isinstance(value, float):
                    value = f"{value:.6g}"
                values.append(str(value).replace("|", "\\|"))
            f.write("| " + " | ".join(values) + " |\n")

    with path.open("w") as f:
        f.write("# Global Planning Benchmark Summary\n\n")
        if method_summary.empty:
            f.write("No global_planning_trials.csv files found.\n")
            return
        f.write("## Method Summary\n\n")
        cols = [
            "density", "method", "cases",
            "global_planning_success_rate", "success_rate",
            "end_to_end_ms_median", "end_to_end_ms_p95",
            "corridor_total_ms_median", "trajectory_optimize_ms_median",
        ]
        emit_table(f, method_summary, cols)
        f.write("\n\n## Paired Speedup\n\n")
        if paired_summary.empty:
            f.write("No paired rows available.\n")
        else:
            emit_table(f, paired_summary, list(paired_summary.columns))
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Directory containing per-density benchmark outputs")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or args.root
    output_dir.mkdir(parents=True, exist_ok=True)
    trials = read_trials(args.root)
    if trials.empty:
        raise SystemExit(f"No global_planning_trials.csv found under {args.root}")
    method_summary = summarize_methods(trials)
    paired_summary = summarize_paired(trials)
    trials.to_csv(output_dir / "global_planning_trials_all.csv", index=False)
    method_summary.to_csv(output_dir / "global_planning_summary.csv", index=False)
    paired_summary.to_csv(output_dir / "global_planning_paired_speedup.csv", index=False)
    write_markdown(method_summary, paired_summary, output_dir / "global_planning_summary.md")


if __name__ == "__main__":
    main()
