#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def safe_read(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def bootstrap_ci(values, n=2000, seed=20260622):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    for i in range(n):
        means[i] = rng.choice(values, size=values.size, replace=True).mean()
    return np.percentile(means, [2.5, 97.5])


def trimmed_mean(values, lower=0.05, upper=0.95):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    if values.size < 20:
        return float(values.mean())
    lo, hi = np.quantile(values, [lower, upper])
    trimmed = values[(values >= lo) & (values <= hi)]
    return float(trimmed.mean()) if trimmed.size else float(values.mean())


def metric_summary(df, group_cols, method_col, time_col, label, success_col="success"):
    if df.empty or time_col not in df:
        return pd.DataFrame()
    rows = []
    for key, group in df.groupby(group_cols + [method_col], dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        vals = pd.to_numeric(group[time_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        success = pd.to_numeric(group.get(success_col, pd.Series(dtype=float)), errors="coerce").fillna(0)
        ci_low, ci_high = bootstrap_ci(vals.values)
        row = dict(zip(group_cols + [method_col], key))
        row.update({
            "table": label,
            "metric": time_col,
            "count": int(len(group)),
            "failure_count": int((success == 0).sum()) if len(success) else 0,
            "success_rate": float(success.mean()) if len(success) else np.nan,
            "mean": vals.mean(),
            "std": vals.std(ddof=1),
            "median": vals.median(),
            "p25": vals.quantile(0.25),
            "p75": vals.quantile(0.75),
            "p95": vals.quantile(0.95),
            "bootstrap_mean_ci_low": ci_low,
            "bootstrap_mean_ci_high": ci_high,
            "successful_trials_only": False,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def paired_speedup(df, pair_cols, method_col, legacy_name, hom_name, time_col, label, extra_group_cols):
    if df.empty or time_col not in df:
        return pd.DataFrame()
    keep = pair_cols + extra_group_cols + [method_col, time_col]
    data = df[keep].copy()
    data[time_col] = pd.to_numeric(data[time_col], errors="coerce")
    wide = data.pivot_table(index=pair_cols + extra_group_cols, columns=method_col, values=time_col, aggfunc="first")
    if legacy_name not in wide or hom_name not in wide:
        return pd.DataFrame()
    wide = wide.replace([np.inf, -np.inf], np.nan).dropna(subset=[legacy_name, hom_name])
    if wide.empty:
        return pd.DataFrame()
    wide["paired_speedup"] = wide[legacy_name] / wide[hom_name]
    wide["paired_abs_diff"] = wide[legacy_name] - wide[hom_name]
    rows = []
    group_levels = list(range(len(pair_cols), len(pair_cols) + len(extra_group_cols)))
    for key, group in wide.groupby(level=group_levels, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        speed = group["paired_speedup"].replace([np.inf, -np.inf], np.nan).dropna()
        diff = group["paired_abs_diff"].replace([np.inf, -np.inf], np.nan).dropna()
        ci_low, ci_high = bootstrap_ci(speed.values)
        row = dict(zip(extra_group_cols, key))
        row.update({
            "table": label,
            "metric": time_col,
            "method": "paired_speedup_legacy_over_hom",
            "count": int(len(group)),
            "failure_count": np.nan,
            "success_rate": np.nan,
            "mean": speed.mean(),
            "std": speed.std(ddof=1),
            "median": speed.median(),
            "p25": speed.quantile(0.25),
            "p75": speed.quantile(0.75),
            "p95": speed.quantile(0.95),
            "paired_abs_diff_mean": diff.mean(),
            "paired_abs_diff_median": diff.median(),
            "bootstrap_mean_ci_low": ci_low,
            "bootstrap_mean_ci_high": ci_high,
            "successful_trials_only": False,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def write_markdown(summary, path):
    cols = [
        "table", "density", "seed_type", "seed_length", "corridor_mode", "metric", "method",
        "count", "success_rate", "median", "p25", "p75", "p95", "mean",
        "bootstrap_mean_ci_low", "bootstrap_mean_ci_high",
    ]
    existing = [c for c in cols if c in summary.columns]
    with path.open("w") as f:
        f.write("# FIri Benchmark Summary\n\n")
        if summary.empty:
            f.write("No rows found.\n")
            return
        for table, group in summary.groupby("table", dropna=False):
            f.write(f"## {table}\n\n")
            table_df = group[existing].fillna("")
            f.write("| " + " | ".join(existing) + " |\n")
            f.write("| " + " | ".join(["---"] * len(existing)) + " |\n")
            for _, row in table_df.iterrows():
                values = [str(row[col]).replace("|", "\\|") for col in existing]
                f.write("| " + " | ".join(values) + " |\n")
            f.write("\n\n")


def plot_summary(summary, output_dir):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if summary.empty:
        return
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    for table in sorted(summary["table"].dropna().unique()):
        group = summary[(summary["table"] == table) & summary["method"].isin(["firi_legacy", "firi_hom"])]
        if group.empty:
            continue
        for metric in sorted(group["metric"].dropna().unique()):
            g = group[group["metric"] == metric].copy()
            label_cols = [c for c in ["density", "seed_type", "seed_length", "corridor_mode"] if c in g.columns]
            g["label"] = g[label_cols].astype(str).agg("/".join, axis=1)
            pivot = g.pivot_table(index="label", columns="method", values="median", aggfunc="first")
            if pivot.empty:
                continue
            ax = pivot.plot(kind="bar", figsize=(max(8, 0.45 * len(pivot)), 4.5))
            ax.set_title(f"{table}: {metric} median")
            ax.set_ylabel("ms")
            ax.grid(axis="y", alpha=0.25)
            plt.tight_layout()
            plt.savefig(plot_dir / f"{table}_{metric}_median.png", dpi=180)
            plt.close()


def write_table_iv(region, output_dir, time_col="region_core_ms", output_stem="table_iv_computation_time"):
    cols = ["density", "seed_type", "seed_length", "method", time_col]
    if region.empty or any(c not in region.columns for c in cols):
        return pd.DataFrame()
    data = region.copy()
    data[time_col] = pd.to_numeric(data[time_col], errors="coerce")
    rows = []
    for key, group in data.groupby(["seed_type", "seed_length", "density", "method"], dropna=False):
        vals = group[time_col].replace([np.inf, -np.inf], np.nan).dropna()
        seed_type, seed_length, density, method = key
        rows.append({
            "seed_type": seed_type,
            "seed_length": seed_length,
            "density": density,
            "method": method,
            "avg_ms": vals.mean(),
            "std_ms": vals.std(ddof=1),
            "min_ms": vals.min(),
            "max_ms": vals.max(),
            "count": int(len(group)),
            "success_rate": pd.to_numeric(group.get("success", pd.Series(dtype=float)), errors="coerce").fillna(0).mean(),
            "time_field": time_col,
        })
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table.to_csv(output_dir / f"{output_stem}.csv", index=False)

    with (output_dir / f"{output_stem}.md").open("w") as f:
        f.write("# TABLE IV Style Computation Time\n\n")
        f.write(f"Time field: `{time_col}`. Values are milliseconds.\n\n")
        for seed_type, seed_group in table.groupby("seed_type", dropna=False):
            f.write(f"## Seed Type: {seed_type}\n\n")
            pivot_rows = []
            for _, row in seed_group.iterrows():
                label = row["density"]
                if str(seed_type) == "line":
                    label = f"{label} / L={row['seed_length']}"
                pivot_rows.append({
                    "scenario": label,
                    "method": row["method"],
                    "avg": row["avg_ms"],
                    "std": row["std_ms"],
                    "min": row["min_ms"],
                    "max": row["max_ms"],
                    "success_rate": row["success_rate"],
                })
            df = pd.DataFrame(pivot_rows).fillna("")
            headers = list(df.columns)
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
            for _, r in df.iterrows():
                f.write("| " + " | ".join(str(r[h]) for h in headers) + " |\n")
            f.write("\n")
    return table


def write_fig7_volume_ratio(region, output_dir, baseline="firi_legacy",
                            min_volume=1.0e-6, plot_stat="median_ratio"):
    required = ["map_id", "case_id", "repeat_id", "density", "seed_type", "seed_length", "method", "polytope_volume"]
    if region.empty or any(c not in region.columns for c in required):
        return pd.DataFrame()
    data = region.copy()
    data["polytope_volume"] = pd.to_numeric(data["polytope_volume"], errors="coerce")
    if "success" in data.columns:
        data = data[pd.to_numeric(data["success"], errors="coerce").fillna(0) == 1]
    wide = data.pivot_table(
        index=["map_id", "case_id", "repeat_id", "density", "seed_type", "seed_length"],
        columns="method",
        values="polytope_volume",
        aggfunc="first",
    )
    if baseline not in wide.columns:
        return pd.DataFrame()
    rows = []
    for method in wide.columns:
        if method == baseline:
            valid = wide[[baseline]].replace([np.inf, -np.inf], np.nan).dropna()
            valid = valid[valid[baseline] > min_volume]
            ratios = pd.Series(np.ones(len(valid)), index=valid.index)
        else:
            valid = wide[[baseline, method]].replace([np.inf, -np.inf], np.nan).dropna()
            valid = valid[(valid[baseline] > min_volume) & (valid[method] > min_volume)]
            ratios = valid[method] / valid[baseline]
        if valid.empty:
            continue
        meta = valid.reset_index()[["density", "seed_type", "seed_length"]]
        ratio_df = meta.copy()
        ratio_df["ratio"] = ratios.values
        ratio_df = ratio_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["ratio"])
        ratio_df = ratio_df[ratio_df["ratio"] >= 0.0]
        for key, group in ratio_df.groupby(["density", "seed_type", "seed_length"], dropna=False):
            density, seed_type, seed_length = key
            vals = group["ratio"].dropna()
            ci_low, ci_high = bootstrap_ci(vals.values)
            rows.append({
                "density": density,
                "seed_type": seed_type,
                "seed_length": seed_length,
                "method": method,
                "baseline": baseline,
                "mean_ratio": vals.mean(),
                "trimmed_mean_ratio": trimmed_mean(vals.values),
                "std_ratio": vals.std(ddof=1),
                "median_ratio": vals.median(),
                "p25_ratio": vals.quantile(0.25),
                "p75_ratio": vals.quantile(0.75),
                "p95_ratio": vals.quantile(0.95),
                "bootstrap_mean_ci_low": ci_low,
                "bootstrap_mean_ci_high": ci_high,
                "count": int(len(vals)),
                "min_volume": min_volume,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out.to_csv(output_dir / "fig7_volume_ratio.csv", index=False)
    try:
        import matplotlib.pyplot as plt
        plot_dir = output_dir / "plots"
        plot_dir.mkdir(exist_ok=True)
        labels = []
        methods = sorted(out["method"].unique())
        pivot_data = {}
        for _, row in out.iterrows():
            label = f"{row['density']}/{row['seed_type']}"
            if str(row["seed_type"]) == "line":
                label += f"/L={row['seed_length']}"
            labels.append(label)
            pivot_data.setdefault(row["method"], {})[label] = row.get(plot_stat, row["median_ratio"])
        labels = sorted(set(labels))
        x = np.arange(len(labels))
        width = 0.8 / max(1, len(methods))
        fig, ax = plt.subplots(figsize=(max(8, 0.65 * len(labels)), 4.5))
        for i, method in enumerate(methods):
            vals = [pivot_data.get(method, {}).get(label, np.nan) for label in labels]
            ax.bar(x + (i - (len(methods) - 1) / 2) * width, vals, width, label=method)
        ax.axhline(1.0, linestyle="--", color="black", linewidth=1.0)
        ax.set_ylabel(f"{plot_stat.replace('_', ' ')} to {baseline}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_dir / "fig7_volume_ratio.png", dpi=180)
        plt.close(fig)
    except Exception:
        pass
    return out


def write_mvie_mechanism_summary(replay, output_dir):
    metrics = [
        "iterations",
        "objective_evaluations",
        "solve_ms",
        "max_constraint_residual",
        "log_volume_gap",
    ]
    required = ["density", "seed_type", "solver", "success"] + metrics
    if replay.empty or any(c not in replay.columns for c in required):
        return pd.DataFrame()

    data = replay.copy()
    for col in metrics + ["success"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    rows = []
    for key, group in data.groupby(["density", "seed_type", "solver"], dropna=False):
        density, seed_type, solver = key
        row = {
            "density": density,
            "seed_type": seed_type,
            "solver": solver,
            "count": int(len(group)),
            "success_rate": float(group["success"].fillna(0).mean()),
        }
        for metric in metrics:
            vals = group[metric].replace([np.inf, -np.inf], np.nan).dropna()
            row[f"{metric}_median"] = vals.median()
            row[f"{metric}_mean"] = vals.mean()
            row[f"{metric}_p25"] = vals.quantile(0.25)
            row[f"{metric}_p75"] = vals.quantile(0.75)
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out.to_csv(output_dir / "mvie_mechanism_summary.csv", index=False)

    display_cols = [
        "density", "seed_type", "solver", "count", "success_rate",
        "iterations_median", "objective_evaluations_median", "solve_ms_median",
        "max_constraint_residual_median", "log_volume_gap_median",
    ]
    with (output_dir / "mvie_mechanism_summary.md").open("w") as f:
        f.write("# MVIE LBFGS Mechanism Summary\n\n")
        f.write("Values are grouped by replayed half-space input and solver. ")
        f.write("`log_volume_gap` is measured against the best feasibility-scaled `logdet_l` in the paired replay.\n\n")
        f.write("| " + " | ".join(display_cols) + " |\n")
        f.write("| " + " | ".join(["---"] * len(display_cols)) + " |\n")
        for _, row in out[display_cols].iterrows():
            f.write("| " + " | ".join(str(row[col]) for col in display_cols) + " |\n")
    return out


def write_planning_time_summary(planning, output_dir):
    metrics = [
        "path_search_ms_shared",
        "surface_extract_ms_shared",
        "corridor_total_ms",
        "trajectory_setup_ms",
        "trajectory_optimize_ms",
        "planning_backend_ms",
        "end_to_end_ms",
        "regions",
        "route_length",
        "trajectory_duration",
    ]
    required = ["density", "method", "success"] + metrics
    if planning.empty or any(c not in planning.columns for c in required):
        return pd.DataFrame(), pd.DataFrame()

    data = planning.copy()
    for col in metrics + [
        "success",
        "path_success",
        "corridor_success",
        "trajectory_optimize_success",
        "trajectory_collision_free",
    ]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    data["sfc_pipeline_ms"] = (
        data["path_search_ms_shared"] +
        data["surface_extract_ms_shared"] +
        data["corridor_total_ms"]
    )

    rows = []
    for key, group in data.groupby(["density", "method"], dropna=False):
        density, method = key
        successful = group[group["success"].fillna(0) == 1]
        corridor_successful = group[(group.get("path_success", 0).fillna(0) == 1) &
                                    (group.get("corridor_success", 0).fillna(0) == 1)]
        row = {
            "density": density,
            "method": method,
            "count": int(len(group)),
            "success_count": int(len(successful)),
            "success_rate": float(group["success"].fillna(0).mean()),
            "path_success_rate": float(group.get("path_success", pd.Series(dtype=float)).fillna(0).mean()),
            "corridor_success_rate": float(group.get("corridor_success", pd.Series(dtype=float)).fillna(0).mean()),
            "trajectory_optimize_success_rate": float(group.get("trajectory_optimize_success", pd.Series(dtype=float)).fillna(0).mean()),
            "trajectory_collision_free_rate": float(group.get("trajectory_collision_free", pd.Series(dtype=float)).fillna(0).mean()),
        }
        sfc_vals = corridor_successful["sfc_pipeline_ms"].replace([np.inf, -np.inf], np.nan).dropna()
        row["sfc_success_count"] = int(len(corridor_successful))
        row["sfc_pipeline_ms_median"] = sfc_vals.median()
        row["sfc_pipeline_ms_mean"] = sfc_vals.mean()
        row["sfc_pipeline_ms_p25"] = sfc_vals.quantile(0.25)
        row["sfc_pipeline_ms_p75"] = sfc_vals.quantile(0.75)
        for metric in metrics:
            vals = successful[metric].replace([np.inf, -np.inf], np.nan).dropna()
            row[f"{metric}_median_success"] = vals.median()
            row[f"{metric}_mean_success"] = vals.mean()
            row[f"{metric}_p25_success"] = vals.quantile(0.25)
            row[f"{metric}_p75_success"] = vals.quantile(0.75)
        rows.append(row)

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary.to_csv(output_dir / "planning_time_summary.csv", index=False)
        display_cols = [
            "density", "method", "count", "sfc_success_count", "success_count", "success_rate",
            "sfc_pipeline_ms_median",
            "planning_backend_ms_median_success", "end_to_end_ms_median_success",
            "corridor_total_ms_median_success", "trajectory_optimize_ms_median_success",
            "regions_median_success", "trajectory_duration_median_success",
        ]
        with (output_dir / "planning_time_summary.md").open("w") as f:
            f.write("# Complete Planning Time Summary\n\n")
            f.write("Timing columns are computed from successful complete planning trials only; failures are reflected in `success_rate`.\n\n")
            f.write("| " + " | ".join(display_cols) + " |\n")
            f.write("| " + " | ".join(["---"] * len(display_cols)) + " |\n")
            for _, row in summary[display_cols].iterrows():
                f.write("| " + " | ".join(str(row[col]) for col in display_cols) + " |\n")

    pair_rows = []
    pair_index = ["map_id", "planning_case_id", "repeat_id", "density"]
    if all(c in data.columns for c in pair_index + ["method"]):
        wide_success = data.pivot_table(index=pair_index, columns="method", values="success", aggfunc="first")
        for metric in ["sfc_pipeline_ms", "planning_backend_ms", "end_to_end_ms", "corridor_total_ms", "trajectory_optimize_ms"]:
            wide = data.pivot_table(index=pair_index, columns="method", values=metric, aggfunc="first")
            if "firi_legacy" not in wide or "firi_hom" not in wide or wide_success.empty:
                continue
            paired = wide.join(wide_success, rsuffix="_success")
            needed = ["firi_legacy", "firi_hom", "firi_legacy_success", "firi_hom_success"]
            paired = paired.replace([np.inf, -np.inf], np.nan).dropna(subset=needed)
            if metric == "sfc_pipeline_ms":
                wide_path = data.pivot_table(index=pair_index, columns="method", values="path_success", aggfunc="first")
                wide_corridor = data.pivot_table(index=pair_index, columns="method", values="corridor_success", aggfunc="first")
                paired = paired.join(wide_path, rsuffix="_path").join(wide_corridor, rsuffix="_corridor")
                paired = paired[(paired["firi_legacy_path"] == 1) &
                                (paired["firi_hom_path"] == 1) &
                                (paired["firi_legacy_corridor"] == 1) &
                                (paired["firi_hom_corridor"] == 1)]
            else:
                paired = paired[(paired["firi_legacy_success"] == 1) & (paired["firi_hom_success"] == 1)]
            paired = paired[(paired["firi_legacy"] > 0.0) & (paired["firi_hom"] > 0.0)]
            if paired.empty:
                continue
            paired["speedup_legacy_over_hom"] = paired["firi_legacy"] / paired["firi_hom"]
            paired["abs_diff_ms"] = paired["firi_legacy"] - paired["firi_hom"]
            for density, group in paired.groupby(level="density", dropna=False):
                speed = group["speedup_legacy_over_hom"].dropna()
                diff = group["abs_diff_ms"].dropna()
                pair_rows.append({
                    "density": density,
                    "metric": metric,
                    "paired_success_count": int(len(group)),
                    "speedup_median": speed.median(),
                    "speedup_mean": speed.mean(),
                    "speedup_p25": speed.quantile(0.25),
                    "speedup_p75": speed.quantile(0.75),
                    "abs_diff_ms_median": diff.median(),
                    "abs_diff_ms_mean": diff.mean(),
                })

    speedup = pd.DataFrame(pair_rows)
    if not speedup.empty:
        speedup.to_csv(output_dir / "planning_paired_speedup.csv", index=False)
        display_cols = [
            "density", "metric", "paired_success_count", "speedup_median",
            "speedup_mean", "abs_diff_ms_median", "abs_diff_ms_mean",
        ]
        with (output_dir / "planning_paired_speedup.md").open("w") as f:
            f.write("# Complete Planning Paired Speedup\n\n")
            f.write("Pairs include only trials where both methods completed successfully.\n\n")
            f.write("| " + " | ".join(display_cols) + " |\n")
            f.write("| " + " | ".join(["---"] * len(display_cols)) + " |\n")
            for _, row in speedup[display_cols].iterrows():
                f.write("| " + " | ".join(str(row[col]) for col in display_cols) + " |\n")

    return summary, speedup


def main():
    parser = argparse.ArgumentParser(description="Summarize and plot FIri benchmark CSV outputs.")
    parser.add_argument("input_dir")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--volume-min", type=float, default=1.0e-6,
                        help="Minimum positive polytope volume used in Fig. 7 paired ratios.")
    parser.add_argument("--volume-plot-stat", default="median_ratio",
                        choices=["mean_ratio", "trimmed_mean_ratio", "median_ratio"],
                        help="Statistic used for the Fig. 7 bar plot.")
    args = parser.parse_args()
    output_dir = Path(args.input_dir)

    region = safe_read(output_dir / "region_trials.csv")
    replay = safe_read(output_dir / "mvie_replay.csv")
    corridor = safe_read(output_dir / "corridor_trials.csv")
    planning = safe_read(output_dir / "planning_trials.csv")

    summaries = []
    summaries.append(metric_summary(region, ["density", "seed_type", "seed_length"], "method", "region_core_ms", "region"))
    summaries.append(metric_summary(region, ["density", "seed_type", "seed_length"], "method", "region_online_ms", "region"))
    for replay_metric in [
        "iterations",
        "objective_evaluations",
        "solve_ms",
        "max_constraint_residual",
        "log_volume_gap",
    ]:
        summaries.append(metric_summary(replay, ["density", "seed_type", "seed_length"],
                                        "solver", replay_metric, "mvie_replay"))
    summaries.append(metric_summary(corridor, ["density", "corridor_mode"], "method", "corridor_total_ms", "corridor"))
    summaries.append(metric_summary(planning, ["density"], "method", "planning_backend_ms", "planning"))
    summaries.append(metric_summary(planning, ["density"], "method", "end_to_end_ms", "planning"))

    summaries.append(paired_speedup(region, ["map_id", "case_id", "repeat_id"], "method",
                                    "firi_legacy", "firi_hom", "region_core_ms", "region",
                                    ["density", "seed_type", "seed_length"]))
    summaries.append(paired_speedup(replay, ["map_id", "region_case_id", "outer_iteration", "repeat_id"], "solver",
                                    "firi_legacy", "firi_hom", "solve_ms", "mvie_replay",
                                    ["density", "seed_type", "seed_length"]))
    summaries.append(paired_speedup(corridor, ["map_id", "planning_case_id", "repeat_id"], "method",
                                    "firi_legacy", "firi_hom", "corridor_total_ms", "corridor",
                                    ["density", "corridor_mode"]))
    summaries.append(paired_speedup(planning, ["map_id", "planning_case_id", "repeat_id"], "method",
                                    "firi_legacy", "firi_hom", "planning_backend_ms", "planning",
                                    ["density"]))

    summary = pd.concat([s for s in summaries if s is not None and not s.empty], ignore_index=True, sort=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    write_markdown(summary, output_dir / "summary.md")
    plot_summary(summary, output_dir)
    write_table_iv(region, output_dir, "region_core_ms", "table_iv_computation_time")
    write_table_iv(region, output_dir, "region_online_ms", "table_iv_online_time")
    write_fig7_volume_ratio(region, output_dir, min_volume=args.volume_min,
                            plot_stat=args.volume_plot_stat)
    write_mvie_mechanism_summary(replay, output_dir)
    write_planning_time_summary(planning, output_dir)


if __name__ == "__main__":
    main()
