"""Rebuild compact Phase 2 evidence and plots without training or loading pickle files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def compact_summary(report):
    if report.get("status") != "completed":
        raise ValueError("Only a completed evaluation can be reported as final evidence")
    audit = report["data_audit"]
    experiments = []
    for item in report["experiments"]:
        selected = item["selected_configuration"]["selected_model"]
        metric = "f1" if item["task"] == "classification" else "rmse"
        row = {key: item[key] for key in ("id", "task", "cohort", "mode", "policy")}
        row.update(
            model=selected,
            cv_metric=metric,
            cv=item["cv_candidates"][selected]["mean_std"][metric],
            holdout=item["selected_internal_holdout"],
            dummy=item["dummy_internal_holdout"],
            importance=item["permutation_importance"],
            stress={key: {name: value for name, value in stress.items()
                          if name in ("selected_metrics", "dummy_metrics", "rmse_skill_vs_dummy", "skipped")}
                    for key, stress in item["stress_tests"].items()},
        )
        if item["threshold"]:
            row["threshold"] = item["threshold"]["value"]
        if item["task"] == "regression":
            row["rmse_skill_vs_dummy"] = item["rmse_skill_vs_dummy"]
        experiments.append(row)
    views = audit["statistical_views"]
    return {
        "status": report["status"], "metadata": report["metadata"],
        "runtime_seconds": report["runtime_seconds"], "protocol": report["protocol"],
        "audit": {
            "raw_rows": audit["rows_read"], "labelled_rows": audit["rows_after_cleaning"],
            "blank_rows": audit["fully_blank_rows_removed"],
            "missing_target_nonblank": audit["missing_target_nonblank_rows_removed"],
            "target_counts": audit["target_distribution_after_cleaning"],
            "physical_findings": audit["labelled"], "duplicates": audit["duplicates"],
            "tds_turbidity_correlations": {
                name: view["pearson_correlations"]["TDS"]["Turbidity"] for name, view in views.items()
            },
        },
        "cohort_counts": {key: {
            "retained": value["audit"]["retained_count"],
            "conflicting_rows_removed": value["audit"]["conflicting_rows_removed_count"],
            "duplicate_rows_removed": value["audit"]["duplicate_rows_removed_count"],
            "train": len(value["split"]["train_row_ids"]),
            "test": len(value["split"]["test_row_ids"]),
        } for key, value in report["cohorts"].items()},
        "experiments": experiments,
    }


def _plots(report, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    source = "Source: supplied Aqua Epoch CSV; Phase 2 internal evaluation (not external validation)"

    def save(fig, name):
        fig.text(.01, .01, source, fontsize=8)
        fig.tight_layout(rect=(0, .05, 1, 1))
        fig.savefig(directory / name, dpi=130)
        plt.close(fig)
        paths.append(name)

    audit = report["data_audit"]
    features = audit["columns"][:-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    missing = audit["labelled"]["missing_per_column_after_coercion"]
    sanitized = audit["labelled"]["missing_per_column_after_physical_sanitization"]
    x = np.arange(len(features))
    ax.bar(x - .2, [missing[f] for f in features], .4, label="Raw missing/coerced")
    ax.bar(x + .2, [sanitized[f] for f in features], .4, label="After physical sanitization")
    ax.set(xticks=x, xticklabels=features, ylabel="Missing measurements (count)",
           xlabel="Input measurement (units undocumented)", title="Missing features among labelled samples")
    ax.tick_params(axis="x", rotation=75)
    ax.legend()
    save(fig, "missingness.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, name in zip(axes, ("raw", "physical_sanitized_spike_excluded")):
        corr = audit["statistical_views"][name]["pearson_correlations"]
        grid = np.array([[corr[a][b] for b in features] for a in features], dtype=float)
        image = ax.imshow(grid, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set(xticks=x, yticks=x, xticklabels=features, yticklabels=features,
               title=name.replace("_", " "), xlabel="Measurement", ylabel="Measurement")
        ax.tick_params(axis="x", rotation=90, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        fig.colorbar(image, ax=ax, label="Pearson correlation (unitless)", shrink=.6)
    fig.suptitle("Correlation sensitivity: exploratory spike exclusion, not training-row deletion")
    save(fig, "correlation_sensitivity.png")

    experiments = report["experiments"]
    classifications = [e for e in experiments if e["task"] == "classification" and e["policy"] == "physical_sanitized"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for row, e in enumerate(classifications):
        for col, key in enumerate(("baseline_threshold_0_5", "conservative_threshold")):
            scores = e["selected_internal_holdout"][key]
            c = scores["confusion_matrix"]
            grid = np.array([[c["actual_unsafe_predicted_unsafe"], c["actual_unsafe_predicted_safe"]],
                             [c["actual_safe_predicted_unsafe"], c["actual_safe_predicted_safe"]]])
            ax = axes[row, col]
            ax.imshow(grid, cmap="Blues")
            for (i, j), v in np.ndenumerate(grid):
                ax.text(j, i, str(v), ha="center", va="center", color="red")
            ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Unsafe", "Safe"],
                   yticklabels=["Unsafe", "Safe"], xlabel="Predicted class (sample counts)",
                   ylabel="Actual class", title=f'{e["mode"]}: threshold {scores["threshold"]:.3f}')
    fig.suptitle("Internal holdout confusion matrices: physical-sanitized models")
    save(fig, "confusion_matrices.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    for e in classifications:
        candidates = e["cv_candidates"]
        names = list(candidates)
        offset = -.16 if e["mode"] == "full" else .16
        ax.bar(np.arange(len(names)) + offset,
               [candidates[n]["mean_std"]["f1"]["mean"] for n in names], .32,
               yerr=[candidates[n]["mean_std"]["f1"]["std"] for n in names], label=e["mode"])
    ax.set(xticks=np.arange(len(names)), xticklabels=names, ylabel="Safe-class F1 (0–1)",
           xlabel="Model candidate", title="Training CV: mean F1 ± fold SD, threshold 0.5")
    ax.legend()
    save(fig, "classification_cv.png")

    full = next(e for e in classifications if e["mode"] == "full")
    importance = sorted(full["permutation_importance"]["features"].items(), key=lambda p: p[1]["mean"])[-10:]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh([name for name, _ in importance], [v["mean"] for _, v in importance],
            xerr=[v["std"] for _, v in importance])
    ax.set(xlabel="ROC-AUC decrease after permutation (mean ± repeat SD)", ylabel="Original input feature",
           title="Full sanitized classifier: holdout permutation importance (non-causal)")
    save(fig, "feature_importance.png")

    regressions = [e for e in experiments if e["task"] == "regression" and e["policy"] == "physical_sanitized"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, cohort in zip(axes, ("raw_targets", "physical_valid_targets")):
        subset = [e for e in regressions if e["cohort"] == cohort]
        for shift, key, label in ((-.17, "selected_internal_holdout", "CV-selected model"),
                                   (.17, "dummy_internal_holdout", "Training-mean baseline")):
            ax.bar(np.arange(len(subset)) + shift, [e[key]["rmse"] for e in subset], .34, label=label)
        ax.set(xticks=np.arange(len(subset)), xticklabels=[e["mode"] for e in subset],
               xlabel="Input mode", ylabel="RMSE (pH units)", title=cohort.replace("_", " "))
        ax.legend(fontsize=8)
    fig.suptitle("Regression vs dummy: different target cohorts, not an apples-to-apples improvement")
    save(fig, "regression_baselines.png")
    return paths


def generate_report(directory):
    """Read completed results; write a compact JSON, Markdown review, and six PNGs."""
    directory = Path(directory)
    report = json.loads((directory / "results.json").read_text())
    summary = compact_summary(report)
    plots_dir = directory / "plots"
    plots_dir.mkdir(exist_ok=True)
    figures = _plots(report, plots_dir)
    summary["figures"] = [f"plots/{name}" for name in figures]
    (directory / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (directory / "data_audit.json").write_text(json.dumps(report["data_audit"], indent=2, allow_nan=False) + "\n")
    audit = summary["audit"]
    lines = [
        "# Phase 2 — internal model validation", "",
        "Experimental evidence only. No Phase 1/backend models were replaced. This is not a drinking-water certification.", "",
        "## Data and protocol", "",
        f'- Source SHA-256: `{summary["metadata"]["dataset_sha256"]}`.',
        f'- Raw rows: {audit["raw_rows"]}; labelled: {audit["labelled_rows"]}; blank: {audit["blank_rows"]}; nonblank unlabelled: {audit["missing_target_nonblank"]}.',
        "- All four model candidates are compared only on training folds. Median imputation/scaling is fit separately within each fold.",
        "- Winner and dummy are evaluated on a 20% internal holdout. The dataset was explored in Phase 1; this is not a pristine external test.",
        "- Policies and target cohorts are reported separately, never chosen using holdout scores.",
        "- Exact duplicate/conflict checks do not rule out near-duplicate augmentation or common source samples. No source IDs are supplied by organisers.",
        "- Classification: positive class is safe (1). FPR means actual unsafe water predicted safe.",
        "- Safety-oriented threshold stays >=0.5 and is selected from training OOF probabilities for empirical FPR <=2%. Holdout FPR may exceed this cap.",
        "- Probabilities are not claimed calibrated; Brier scores and Wilson FPR intervals are reported.",
        "- OOF candidate selection and threshold selection are not nested. Repeated/block splits reuse dataset rows and primary-selected configuration; they are sensitivity checks, not independent validation.",
        "- Units remain undocumented. Negative concentrations become missing only under the explicit sanitized policy. Positive spikes and pH=0.2 are retained.",
        "", "## Data findings", "",
        f'- {audit["physical_findings"]["invalid_ph_observed"]} observed pH values fall outside operational [0,14]; {audit["physical_findings"]["missing_ph"]} pH targets are missing.',
        f'- {audit["physical_findings"]["paired_negative_hardness_and_fluoride"]} rows contain both negative hardness and negative fluoride.',
    ]
    for name, value in audit["tds_turbidity_correlations"].items():
        lines.append(f'- TDS/Turbidity Pearson correlation, {name}: {value:.5f}.' if value is not None else f'- {name}: correlation undefined.')
    lines += ["", "## Results by predeclared policy and cohort", ""]
    for e in summary["experiments"]:
        lines += [f'### {e["id"]}', "", f'- CV-selected model: **{e["model"]}**; CV {e["cv_metric"]}: {e["cv"]["mean"]:.5f} ± {e["cv"]["std"]:.5f} (fold SD).']
        if e["task"] == "classification":
            for label, scores in e["holdout"].items():
                lo, hi = scores["unsafe_fpr_wilson_95"]
                lines.append(f'- {label}: threshold {scores["threshold"]:.5f}; F1 {scores["f1"]:.5f}; AUC {scores["roc_auc"]:.5f}; unsafe→safe FPR {scores["unsafe_to_safe_fpr"]:.2%} (Wilson {lo:.2%}–{hi:.2%}); safe precision {scores["safe_precision"]:.2%}; safe recall {scores["safe_recall"]:.2%}; safe-call coverage {scores["safe_prediction_coverage"]:.2%}.')
            for name, stress in e["stress"].items():
                if "selected_metrics" in stress:
                    s = stress["selected_metrics"]["conservative_threshold"]
                    lines.append(f'- Stress {name}, safety operating point: F1 {s["f1"]:.5f}, FPR {s["unsafe_to_safe_fpr"]:.2%}.')
        else:
            s, d = e["holdout"], e["dummy"]
            lines += [f'- Internal holdout: RMSE {s["rmse"]:.5f}, MAE {s["mae"]:.5f}, R² {s["r2"]:.5f}.',
                      f'- Dummy RMSE {d["rmse"]:.5f}; relative RMSE reduction {e["rmse_skill_vs_dummy"]:.2%}. A small gain is not strong predictability.']
            for name, stress in e["stress"].items():
                if "selected_metrics" in stress:
                    lines.append(f'- Stress {name}: RMSE {stress["selected_metrics"]["rmse"]:.5f}, skill vs dummy {stress["rmse_skill_vs_dummy"]:.2%}.')
        lines.append("")
    lines += ["## Interpretation and promotion gate", "",
              "- High classification scores describe reproduction of supplied labels, not microbiological safety or BIS compliance.",
              "- FPR reduction can sacrifice almost all safe recall in quick mode. Low FPR alone must not be sold as a useful or trustworthy safe-water detector; inspect safe precision and coverage too.",
              "- A lower pH RMSE after removing invalid targets reflects a different evaluation cohort. Keep raw-target scores alongside it.",
              "- Do not use Potability to predict pH, fabricate improved targets, or hide bad predictions by clipping test labels.",
              "- Phase 3 should explicitly choose and validate deployment policy, handle uncertain/out-of-distribution inputs, and integrate thresholds/feature names with backend before promotion.",
              "- Existing backend loads Phase 1 models and has NOT been changed by this evaluation.", "", "## Reproducibility", "",
              "See runtime_config.json for exact versions and arguments, split_indices.json for original source row indices (CSV line=index+2), comparison.csv for all CV candidates, and results.json for full evidence.",
              "Candidate joblib files require this project's ml.quality module. Only load files you generated/trust; joblib can execute code.", "", "## Figures", ""]
    lines.extend(f'![{name.replace("_", " ").replace(".png", "")}](plots/{name})\n' for name in figures)
    (directory / "REVIEW.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("reports/phase2_validated"))
    args = parser.parse_args()
    generate_report(args.results_dir)
    print(f"Generated summary.json, data_audit.json, REVIEW.md and six plots in {args.results_dir}")


if __name__ == "__main__":
    main()
