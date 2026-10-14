"""Run the isolated 24-hour any-component Azure PdM benchmark experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .evaluation import (
    best_f1_threshold,
    event_metrics,
    machine_bootstrap_ap_ci,
    row_metrics,
    validation_alert_budget_curve,
)
from .pipeline import (
    SENSORS,
    audit_summary,
    build_features,
    build_targets,
    chronological_masks,
    load_tables,
    machine_group_masks,
)


ROOT = Path(__file__).resolve().parent
EXPECTED_DATASET_SHA256 = "7575444d4d605b1f7155ab937cf541243afe07fce1afba998bf4195e8e84e1eb"
ACTIVE_STATUS: Path | None = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_columns(frame: pd.DataFrame, variant: str) -> list[str]:
    """Return a predeclared feature set; identifiers and labels are never predictors."""
    excluded = {"machineID", "datetime", "model"}
    candidates = [
        col for col in frame.columns
        if col not in excluded and not col.startswith("y_") and not col.startswith("valid_")
    ]
    if variant == "all_features":
        return candidates
    if variant == "without_error_history":
        return [col for col in candidates if not col.startswith("past_errorID_")]
    if variant == "telemetry_only":
        return [col for col in candidates if col.startswith(tuple(f"{sensor}_" for sensor in SENSORS))]
    if variant == "plus_model_ablation":
        return [*candidates, "model"]
    raise ValueError(f"Unknown feature variant: {variant}")


def make_preprocessor(columns: list[str]) -> ColumnTransformer:
    numeric = [col for col in columns if col != "model"]
    transformers: list[tuple[str, Any, list[str]]] = [
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric),
    ]
    if "model" in columns:
        transformers.append((
            "model",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ["model"],
        ))
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)


def candidate_estimators(config: dict[str, Any]) -> dict[str, Any]:
    settings = config["models"]
    return {
        "logistic": LogisticRegression(
            C=float(settings["logistic_c"]),
            max_iter=int(settings["logistic_max_iter"]),
            solver="liblinear",
            random_state=int(config["seed"]),
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=int(settings["tree_max_depth"]),
            min_samples_leaf=int(settings["tree_min_samples_leaf"]),
            class_weight=None,
            random_state=int(config["seed"]),
        ),
    }


def probability_of_positive(estimator: Any, matrix: np.ndarray) -> np.ndarray:
    classes = estimator.classes_
    if 1 not in classes:
        return np.zeros(len(matrix), dtype=float)
    return estimator.predict_proba(matrix)[:, int(np.flatnonzero(classes == 1)[0])]


def selection_key(candidate: dict[str, Any]) -> tuple[float, str, str]:
    """Select solely by validation AP; deterministic names break exact ties."""
    ap = candidate["validation_average_precision"]
    return (float(ap) if ap is not None else -1.0, candidate["feature_variant"], candidate["model"])


def _metrics_for(labels: np.ndarray, scores: np.ndarray, threshold: float,
                 rows: pd.DataFrame, failures: pd.DataFrame) -> dict[str, Any]:
    result = row_metrics(labels, scores, threshold)
    result["event_metrics"] = event_metrics(rows, failures, 24, "any", scores, threshold)
    return result


def _split_data(rows: pd.DataFrame, config: dict[str, Any], tables: Any) -> dict[str, dict[str, pd.DataFrame]]:
    eligible = rows["valid_24h"].to_numpy(dtype=bool)
    chronological = chronological_masks(rows, {
        "max_horizon_hours": config["split"]["purge_hours"],
        "split": config["split"],
    })
    group, group_ids = machine_group_masks(rows, tables.machines, {
        "seed": config["split"]["machine_group_seed"],
        "max_horizon_hours": config["split"]["purge_hours"],
        "split": config["split"],
    })
    output: dict[str, dict[str, pd.DataFrame]] = {}
    for name, masks in (("chronological_known_machines", chronological),
                        ("future_unseen_machines", group)):
        output[name] = {
            part: rows.loc[mask & eligible].reset_index(drop=True)
            for part, mask in masks.items()
        }
        for part, subset in output[name].items():
            if subset.empty:
                raise ValueError(f"No eligible {part} rows for {name}")
    return output


def run(config_path: Path, dataset_override: str | None = None) -> dict[str, Any]:
    global ACTIVE_STATUS
    config = json.loads(config_path.read_text())
    zip_path = Path(dataset_override or config["dataset_zip"]).expanduser().resolve()
    if not zip_path.is_file():
        raise FileNotFoundError(f"Dataset ZIP not found: {zip_path}")
    dataset_hash = sha256(zip_path)
    expected_hash = config.get("expected_dataset_zip_sha256", EXPECTED_DATASET_SHA256)
    if dataset_hash != expected_hash or dataset_hash != EXPECTED_DATASET_SHA256:
        raise ValueError("Dataset ZIP hash differs from the audited archive; stop and audit before training")

    directories = {key: ROOT / value for key, value in config["directories"].items()}
    run_id = time.strftime("run_primary24h_%Y%m%dT%H%M%S", time.gmtime()) + f"_{time.time_ns() % 1_000_000_000:09d}Z"
    run_dirs = {key: (path / run_id) for key, path in directories.items()}
    for path in run_dirs.values():
        path.mkdir(parents=True, exist_ok=False)
    ACTIVE_STATUS = run_dirs["metrics"] / "run_status.json"
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    ACTIVE_STATUS.write_text(json.dumps({"status": "running", "run_id": run_id,
                                         "started_utc": started_utc}, indent=2) + "\n")

    tables = load_tables(zip_path)
    audit = audit_summary(tables)
    features = build_features(tables, include_model=True)
    targets = build_targets(features, tables.failures)
    rows = pd.concat([
        features.reset_index(drop=True),
        targets.drop(columns=["machineID", "datetime"]).reset_index(drop=True),
    ], axis=1)
    split_frames = _split_data(rows, config, tables)

    result: dict[str, Any] = {
        "experiment": config["experiment_name"],
        "experiment_version": config["experiment_version"],
        "run_id": run_id,
        "dataset_zip_sha256": dataset_hash,
        "config_sha256": sha256(config_path),
        "config_path": str(config_path),
        "seed": int(config["seed"]),
        "target": config["task"],
        "threshold_policy": config["models"]["threshold_selection"],
        "candidate_selection": config["models"]["candidate_selection"],
        "false_alarm_budget_per_machine_week": config["false_alarm_budget_per_machine_week"],
        "minimum_event_recall": config["minimum_event_recall"],
        "minimum_actionable_lead_time_hours": config["minimum_actionable_lead_time_hours"],
        "probability_status": "uncalibrated raw scores; no probability claim",
        "deployment_acceptance": "undecided: operational gates remain unset",
        "data_audit": audit,
        "split_results": {},
        "elapsed_seconds": None,
    }
    started = time.time()
    feature_variants = config["feature_variants"]

    for split_name, parts in split_frames.items():
        train, validation, test = (parts[key] for key in ("train", "validation", "test"))
        y_train = train["y_24h_any"].to_numpy(dtype=np.uint8)
        y_val = validation["y_24h_any"].to_numpy(dtype=np.uint8)
        candidate_records: list[dict[str, Any]] = []
        fitted: dict[tuple[str, str], tuple[Any, Any, list[str], np.ndarray, np.ndarray]] = {}

        for variant in feature_variants:
            columns = feature_columns(train, variant)
            preprocessor = make_preprocessor(columns)
            x_train = preprocessor.fit_transform(train[columns]).astype(np.float32, copy=False)
            x_val = preprocessor.transform(validation[columns]).astype(np.float32, copy=False)
            for model_name, estimator in candidate_estimators(config).items():
                estimator.fit(x_train, y_train)
                val_scores = probability_of_positive(estimator, x_val)
                val_ap = float(row_metrics(y_val, val_scores, 0.5)["average_precision"])
                threshold = best_f1_threshold(y_val, val_scores)
                candidate_records.append({
                    "feature_variant": variant,
                    "model": model_name,
                    "feature_columns": columns,
                    "validation_average_precision": val_ap,
                    "validation_threshold_max_f1": threshold,
                    "validation_operating_metrics": row_metrics(y_val, val_scores, threshold),
                })
                fitted[(variant, model_name)] = (preprocessor, estimator, columns, x_val, val_scores)

        selected = max(candidate_records, key=selection_key)
        selected_key = (selected["feature_variant"], selected["model"])
        preprocessor, estimator, columns, x_val, val_val_scores = fitted[selected_key]
        threshold = float(selected["validation_threshold_max_f1"])

        # Locked-test labels/scores are evaluated only after the validation-only selection is fixed.
        y_test = test["y_24h_any"].to_numpy(dtype=np.uint8)
        x_test = preprocessor.transform(test[columns]).astype(np.float32, copy=False)
        test_scores = probability_of_positive(estimator, x_test)
        test_metrics = _metrics_for(y_test, test_scores, threshold, test, tables.failures)
        test_ci = machine_bootstrap_ap_ci(
            test[["machineID", "datetime"]], y_test, test_scores,
            int(config["seed"]) + (1 if split_name == "chronological_known_machines" else 2),
            int(config["machine_bootstrap_replicates"]),
        )

        train_prior = float(y_train.mean())
        val_prior_scores = np.full(len(validation), train_prior, dtype=float)
        test_prior_scores = np.full(len(test), train_prior, dtype=float)
        val_prior = _metrics_for(y_val, val_prior_scores, 0.5, validation, tables.failures)
        test_prior = _metrics_for(y_test, test_prior_scores, 0.5, test, tables.failures)

        err_columns = [c for c in train.columns if c.startswith("past_errorID_") and c.endswith("_24h")]
        val_rule_scores = (validation[err_columns].sum(axis=1).to_numpy() > 0).astype(float)
        test_rule_scores = (test[err_columns].sum(axis=1).to_numpy() > 0).astype(float)
        val_error_rule = _metrics_for(y_val, val_rule_scores, 0.5, validation, tables.failures)
        test_error_rule = _metrics_for(y_test, test_rule_scores, 0.5, test, tables.failures)

        artifact_dir = run_dirs["artifacts"] / split_name
        artifact_dir.mkdir(parents=True, exist_ok=False)
        artifact_path = artifact_dir / "selected_any_failure_24h.joblib"
        joblib.dump({
            "preprocessor": preprocessor,
            "estimator": estimator,
            "feature_columns": columns,
            "target": config["task"],
            "model_version": f"azure-pdm-{run_id}-{split_name}-{selected['feature_variant']}-{selected['model']}",
            "dataset_zip_sha256": dataset_hash,
            "config_sha256": result["config_sha256"],
            "threshold_selected_on_validation": threshold,
            "threshold_policy": config["models"]["threshold_selection"],
            "probabilities_calibrated": False,
            "production_eligible": False,
        }, artifact_path, compress=3)

        result["split_results"][split_name] = {
            "eligible_rows": {part: int(len(frame)) for part, frame in parts.items()},
            "positive_rows": {part: int(frame["y_24h_any"].sum()) for part, frame in parts.items()},
            "unique_machine_ids": {part: int(frame.machineID.nunique()) for part, frame in parts.items()},
            "candidate_validation_comparison": [
                {k: v for k, v in rec.items() if k not in ("feature_columns",)}
                for rec in candidate_records
            ],
            "selected_by_validation": {
                "feature_variant": selected["feature_variant"],
                "model": selected["model"],
                "validation_average_precision": selected["validation_average_precision"],
                "threshold": threshold,
                "threshold_selection": config["models"]["threshold_selection"],
            },
            "validation_baselines": {
                "constant_prior": val_prior,
                "any_error_in_prior_24h": val_error_rule,
            },
            "locked_test": {
                "selected_candidate": test_metrics,
                "selected_candidate_machine_cluster_ap_ci_95": test_ci,
                "constant_prior": test_prior,
                "any_error_in_prior_24h": test_error_rule,
            },
            "machine_ids": {
                part: sorted(map(int, frame.machineID.unique())) for part, frame in parts.items()
            },
            "artifact": str(artifact_path.relative_to(ROOT)),
            "feature_columns": columns,
        }

    result["elapsed_seconds"] = round(time.time() - started, 3)
    result["split_metadata"] = {
        "chronological_cutoffs": config["split"],
        "purge_hours": int(config["split"]["purge_hours"]),
        "machine_group_split": "seeded 60/20/20 by machine ID, stratified on static model category; then chronological blocks",
        "feature_cutoff": "strictly before t",
        "label_interval": "(t, t+24h]",
        "preprocessing_fit": "training partition only",
        "locked_test_policy": "candidate and threshold selected from validation before test scoring",
    }
    manifest_path = run_dirs["artifacts"] / "artifact_manifest.json"
    manifest_artifacts = {}
    for split_name, info in result["split_results"].items():
        artifact_path = ROOT / info["artifact"]
        manifest_artifacts[split_name] = {
            "path": info["artifact"],
            "sha256": sha256(artifact_path),
            "model": info["selected_by_validation"]["model"],
            "feature_variant": info["selected_by_validation"]["feature_variant"],
            "feature_columns": info["feature_columns"],
            "threshold_selected_on_validation": info["selected_by_validation"]["threshold"],
            "probabilities_calibrated": False,
            "production_eligible": False,
        }
    manifest_path.write_text(json.dumps({
        "run_id": run_id,
        "dataset_zip_sha256": dataset_hash,
        "config_sha256": result["config_sha256"],
        "target": config["task"],
        "artifacts": manifest_artifacts,
    }, indent=2) + "\n")
    result["artifact_manifest"] = str(manifest_path.relative_to(ROOT))
    metrics_path = run_dirs["metrics"] / "evaluation.json"
    metrics_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (run_dirs["reports"] / "config_snapshot.json").write_text(json.dumps(config, indent=2) + "\n")
    (run_dirs["reports"] / "data_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    report_path = run_dirs["reports"] / "experiment_report.md"
    write_report(report_path, result, config, metrics_path)
    ACTIVE_STATUS.write_text(json.dumps({
        "status": "complete", "run_id": run_id, "started_utc": started_utc,
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2) + "\n")
    result["output_paths"] = {
        "metrics": str(metrics_path.relative_to(ROOT)),
        "report": str(report_path.relative_to(ROOT)),
        "artifacts": str(run_dirs["artifacts"].relative_to(ROOT)),
    }
    # Keep the machine-readable file self-contained with its output path references.
    metrics_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    ACTIVE_STATUS = None
    return result


def write_report(path: Path, result: dict[str, Any], config: dict[str, Any], metrics_path: Path) -> None:
    lines = [
        "# Azure PdM 24-hour Any-Component Experiment",
        "",
        "This is an isolated offline experiment on a synthetic benchmark. The target is a failure-table event logged in `(t, t+24h]`, not verified physical failure onset or a company-machine prediction. Scores are uncalibrated. No artifact is connected to production.",
        "",
        f"- Run: `{result['run_id']}`",
        f"- Dataset SHA-256: `{result['dataset_zip_sha256']}`",
        f"- Config SHA-256: `{result['config_sha256']}`",
        f"- Seed: `{result['seed']}`",
        f"- Split policy: {result['split_metadata']['machine_group_split']}",
        f"- Temporal purge: {result['split_metadata']['purge_hours']} hours; this does not account for unknown event logging delay.",
        "- Data provenance: consistent with the Microsoft Azure PdM synthetic benchmark. The supplied archive has no dataset-specific license; rights are unverified.",
        "- Acceptance: undecided; alert budget, minimum event recall, and minimum actionable lead time are unset.",
        "",
        "## Validation-only candidate selection",
        "",
        "Candidate selection uses validation AP only; candidate thresholds maximize validation F1 with ties resolved toward fewer alerts. The fixed error rule and prior are baselines, not tuned candidates.",
        "",
        "| Split | Candidate | Feature variant | Validation AP | Validation threshold | Validation precision | Validation recall |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for split_name, info in result["split_results"].items():
        for candidate in info["candidate_validation_comparison"]:
            validation_metrics = candidate["validation_operating_metrics"]
            lines.append(
                f"| {split_name} | {candidate['model']} | {candidate['feature_variant']} | "
                f"{candidate['validation_average_precision']:.6f} | "
                f"{candidate['validation_threshold_max_f1']:.8g} | "
                f"{validation_metrics['precision']:.4f} | {validation_metrics['recall']:.4f} |"
            )
        for baseline_name, baseline in info["validation_baselines"].items():
            lines.append(
                f"| {split_name} | {baseline_name} | fixed baseline | "
                f"{baseline['average_precision']:.6f} | 0.5 (fixed) | "
                f"{baseline['precision']:.4f} | {baseline['recall']:.4f} |"
            )
    lines.extend([
        "",
        "## Selected candidate and locked-test report",
        "",
        "Locked-test measurements below are reported only after candidate and threshold selection was fixed from validation. Constant-prior and error-rule baselines use predeclared scores and a fixed 0.5 threshold.",
        "",
        "| Split | Partition | Candidate / baseline | Feature variant | AP | Precision | Recall | FP rows | FN rows | Event recall | False alarms / machine-week | Missed events | Events | Lead median (P10–P90) h | Warned >=24h / all events |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for split_name, info in result["split_results"].items():
        selected = info["selected_by_validation"]
        rows_to_report = [
            ("validation", f"{selected['model']} / selected on validation", selected["feature_variant"],
             next(x["validation_operating_metrics"] for x in info["candidate_validation_comparison"]
                  if x["model"] == selected["model"] and x["feature_variant"] == selected["feature_variant"])),
            ("locked test", f"{selected['model']} / validation-selected", selected["feature_variant"],
             info["locked_test"]["selected_candidate"]),
            ("validation", "constant prior", "train prevalence", info["validation_baselines"]["constant_prior"]),
            ("locked test", "constant prior", "train prevalence", info["locked_test"]["constant_prior"]),
            ("validation", "any error in prior 24h", "fixed rule", info["validation_baselines"]["any_error_in_prior_24h"]),
            ("locked test", "any error in prior 24h", "fixed rule", info["locked_test"]["any_error_in_prior_24h"]),
        ]
        for partition, label, variant, metric in rows_to_report:
            ev = metric.get("event_metrics", {})
            cov = ev.get("warning_coverage", {})
            lead_summary = ev.get("lead_time_hours", {})
            lead = lead_summary.get("median")
            p10 = lead_summary.get("p10")
            p90 = lead_summary.get("p90")
            lead_text = "n/a" if lead is None else f"{lead} ({p10}–{p90})"
            fraction = cov.get("fraction_of_all_events_warned_at_least_24h")
            event_recall = ev.get("event_recall")
            false_alarm_rate = ev.get("false_alarms_per_machine_week")
            missed_events = ev.get("missed_events")
            event_count = ev.get("event_count")
            warned_24h = cov.get("events_warned_at_least_24h", 0)
            coverage_text = "n/a" if fraction is None else f"{warned_24h} / {fraction}"
            lines.append(
                f"| {split_name} | {partition} | {label} | {variant} | {metric.get('average_precision')} | "
                f"{metric.get('precision')} | {metric.get('recall')} | {metric.get('false_positives')} | "
                f"{metric.get('false_negatives')} | {event_recall if event_recall is not None else 'n/a'} | "
                f"{false_alarm_rate if false_alarm_rate is not None else 'n/a'} | "
                f"{missed_events if missed_events is not None else 'n/a'} | "
                f"{event_count if event_count is not None else 'n/a'} | "
                f"{lead_text} | {coverage_text} |"
            )
        ci = info["locked_test"]["selected_candidate_machine_cluster_ap_ci_95"]
        lines.extend([
            "",
            f"**{split_name} validation-selected candidate:** `{selected['model']}` with `{selected['feature_variant']}`; validation AP `{selected['validation_average_precision']:.6f}`, validation-selected threshold `{selected['threshold']:.8g}`. Test AP machine-cluster 95% interval: `{ci['lower_95']}` to `{ci['upper_95']}` ({ci['valid_replicates']}/{ci['replicates']} valid resamples).",
            f"Eligible rows: {info['eligible_rows']}; positive row counts: {info['positive_rows']}; machine counts: {info['unique_machine_ids']}.",
            "",
        ])
    lines.extend([
        "",
        "## Comparability with the saved multi-output experiment",
        "",
        "The earlier general runner trained four component classifiers and formed the any-component score as the maximum of component scores. This Prompt 2 run fits the specified direct binary `y_24h_any` classifier. The target label concept is the same, but the learning objective and score construction differ, so candidate AP values should not be treated as a like-for-like reproduction. The saved earlier chronological test metric records 98 any-component event instances, 94 detected, and 4 missed; the Prompt 1 readiness prose that called 95 the total event count conflated detected count with the event denominator. This run records 98 events, 95 detected, and 3 missed for its own selected direct-binary candidate.",
        "## Interpretation and limitations",
        "",
        "Precision/recall are row-level; event recall, missed events, false alarms per machine-week, and lead times use deduplicated alert windows. The lead-time timestamps are rounded logged-event times; warning coverage is not a guarantee of actionable notice. There is no explicit operating-state exposure, and this synthetic benchmark cannot establish real-world performance. Calibration was not performed. No operational alert budget or minimum recall was approved, so no deployment acceptance is made.",
        "",
        "The error-history ablations show strong reliance on benchmark error codes: AP falls materially when error features are removed, while the simple any-error rule has much lower AP and substantially more false alarms despite high event recall. The selected model-category ablation for unseen machines differs from the non-model candidate by only a very small validation AP margin; that does not establish a meaningful benefit from machine model category.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "cd /Users/ashishsachinas/Desktop/SMARTA/SMARTA",
        "../.venv/bin/python -m experiments.azure_pdm.train_primary_24h --config experiments/azure_pdm/config_primary_24h_20261013.json",
        "../.venv/bin/python -m unittest discover -s experiments/azure_pdm/tests -v",
        "```",
        "",
        f"Metrics: `{metrics_path.relative_to(ROOT)}`. Artifact hashes and schemas: `{result['artifact_manifest']}`. Model artifacts remain under the isolated run artifact directory and are not eligible for production loading.",
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config_primary_24h_20261013.json")
    parser.add_argument("--dataset-zip", default=None)
    args = parser.parse_args()
    try:
        result = run(args.config.resolve(), args.dataset_zip)
    except BaseException as exc:
        if ACTIVE_STATUS is not None and ACTIVE_STATUS.exists():
            ACTIVE_STATUS.write_text(json.dumps({
                "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                "error_type": type(exc).__name__, "message": str(exc),
            }, indent=2) + "\n")
        raise
    print(json.dumps({
        "run_id": result["run_id"],
        "dataset_zip_sha256": result["dataset_zip_sha256"],
        "elapsed_seconds": result["elapsed_seconds"],
        "output_paths": result["output_paths"],
        "selected": {name: val["selected_by_validation"] for name, val in result["split_results"].items()},
        "acceptance": result["deployment_acceptance"],
    }, indent=2))


if __name__ == "__main__":
    main()
