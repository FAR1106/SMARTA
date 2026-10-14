"""Reproducible AI4I dataset loading, training, evaluation, and inference."""

from __future__ import annotations

import json
import math
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "ai4i"
MODEL_DIR = ROOT / "models"
DEFAULT_ARCHIVE = Path.home() / "Downloads" / "ai4i+2020+predictive+maintenance+dataset.zip"
FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
NUMERIC_FEATURES = FEATURES[1:]
IDENTIFIERS = ["UDI", "Product ID"]
FAILURE_TARGET = "Machine failure"
MODE_TARGETS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
MIN_MODE_POSITIVES = 20
MODEL_VERSION = "ai4i-rf-v1"


class DatasetError(ValueError):
    pass


class ModelUnavailable(RuntimeError):
    pass


def inspect_archive(archive: Path = DEFAULT_ARCHIVE, destination: Path = DATA_DIR) -> Path:
    """Safely locate and extract the single CSV dataset from a supplied ZIP."""
    archive = Path(archive).expanduser().resolve()
    if not archive.is_file():
        raise DatasetError(f"AI4I ZIP not found: {archive}")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        candidates = [info for info in zf.infolist() if not info.is_dir() and info.filename.lower().endswith(".csv")]
        if len(candidates) != 1:
            raise DatasetError(f"Expected exactly one CSV in archive; found {len(candidates)}.")
        info = candidates[0]
        member = PurePosixPath(info.filename)
        if member.is_absolute() or ".." in member.parts:
            raise DatasetError("Unsafe path in dataset archive.")
        target = destination / member.name
        with zf.open(info) as source, target.open("wb") as output:
            shutil.copyfileobj(source, output)
    return target


def load_dataset(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"Dataset file not found: {path}")
    frame = pd.read_csv(path)
    required = set(FEATURES + [FAILURE_TARGET] + MODE_TARGETS)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DatasetError(f"Dataset schema is missing required fields: {', '.join(missing)}")
    if frame.empty:
        raise DatasetError("Dataset contains no records.")
    if frame[FEATURES + [FAILURE_TARGET] + MODE_TARGETS].isna().any().any():
        raise DatasetError("Required dataset fields contain missing values.")
    if not set(frame["Type"].astype(str).unique()).issubset({"L", "M", "H"}):
        raise DatasetError("Unexpected product Type values.")
    for target in [FAILURE_TARGET, *MODE_TARGETS]:
        values = set(pd.to_numeric(frame[target], errors="coerce").dropna().unique())
        if not values.issubset({0, 1}) or len(values) < 2:
            raise DatasetError(f"Target {target} must be binary and contain both classes.")
    for column in NUMERIC_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column].to_numpy()).all():
            raise DatasetError(f"Feature {column} contains a non-finite value.")
    return frame


def dataset_report(frame: pd.DataFrame) -> dict[str, Any]:
    mode_union = frame[MODE_TARGETS].astype(int).max(axis=1)
    machine_failure = frame[FAILURE_TARGET].astype(int)
    mismatch = mode_union != machine_failure
    return {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "missing_values": {str(k): int(v) for k, v in frame.isna().sum().items()},
        "duplicate_rows": int(frame.duplicated().sum()),
        "feature_types": {name: str(frame[name].dtype) for name in FEATURES},
        "class_distribution": {
            target: {str(int(k)): int(v) for k, v in frame[target].value_counts().sort_index().items()}
            for target in [FAILURE_TARGET, *MODE_TARGETS]
        },
        "target_consistency": {
            "machine_failure_mode_union_mismatch_rows": int(mismatch.sum()),
            "machine_failure_without_mode_label_rows": int(((machine_failure == 1) & (mode_union == 0)).sum()),
            "mode_label_without_machine_failure_rows": int(((mode_union == 1) & (machine_failure == 0)).sum()),
            "multiple_mode_label_rows": int((frame[MODE_TARGETS].sum(axis=1) > 1).sum()),
        },
    }


def _pipeline(seed: int, *, dummy: bool = False) -> Pipeline:
    transform = ColumnTransformer(
        [("type", OneHotEncoder(handle_unknown="ignore"), ["Type"])],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    classifier = DummyClassifier(strategy="prior") if dummy else RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", transform), ("classifier", classifier)])


def _metrics(estimator: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    y_pred = estimator.predict(x_test)
    probs = estimator.predict_proba(x_test)[:, 1]
    calibration_bins = []
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        mask = (probs >= lower) & (probs < lower + 0.1)
        if lower > 0.8:
            mask |= probs == 1.0
        if mask.any():
            mean_score = float(probs[mask].mean())
            observed_rate = float(np.asarray(y_test)[mask].mean())
            ece += float(mask.mean()) * abs(mean_score - observed_rate)
            calibration_bins.append({"count": int(mask.sum()), "mean_score": mean_score, "observed_positive_rate": observed_rate})
    return {
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist(),
        "pr_auc": float(average_precision_score(y_test, probs)),
        "roc_auc": float(roc_auc_score(y_test, probs)) if len(set(y_test)) == 2 else None,
        "brier_score": float(brier_score_loss(y_test, probs)),
        "expected_calibration_error_10_bins": float(ece),
        "calibration_bins": calibration_bins,
        "classification_report": classification_report(y_test, y_pred, labels=[0, 1], output_dict=True, zero_division=0),
    }


def train_model(dataset_path: Path, model_dir: Path = MODEL_DIR, seed: int = 42) -> dict[str, Any]:
    """Train held-out evaluated classifiers. Mode labels are modeled independently."""
    frame = load_dataset(dataset_path)
    x = frame[FEATURES].copy()
    y = frame[FAILURE_TARGET].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=seed, stratify=y
    )
    model = _pipeline(seed).fit(x_train, y_train)
    baseline = _pipeline(seed, dummy=True).fit(x_train, y_train)
    evaluation: dict[str, Any] = {
        "dataset": dataset_report(frame),
        "target": FAILURE_TARGET,
        "features": FEATURES,
        "excluded_identifiers": IDENTIFIERS,
        "excluded_targets": [FAILURE_TARGET, *MODE_TARGETS],
        "split": {"method": "stratified 80/20 train/test", "seed": seed, "train_rows": len(x_train), "test_rows": len(x_test)},
        "machine_failure": {
            "random_forest": _metrics(model, x_test, y_test),
            "prior_baseline": _metrics(baseline, x_test, y_test),
            "test_class_distribution": {str(int(k)): int(v) for k, v in y_test.value_counts().sort_index().items()},
        },
        "failure_modes": {},
        "limitations": [
            "AI4I 2020 is synthetic; held-out performance measures separation within this dataset distribution only.",
            "This classifier categorizes the supplied operating point and does not forecast failure dates or lead time.",
            "Random split evaluation does not establish performance on real equipment or new production environments.",
            "Random-forest scores are not calibrated probabilities; they are reported as uncalibrated model scores.",
        ],
    }
    modes: dict[str, Pipeline] = {}
    unsupported_modes: dict[str, str] = {}
    for target in MODE_TARGETS:
        labels = frame[target].astype(int)
        positives = int(labels.sum())
        if positives < MIN_MODE_POSITIVES:
            reason = f"Only {positives} positive examples; minimum is {MIN_MODE_POSITIVES}."
            unsupported_modes[target] = reason
            evaluation["failure_modes"][target] = {
                "supported": False,
                "reason": reason,
                "positive_examples": positives,
            }
            continue
        xm_train, xm_test, ym_train, ym_test = train_test_split(
            x, labels, test_size=0.20, random_state=seed, stratify=labels
        )
        mode_model = _pipeline(seed).fit(xm_train, ym_train)
        modes[target] = mode_model
        evaluation["failure_modes"][target] = {
            "supported": True,
            "positive_examples": positives,
            "test_positive_examples": int(ym_test.sum()),
            "metrics": _metrics(mode_model, xm_test, ym_test),
        }

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "features": FEATURES,
        "feature_ranges": {
            name: [float(frame[name].min()), float(frame[name].max())] for name in NUMERIC_FEATURES
        },
        "allowed_types": sorted(frame["Type"].astype(str).unique().tolist()),
        "machine_failure": model,
        "failure_modes": modes,
        "supported_modes": sorted(modes),
        "unsupported_modes": unsupported_modes,
        "evaluation": evaluation,
    }
    joblib.dump(bundle, model_dir / "ai4i_model.joblib")
    (model_dir / "ai4i_metrics.json").write_text(json.dumps(evaluation, indent=2, allow_nan=False) + "\n")
    return evaluation


class Predictor:
    def __init__(self, artifact_path: Path | None = None):
        self.artifact_path = Path(artifact_path or (MODEL_DIR / "ai4i_model.joblib")).resolve()
        self._bundle: dict[str, Any] | None = None

    def load(self) -> dict[str, Any]:
        if self._bundle is None:
            if not self.artifact_path.is_file():
                raise ModelUnavailable("The predictive model is not trained. Run predictive/train_ai4i.py first.")
            self._bundle = joblib.load(self.artifact_path)
        return self._bundle

    def metadata(self) -> dict[str, Any]:
        bundle = self.load()
        return {
            "ready": True,
            "model_version": bundle["version"],
            "trained_at": bundle.get("trained_at"),
            "features": bundle["features"],
            "feature_ranges": bundle["feature_ranges"],
            "allowed_types": bundle["allowed_types"],
            "supported_failure_modes": bundle["supported_modes"],
            "unsupported_failure_modes": bundle.get("unsupported_modes", {}),
        }

    def predict(self, features: dict[str, Any], machine_id: str | None = None) -> dict[str, Any]:
        bundle = self.load()
        missing = [name for name in FEATURES if name not in features]
        extra = sorted(set(features) - set(FEATURES))
        if missing or extra:
            raise ValueError(f"Feature fields invalid; missing={missing}, unexpected={extra}.")
        if str(features["Type"]) not in bundle["allowed_types"]:
            raise ValueError("Type must be one of L, M, or H.")
        record = {name: features[name] for name in FEATURES}
        out_of_training_range: dict[str, dict[str, Any]] = {}
        for name in NUMERIC_FEATURES:
            try:
                value = float(record[name])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be numeric.") from exc
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            if name in ("Air temperature [K]", "Process temperature [K]") and value <= 0:
                raise ValueError(f"{name} must be above absolute zero (0 K).")
            if name == "Tool wear [min]" and value < 0:
                raise ValueError("Tool wear [min] cannot be negative.")
            low, high = bundle["feature_ranges"][name]
            if value < low or value > high:
                out_of_training_range[name] = {
                    "value": value,
                    "training_range": [low, high],
                }
            record[name] = value
        x = pd.DataFrame([record], columns=FEATURES)
        machine_score = float(bundle["machine_failure"].predict_proba(x)[0, 1])
        mode_results = {}
        for mode, estimator in bundle["failure_modes"].items():
            mode_results[mode] = {
                "score": float(estimator.predict_proba(x)[0, 1]),
                "score_type": "uncalibrated_random_forest_score",
                "classification": int(estimator.predict(x)[0]),
            }
        return {
            "machine_id": machine_id,
            "prediction_timestamp": datetime.now(timezone.utc).isoformat(),
            "model_version": bundle["version"],
            "machine_failure": {
                "classification": int(bundle["machine_failure"].predict(x)[0]),
                "score": machine_score,
                "score_type": "uncalibrated_random_forest_score",
            },
            "failure_modes": mode_results,
            "unsupported_failure_modes": bundle.get("unsupported_modes", {}),
            "input_features": record,
            "out_of_training_range": out_of_training_range,
            "limitations": "AI4I synthetic-data classification only; not a future failure date or a real-machine forecast.",
        }


def train_from_default_archive() -> dict[str, Any]:
    csv_path = inspect_archive(DEFAULT_ARCHIVE)
    return train_model(csv_path)
