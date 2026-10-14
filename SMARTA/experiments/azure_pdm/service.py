"""Strict, offline-only interface for saved Azure PdM 24-hour candidates.

This module is intentionally not imported or mounted by ``api.py``. Callers
must supply the exact engineered feature vector produced by the audited
historical feature pipeline; this service does not turn a single sensor row
into a prediction.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parent
DEFAULT_RUN_ID = "run_primary24h_20261013T171456_789146000Z"
ALLOWED_SPLITS = {"chronological_known_machines", "future_unseen_machines"}
EXPECTED_DATASET_SHA256 = "7575444d4d605b1f7155ab937cf541243afe07fce1afba998bf4195e8e84e1eb"
EXPECTED_CONFIG_SHA256 = "f28d7436becf6f7386dd7a129c80a04b1914137569f865e1b890d80ae2864d7c"
EXPECTED_ARTIFACT_SHA256 = {
    "chronological_known_machines": "560b9c60ed7367e941f38c60ebcb4f28237ca8e5450b39f896c4d89e3081d24c",
    "future_unseen_machines": "209d4e82f1b07e0e5f162b3507d5da2e83c15c1802be33611567b0637435c945",
}
EXPECTED_FEATURE_PIPELINE = "experiments.azure_pdm.pipeline.build_features"
TARGET_DESCRIPTION = "A failure-table event logged in (t, t+24h]; not confirmed physical failure onset."


class ExperimentalArtifactError(RuntimeError):
    """Raised when a saved experimental artifact cannot be trusted or loaded."""


class ExperimentalInputError(ValueError):
    """Raised when a request does not satisfy the engineered-feature contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _timestamp(value: Any, field: str) -> pd.Timestamp:
    if not isinstance(value, (str, pd.Timestamp)) and not hasattr(value, "isoformat"):
        raise ExperimentalInputError(f"{field} must be an ISO-8601 timestamp or datetime")
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ExperimentalInputError(f"{field} is not a valid timestamp") from exc
    if pd.isna(result):
        raise ExperimentalInputError(f"{field} is not a valid timestamp")
    return result


def _same_timezone_awareness(left: pd.Timestamp, right: pd.Timestamp) -> bool:
    return (left.tzinfo is None) == (right.tzinfo is None)


class AzurePdmExperimentalService:
    """Load and score one exact saved Prompt 2 candidate.

    ``evaluation_split`` identifies which validation-selected artifact to use.
    The default is the chronological-known-machines candidate. The optional
    unseen-machines artifact is a separate model and has its own schema and
    threshold; it is not an ensemble or a production recommendation.
    """

    def __init__(self, run_id: str = DEFAULT_RUN_ID,
                 evaluation_split: str = "chronological_known_machines",
                 root: Path | None = None):
        if evaluation_split not in ALLOWED_SPLITS:
            raise ExperimentalArtifactError(f"Unsupported evaluation split: {evaluation_split}")
        if run_id != DEFAULT_RUN_ID:
            raise ExperimentalArtifactError("Only the audited Prompt 2 primary 24-hour run is supported")
        self.root = Path(root or ROOT).resolve()
        manifest_path = self.root / "artifacts" / run_id / "artifact_manifest.json"
        if not manifest_path.is_file():
            raise ExperimentalArtifactError(f"Artifact manifest not found for run {run_id}")
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ExperimentalArtifactError("Artifact manifest is unreadable or malformed") from exc
        if manifest.get("run_id") != run_id:
            raise ExperimentalArtifactError("Manifest run identifier does not match requested run")
        if manifest.get("dataset_zip_sha256") != EXPECTED_DATASET_SHA256:
            raise ExperimentalArtifactError("Run dataset provenance does not match the audited Azure archive")
        if manifest.get("config_sha256") != EXPECTED_CONFIG_SHA256:
            raise ExperimentalArtifactError("Run configuration does not match the audited Prompt 2 configuration")
        entry = manifest.get("artifacts", {}).get(evaluation_split)
        if not isinstance(entry, dict):
            raise ExperimentalArtifactError(f"No saved candidate for split {evaluation_split}")
        artifact_path = (self.root / entry.get("path", "")).resolve()
        if not artifact_path.is_relative_to(self.root):
            raise ExperimentalArtifactError("Artifact path escapes the Azure experiment directory")
        if not artifact_path.is_file():
            raise ExperimentalArtifactError(f"Saved candidate artifact is missing: {artifact_path}")
        actual_hash = _sha256(artifact_path)
        if actual_hash != entry.get("sha256") or actual_hash != EXPECTED_ARTIFACT_SHA256[evaluation_split]:
            raise ExperimentalArtifactError("Saved candidate artifact hash does not match its manifest")
        try:
            bundle = joblib.load(artifact_path)
        except Exception as exc:
            raise ExperimentalArtifactError("Saved candidate artifact could not be deserialized") from exc

        columns = bundle.get("feature_columns")
        if not isinstance(columns, list) or not columns or columns != entry.get("feature_columns"):
            raise ExperimentalArtifactError("Artifact feature order does not match its manifest")
        preprocessor = bundle.get("preprocessor")
        preprocessor_columns = getattr(preprocessor, "feature_names_in_", None)
        if preprocessor_columns is None or list(preprocessor_columns) != columns:
            raise ExperimentalArtifactError("Saved preprocessing feature order does not match the model contract")
        if bundle.get("dataset_zip_sha256") != manifest["dataset_zip_sha256"]:
            raise ExperimentalArtifactError("Artifact dataset provenance does not match its manifest")
        if bundle.get("config_sha256") != manifest.get("config_sha256"):
            raise ExperimentalArtifactError("Artifact config provenance does not match its manifest")
        threshold = bundle.get("threshold_selected_on_validation")
        if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
                or not math.isfinite(float(threshold)) or not 0.0 <= float(threshold) <= 1.0
                or float(threshold) != float(entry.get("threshold_selected_on_validation"))):
            raise ExperimentalArtifactError("Saved threshold is missing or inconsistent with the manifest")
        target = bundle.get("target")
        if (not isinstance(target, dict) or target.get("target_column") != "y_24h_any"
                or target.get("horizon_hours") != 24
                or target.get("target") != "any_component_logged_failure"):
            raise ExperimentalArtifactError("Saved candidate target is not the approved 24-hour any-component target")
        if target.get("interval") != "(t, t + 24 hours]" or target.get("feature_cutoff") != "strictly_before_t":
            raise ExperimentalArtifactError("Saved candidate target timing does not match the approved specification")
        if bundle.get("probabilities_calibrated") is not False:
            raise ExperimentalArtifactError("Candidate calibration metadata is missing or unexpected")
        if bundle.get("production_eligible") is not False:
            raise ExperimentalArtifactError("Candidate must remain explicitly ineligible for production")

        self.run_id = run_id
        self.evaluation_split = evaluation_split
        self.manifest_path = manifest_path
        self.artifact_path = artifact_path
        self.artifact_sha256 = actual_hash
        self.dataset_sha256 = manifest["dataset_zip_sha256"]
        self.config_sha256 = manifest["config_sha256"]
        self.feature_columns = list(columns)
        self.threshold = float(threshold)
        self.target = dict(target)
        self.model_id = str(bundle.get("model_version"))
        if not self.model_id or self.model_id == "None":
            raise ExperimentalArtifactError("Candidate model identifier is missing")
        self.feature_variant = str(entry.get("feature_variant"))
        self.estimator = bundle.get("estimator")
        self.preprocessor = preprocessor
        if not callable(getattr(self.estimator, "predict_proba", None)):
            raise ExperimentalArtifactError("Saved candidate does not expose predict_proba")
        if not isinstance(self.estimator, DecisionTreeClassifier) or not hasattr(self.estimator, "tree_"):
            raise ExperimentalArtifactError("Saved candidate does not support the audited decision-tree explanation")
        if len(self.estimator.feature_importances_) != len(self.preprocessor.get_feature_names_out()):
            raise ExperimentalArtifactError("Tree and preprocessing output feature dimensions do not match")
        self.allowed_machine_models: set[str] | None = None
        if "model" in self.feature_columns:
            try:
                encoder = self.preprocessor.named_transformers_["model"]
                self.allowed_machine_models = set(map(str, encoder.categories_[0]))
            except (AttributeError, KeyError, IndexError) as exc:
                raise ExperimentalArtifactError("Model-category encoder metadata is missing") from exc
        self.transformed_feature_names = list(map(str, self.preprocessor.get_feature_names_out()))
        self._transformed_feature_contract = self._make_transformed_feature_contract()

    def _prepare(self, request: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
        if not isinstance(request, Mapping):
            raise ExperimentalInputError("Request must be a mapping")
        required = {"features", "prediction_cutoff", "historical_context"}
        missing_top = required - set(request)
        extra_top = set(request) - required
        if missing_top:
            raise ExperimentalInputError(f"Missing request fields: {sorted(missing_top)}")
        if extra_top:
            raise ExperimentalInputError(f"Unsupported request fields: {sorted(extra_top)}")

        features = request["features"]
        if not isinstance(features, Mapping):
            raise ExperimentalInputError("features must be a mapping of exact engineered feature names to values")
        missing = set(self.feature_columns) - set(features)
        extra = set(features) - set(self.feature_columns)
        if missing:
            raise ExperimentalInputError(f"Missing engineered features: {sorted(missing)}")
        if extra:
            raise ExperimentalInputError(f"Unsupported engineered features: {sorted(extra)}")

        cutoff = _timestamp(request["prediction_cutoff"], "prediction_cutoff")
        context = request["historical_context"]
        if not isinstance(context, Mapping):
            raise ExperimentalInputError("historical_context must be a mapping")
        expected_context = {
            "history_start", "latest_telemetry_at", "telemetry_history_complete",
            "event_history_complete_before_cutoff", "feature_pipeline",
        }
        missing_context = expected_context - set(context)
        extra_context = set(context) - expected_context
        if missing_context:
            raise ExperimentalInputError(f"Missing historical-context fields: {sorted(missing_context)}")
        if extra_context:
            raise ExperimentalInputError(f"Unsupported historical-context fields: {sorted(extra_context)}")
        history_start = _timestamp(context["history_start"], "history_start")
        latest_telemetry = _timestamp(context["latest_telemetry_at"], "latest_telemetry_at")
        if not _same_timezone_awareness(cutoff, history_start) or not _same_timezone_awareness(cutoff, latest_telemetry):
            raise ExperimentalInputError("All timestamps must use the same timezone awareness")
        if latest_telemetry >= cutoff:
            raise ExperimentalInputError("latest_telemetry_at must be strictly before prediction_cutoff")
        if latest_telemetry != cutoff - pd.Timedelta(hours=1):
            raise ExperimentalInputError("latest_telemetry_at must be exactly one hour before cutoff for this hourly benchmark")
        if history_start > cutoff - pd.Timedelta(hours=168):
            raise ExperimentalInputError("At least 168 hours of complete prior telemetry history is required")
        if context["telemetry_history_complete"] is not True:
            raise ExperimentalInputError("telemetry_history_complete must be true")
        if context["event_history_complete_before_cutoff"] is not True:
            raise ExperimentalInputError("Historical event streams must be complete strictly before cutoff")
        if context["feature_pipeline"] != EXPECTED_FEATURE_PIPELINE:
            raise ExperimentalInputError(f"feature_pipeline must be {EXPECTED_FEATURE_PIPELINE!r}")

        clean: dict[str, Any] = {}
        for column in self.feature_columns:
            value = features[column]
            if column == "model":
                if not isinstance(value, str) or value not in (self.allowed_machine_models or set()):
                    raise ExperimentalInputError(f"Unsupported model category for feature 'model': {value!r}")
                clean[column] = value
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
                raise ExperimentalInputError(f"Feature {column!r} must be a finite numeric value")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ExperimentalInputError(f"Feature {column!r} must be finite")
            if column.startswith(("past_errorID_", "past_failure_", "past_comp_")):
                if numeric < 0 or not numeric.is_integer():
                    raise ExperimentalInputError(f"Historical count feature {column!r} must be a non-negative integer")
            if "_std_" in column and numeric < 0:
                raise ExperimentalInputError(f"Standard-deviation feature {column!r} cannot be negative")
            clean[column] = numeric

        frame = pd.DataFrame([[clean[name] for name in self.feature_columns]], columns=self.feature_columns)
        return frame, {"prediction_cutoff": cutoff, "history_start": history_start,
                       "latest_telemetry_at": latest_telemetry}

    @staticmethod
    def _feature_definition(name: str) -> str:
        """Describe an engineered feature from pipeline naming and code semantics."""
        import re

        match = re.fullmatch(r"(volt|rotate|pressure|vibration)_(mean|std|min|max)_(3|24|168)h", name)
        if match:
            sensor, statistic, hours = match.groups()
            statistic_name = {"mean": "mean", "std": "population standard deviation",
                              "min": "minimum", "max": "maximum"}[statistic]
            return f"Trailing {hours}-hour {statistic_name} of {sensor}, using samples strictly before t."
        match = re.fullmatch(r"(volt|rotate|pressure|vibration)_delta_3h", name)
        if match:
            return f"Difference between the latest completed {match.group(1)} reading and the reading three hours earlier, both strictly before t."
        match = re.fullmatch(r"past_errorID_(error[1-5])_(24|168)h", name)
        if match:
            code, hours = match.groups()
            return f"Count of logged {code} events in [t-{hours}h, t)."
        match = re.fullmatch(r"past_failure_(comp[1-4])_(24|168)h", name)
        if match:
            component, hours = match.groups()
            return f"Count of prior logged {component} failure events in [t-{hours}h, t)."
        match = re.fullmatch(r"past_comp_(comp[1-4])_(24|168)h", name)
        if match:
            component, hours = match.groups()
            return f"Count of prior {component} maintenance events in [t-{hours}h, t)."
        if name == "model":
            return "Static machine model category from the benchmark machine metadata."
        return "Definition unavailable from the saved feature-engineering pipeline."

    def _make_transformed_feature_contract(self) -> list[dict[str, Any]]:
        numeric_names = [name for name in self.feature_columns if name != "model"]
        contract: list[dict[str, Any]] = []
        for name in numeric_names:
            contract.append({"feature_name": name, "transformed_feature": f"numeric__{name}",
                             "kind": "numeric", "definition": self._feature_definition(name)})
        if "model" in self.feature_columns:
            encoder = self.preprocessor.named_transformers_["model"]
            for category in map(str, encoder.categories_[0]):
                contract.append({"feature_name": "model", "transformed_feature": f"model__model_{category}",
                                 "kind": "one_hot", "category": category,
                                 "definition": self._feature_definition("model")})
        if [item["transformed_feature"] for item in contract] != self.transformed_feature_names:
            raise ExperimentalArtifactError("Transformed feature names do not match the saved preprocessing contract")
        return contract

    def _explain_tree_path(self, frame: pd.DataFrame, transformed: np.ndarray,
                           raw_score: float) -> dict[str, Any]:
        """Describe the exact fitted-tree path; deltas telescope to its leaf score."""
        tree = self.estimator.tree_
        vector = np.asarray(transformed, dtype=float).reshape(-1)
        node = 0
        path: list[dict[str, Any]] = []
        root_values = np.asarray(tree.value[0][0], dtype=float)
        root_proportions = root_values / root_values.sum()
        parent_positive = float(root_proportions[np.flatnonzero(self.estimator.classes_ == 1)[0]])

        while tree.children_left[node] != tree.children_right[node]:
            feature_idx = int(tree.feature[node])
            if feature_idx < 0 or feature_idx >= len(self._transformed_feature_contract):
                raise ExperimentalArtifactError("Decision-tree path references an unknown transformed feature")
            contract = self._transformed_feature_contract[feature_idx]
            transformed_value = float(vector[feature_idx])
            split_threshold = float(tree.threshold[node])
            went_left = transformed_value <= split_threshold
            child = int(tree.children_left[node] if went_left else tree.children_right[node])
            child_values = np.asarray(tree.value[child][0], dtype=float)
            child_proportions = child_values / child_values.sum()
            positive_idx = int(np.flatnonzero(self.estimator.classes_ == 1)[0])
            child_positive = float(child_proportions[positive_idx])
            contribution = child_positive - parent_positive

            if contract["kind"] == "numeric":
                feature_value: Any = float(frame.iloc[0][contract["feature_name"]])
                scaler = self.preprocessor.named_transformers_["numeric"].named_steps["scale"]
                numeric_idx = [name for name in self.feature_columns if name != "model"].index(contract["feature_name"])
                raw_threshold = split_threshold * float(scaler.scale_[numeric_idx]) + float(scaler.mean_[numeric_idx])
                split_condition = f"{contract['feature_name']} <= {raw_threshold:.12g} (original feature scale)"
                transformed_condition = f"{contract['transformed_feature']} <= {split_threshold:.12g}"
            else:
                feature_value = str(frame.iloc[0]["model"])
                encoded = int(feature_value == contract["category"])
                split_condition = f"one-hot ({contract['feature_name']} == {contract['category']!r}) <= {split_threshold:.12g}"
                transformed_condition = f"{contract['transformed_feature']} <= {split_threshold:.12g}"
                transformed_value = float(encoded)

            path.append({
                "node_id": int(node),
                "feature_name": contract["feature_name"],
                "transformed_feature": contract["transformed_feature"],
                "feature_definition": contract["definition"],
                "feature_value": feature_value,
                "transformed_value": transformed_value,
                "transformed_split_condition": transformed_condition,
                "split_condition_original_scale": split_condition,
                "branch_taken": "left (<= threshold)" if went_left else "right (> threshold)",
                "child_node_id": child,
                "positive_class_share_change_from_parent": contribution,
            })
            node = child
            parent_positive = child_positive

        leaf_values = np.asarray(tree.value[node][0], dtype=float)
        leaf_total = float(leaf_values.sum())
        leaf_proportions = leaf_values / leaf_total
        class_index = int(np.flatnonzero(self.estimator.classes_ == 1)[0])
        leaf_score = float(leaf_proportions[class_index])
        applied_leaf = int(np.asarray(self.estimator.apply(transformed)).reshape(-1)[0])
        if applied_leaf != node or not math.isclose(leaf_score, raw_score, rel_tol=1e-12, abs_tol=1e-12):
            raise ExperimentalArtifactError("Tree-path explanation does not reproduce estimator inference")

        grouped_path: dict[str, dict[str, Any]] = {}
        for item in path:
            grouped = grouped_path.setdefault(item["feature_name"], {
                "feature_name": item["feature_name"],
                "transformed_features": [],
                "feature_definition": item["feature_definition"],
                "feature_values": [],
                "path_node_ids": [],
                "positive_class_share_change_along_path": 0.0,
            })
            if item["transformed_feature"] not in grouped["transformed_features"]:
                grouped["transformed_features"].append(item["transformed_feature"])
            if item["feature_value"] not in grouped["feature_values"]:
                grouped["feature_values"].append(item["feature_value"])
            grouped["path_node_ids"].append(item["node_id"])
            grouped["positive_class_share_change_along_path"] += item["positive_class_share_change_from_parent"]
        top_path = sorted(grouped_path.values(), key=lambda item: (
            -abs(item["positive_class_share_change_along_path"]), item["feature_name"]
        ))[:5]
        return {
            "available": True,
            "method": "exact fitted decision-tree path after saved preprocessing",
            "path_nodes": path,
            "top_path_features": top_path,
            "root_positive_class_share": float(root_proportions[class_index]),
            "leaf": {
                "node_id": node,
                "class_labels": [int(value) for value in self.estimator.classes_],
                "stored_class_proportions": [float(value) for value in leaf_proportions],
                "sample_count": int(tree.n_node_samples[node]),
                "weighted_sample_count": float(tree.weighted_n_node_samples[node]),
                "positive_class_share_matches_raw_score": True,
                "semantics": "Tree leaf class share reproduces this estimator score; it is not calibrated and is not a failure probability.",
            },
            "path_score_accounting": {
                "root_share_plus_path_changes": float(root_proportions[class_index] + sum(
                    item["positive_class_share_change_from_parent"] for item in path
                )),
                "raw_model_score": raw_score,
                "matches_raw_model_score": True,
                "semantics": "Descriptive decomposition of this tree's class share, not causal attribution.",
            },
        }

    def _global_importance(self) -> dict[str, Any]:
        importance = np.asarray(self.estimator.feature_importances_, dtype=float)
        nonzero = [idx for idx, value in enumerate(importance) if value > 0.0]
        ranked = sorted(nonzero, key=lambda idx: (-importance[idx], self.transformed_feature_names[idx]))[:10]
        features = []
        for idx in ranked:
            contract = self._transformed_feature_contract[idx]
            features.append({
                "feature_name": contract["feature_name"],
                "transformed_feature": contract["transformed_feature"],
                "feature_definition": contract["definition"],
                "native_tree_importance": float(importance[idx]),
            })
        return {
            "available": True,
            "method": "DecisionTreeClassifier.feature_importances_ (impurity-based)",
            "scope": "global model-level summary; not an explanation of an individual prediction",
            "ranked_features": features,
            "limitations": [
                "Impurity-based importance can favor features with more split opportunities or cardinality.",
                "Importance is association within this fitted synthetic-benchmark model, not causation.",
            ],
        }

    def predict(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Score a caller-generated, audited 78/79-field historical feature vector."""
        frame, timestamps = self._prepare(request)
        transformed = self.preprocessor.transform(frame[self.feature_columns])
        probabilities = np.asarray(self.estimator.predict_proba(transformed), dtype=float)
        classes = np.asarray(getattr(self.estimator, "classes_", []))
        if probabilities.ndim != 2 or probabilities.shape[0] != 1 or 1 not in classes:
            raise ExperimentalArtifactError("Candidate returned a malformed binary score array")
        score = float(probabilities[0, int(np.flatnonzero(classes == 1)[0])])
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ExperimentalArtifactError("Candidate returned a non-finite or out-of-range score")
        predicted = score >= self.threshold
        local_explanation = self._explain_tree_path(frame, transformed, score)
        return {
            "experimental": True,
            "prediction": "logged_failure_within_horizon" if predicted else "no_logged_failure_classified_within_horizon",
            "predicted_class": int(predicted),
            "threshold": self.threshold,
            "raw_model_score": score,
            "score_semantics": "uncalibrated classifier score; not a calibrated probability",
            "probabilities_calibrated": False,
            "target": TARGET_DESCRIPTION,
            "prediction_cutoff": timestamps["prediction_cutoff"].isoformat(),
            "model": {
                "model_id": self.model_id,
                "run_id": self.run_id,
                "evaluation_split": self.evaluation_split,
                "feature_variant": self.feature_variant,
                "artifact_path": str(self.artifact_path),
                "artifact_sha256": self.artifact_sha256,
                "dataset_sha256": self.dataset_sha256,
                "config_sha256": self.config_sha256,
            },
            "input_quality": {
                "historical_window_complete_as_declared": True,
                "latest_telemetry_strictly_precedes_cutoff": True,
                "latest_telemetry_cadence_hours": 1,
                "feature_pipeline": EXPECTED_FEATURE_PIPELINE,
                "training_range_check": "unavailable; feature ranges were not saved with this artifact",
            },
            "warnings": [
                "Experimental result from a synthetic public benchmark; not validated on company machinery.",
                "The label is a future logged failure-table event, not verified physical failure onset or a guaranteed warning.",
                "The raw score is uncalibrated and must not be interpreted as a probability.",
                "Operational alert budget and minimum useful lead-time requirements are unresolved.",
                "Historical feature values and completeness metadata are caller-supplied and are not independently verified by this service.",
            ],
            "explanation": {
                "available": True,
                "semantics": "Model-behavior explanation only; not a physical diagnosis, causal explanation, or maintenance recommendation.",
                "local_decision_path": local_explanation,
                "global_feature_importance": self._global_importance(),
            },
        }
