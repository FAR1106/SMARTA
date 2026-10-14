# Azure PdM experimental prediction service

This is a local, callable Python interface for the Prompt 2 benchmark candidate. It is intentionally separate from `api.py`: it does not mount an HTTP route, change `/predict`, load from the production model directory, or connect to company systems.

## Call it locally

From the repository root, use the repository virtual environment:

```python
from pathlib import Path
import pandas as pd

from experiments.azure_pdm.pipeline import build_features, load_tables
from experiments.azure_pdm.service import (
    AzurePdmExperimentalService,
    EXPECTED_FEATURE_PIPELINE,
)

archive = Path("/Users/ashishsachinas/Downloads/archive.zip")
tables = load_tables(archive)  # validates the five supplied CSVs
features = build_features(tables)  # trailing-only, prior-only feature generation
service = AzurePdmExperimentalService()  # chronological known-machine candidate

# Choose a row produced by the audited feature pipeline. A live caller must
# supply the same schema from a complete history window, not one raw reading.
row = features.iloc[0]
cutoff = pd.Timestamp(row["datetime"])
request = {
    "features": {name: float(row[name]) for name in service.feature_columns},
    "prediction_cutoff": cutoff.isoformat(),
    "historical_context": {
        "history_start": (cutoff - pd.Timedelta(hours=168)).isoformat(),
        "latest_telemetry_at": (cutoff - pd.Timedelta(hours=1)).isoformat(),
        "telemetry_history_complete": True,
        "event_history_complete_before_cutoff": True,
        "feature_pipeline": EXPECTED_FEATURE_PIPELINE,
    },
}
result = service.predict(request)
print(result)
```

The sample above calls the same existing feature builder on the supplied benchmark archive. The `True` completeness declarations must only be used when the caller has actually verified complete data. The service validates these declarations and timestamps, but cannot independently verify the underlying history or how the caller computed the feature values.

To load the separately saved unseen-machine candidate, pass `evaluation_split="future_unseen_machines"`. That artifact has its own 79-column schema including the categorical `model` value and its own validation-selected threshold. It is not interchangeable with the default artifact.

## Input contract

The default candidate requires exactly the 78 ordered numeric features in `artifacts/run_primary24h_20261013T171456_789146000Z/artifact_manifest.json` under `chronological_known_machines.feature_columns`. The order is checked against the manifest and the serialized preprocessor at load time. It comprises:

- `volt`, `rotate`, `pressure`, and `vibration`: trailing 3-, 24-, and 168-hour mean, population standard deviation, minimum, and maximum, plus trailing 3-hour delta (52 values).
- `past_errorID_error1` through `past_errorID_error5`: counts in `[t-24h,t)` and `[t-168h,t)` (10 values).
- `past_failure_comp1` through `past_failure_comp4` and `past_comp_comp1` through `past_comp_comp4`: prior failure and maintenance counts in the same two windows (16 values).

The source files do not define physical units for the four sensor channels, so the service does not invent or convert units. Counts must be non-negative integers; standard deviations must be non-negative; all numeric features must be finite. Identifiers and timestamps are not model features. For the unseen-machine artifact, `model` must be one of the categories in the saved encoder.

Each call also requires `prediction_cutoff`, an ISO-8601 timestamp; and `historical_context` with `history_start`, `latest_telemetry_at`, `telemetry_history_complete`, `event_history_complete_before_cutoff`, and `feature_pipeline`. Timestamps must have consistent timezone awareness, `latest_telemetry_at` must equal `t - 1 hour`, and at least 168 hours of complete prior telemetry and prior event history must be declared. Feature and event windows exclude the cutoff timestamp and anything after it. A single raw telemetry row is not sufficient to form this contract.

Only the exact engineered features are accepted. Future failures, maintenance, and error outcomes are not fields in the request. Invalid, missing, extra, non-finite, or unsupported values raise `ExperimentalInputError`; an unavailable or mismatched artifact raises `ExperimentalArtifactError`.

## Saved candidate and interpretation

- Run: `run_primary24h_20261013T171456_789146000Z`.
- Default candidate: decision tree with the saved all-features preprocessing pipeline, chronological known-machine split.
- Validation-selected threshold: `0.5825242718446602` (loaded from the saved candidate and cross-checked against its manifest; not selected by this service).
- Artifact SHA-256: `560b9c60ed7367e941f38c60ebcb4f28237ca8e5450b39f896c4d89e3081d24c`.
- Dataset archive SHA-256: `7575444d4d605b1f7155ab937cf541243afe07fce1afba998bf4195e8e84e1eb`.
- Config SHA-256: `f28d7436becf6f7386dd7a129c80a04b1914137569f865e1b890d80ae2864d7c`.

The target is a failure-table event logged in `(t, t+24h]`, any component. This is not a verified physical failure onset, a guaranteed warning, or a company-machine forecast. The response reports the estimator's raw score and threshold decision. That score is uncalibrated and must not be read as a probability. The artifact contains no saved training range metadata, so range checking is reported as unavailable. The service now exposes a faithful decision-tree path and native global tree importance; neither is a physical explanation or a maintenance recommendation.

## Explanation semantics

The local explanation traverses the saved `DecisionTreeClassifier` using the exact transformed row returned by the saved `ColumnTransformer`. Numeric split thresholds are also mapped back through the saved scaler to the corresponding original engineered-feature scale. The response includes each visited node, transformed and original feature names, actual request value, split condition, branch direction, and change in the tree's positive-class share from the parent node to the chosen child. `top_path_features` groups repeated splits by input feature and sums their signed path changes. The root share plus all node changes telescopes to the saved tree's leaf score; the service checks that the leaf from this traversal matches `estimator.apply()` and that its stored positive-class share matches `predict_proba()`.

Leaf proportions and sample counts describe the fitted tree's training leaf. They are not calibrated probabilities. Local path changes describe how this particular tree partitions its training data, not causes of a physical failure. Global feature importance is reported separately from `DecisionTreeClassifier.feature_importances_`, at the transformed feature level. It is an impurity-based summary over the fitted model, not an explanation of a specific prediction; it can favor features with more split opportunities and is not causal. On the unseen-machine artifact, `model` is correctly one-hot encoded in the preprocessor; the fitted tree uses no machine-model dummy in its splits, so model-category importance remains zero and no such node appears in the local paths.

Feature definitions are derived from names and calculations in `pipeline.build_features`: trailing sensor statistics use only completed observations before cutoff; event counts cover `[t-window,t)`. The service accepts engineered features, not raw sensor readings, and does not try to infer physical causes from them. If a future supported artifact does not expose a compatible fitted decision tree, loading fails explicitly instead of returning a made-up explanation.

## Actual benchmark-derived example

The following excerpt was produced by the documented code above using the first row from `build_features(load_tables(archive))` and the exact saved default artifact. It is an offline smoke example, not a validation or test metric. It keeps a few actual path nodes and importance entries for readability; the live response contains the complete path and up to ten nonzero global entries.

```json
{
  "experimental": true,
  "prediction": "no_logged_failure_classified_within_horizon",
  "predicted_class": 0,
  "threshold": 0.5825242718446602,
  "raw_model_score": 0.00017541211648996276,
  "score_semantics": "uncalibrated classifier score; not a calibrated probability",
  "probabilities_calibrated": false,
  "target": "A failure-table event logged in (t, t+24h]; not confirmed physical failure onset.",
  "model": {
    "model_id": "azure-pdm-run_primary24h_20261013T171456_789146000Z-chronological_known_machines-all_features-decision_tree",
    "run_id": "run_primary24h_20261013T171456_789146000Z",
    "evaluation_split": "chronological_known_machines",
    "artifact_sha256": "560b9c60ed7367e941f38c60ebcb4f28237ca8e5450b39f896c4d89e3081d24c"
  },
  "explanation": {
    "available": true,
    "semantics": "Model-behavior explanation only; not a physical diagnosis, causal explanation, or maintenance recommendation.",
    "local_decision_path": {
      "method": "exact fitted decision-tree path after saved preprocessing",
      "path_nodes": [
        {
          "node_id": 0,
          "feature_name": "past_errorID_error5_24h",
          "feature_value": 0.0,
          "split_condition_original_scale": "past_errorID_error5_24h <= 0.500000009281 (original feature scale)",
          "branch_taken": "left (<= threshold)"
        },
        {
          "node_id": 2,
          "feature_name": "volt_mean_24h",
          "feature_value": 171.59645080566406,
          "split_condition_original_scale": "volt_mean_24h <= 185.248108109 (original feature scale)",
          "branch_taken": "left (<= threshold)"
        }
      ],
      "leaf": {
        "node_id": 15,
        "class_labels": [0, 1],
        "stored_class_proportions": [0.99982458788351, 0.00017541211648996276],
        "sample_count": 490274,
        "positive_class_share_matches_raw_score": true,
        "semantics": "Tree leaf class share reproduces this estimator score; it is not calibrated and is not a failure probability."
      },
      "path_score_accounting": {
        "root_share_plus_path_changes": 0.0001754121164899633,
        "raw_model_score": 0.00017541211648996276,
        "matches_raw_model_score": true
      }
    },
    "global_feature_importance": {
      "method": "DecisionTreeClassifier.feature_importances_ (impurity-based)",
      "scope": "global model-level summary; not an explanation of an individual prediction",
      "ranked_features": [
        {
          "feature_name": "past_errorID_error2_24h",
          "native_tree_importance": 0.21926824047667187
        },
        {
          "feature_name": "past_errorID_error1_24h",
          "native_tree_importance": 0.1855391615510956
        }
      ]
    }
  },
  "warnings": [
    "Experimental result from a synthetic public benchmark; not validated on company machinery.",
    "The raw score is uncalibrated and must not be interpreted as a probability."
  ]
}
```

## Limits and evidence needed before deployment

This remains an offline experiment on a synthetic public benchmark. The supplied archive's redistribution/deployment license is unverified. The service has no company telemetry ingestion, event-time reconciliation, or authenticated provenance mechanism. Before considering company use, obtain permissioned timestamped company sensor history with documented units and sampling gaps, verified failure and intervention records, operating exposure, reporting-delay semantics, and a time-respecting company holdout. Agree on alert budget, minimum event recall, and useful lead time; assess calibration on separate time-respecting data if probabilities will be shown. None of those deployment gates is established here.

Run the focused service tests with:

```bash
../.venv/bin/python -m unittest discover -s experiments/azure_pdm/tests -v
```

For the explicit chat opt-in command, evidence boundaries, and failure behavior, see [experimental_chat_integration.md](experimental_chat_integration.md).
