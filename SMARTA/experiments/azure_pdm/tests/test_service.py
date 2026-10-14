from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.azure_pdm.service import (
    DEFAULT_RUN_ID,
    EXPECTED_FEATURE_PIPELINE,
    AzurePdmExperimentalService,
    ExperimentalArtifactError,
    ExperimentalInputError,
)


ROOT = Path(__file__).resolve().parents[1]


class StubEstimator:
    classes_ = np.array([0, 1])

    def __init__(self, score: float):
        self.score = score

    def predict_proba(self, _matrix):
        return np.array([[1.0 - self.score, self.score]])


def valid_request(service: AzurePdmExperimentalService) -> dict:
    features = {
        column: ("model1" if column == "model" else 1.0)
        for column in service.feature_columns
    }
    return {
        "features": features,
        "prediction_cutoff": "2020-01-08T00:00:00",
        "historical_context": {
            "history_start": "2020-01-01T00:00:00",
            "latest_telemetry_at": "2020-01-07T23:00:00",
            "telemetry_history_complete": True,
            "event_history_complete_before_cutoff": True,
            "feature_pipeline": EXPECTED_FEATURE_PIPELINE,
        },
    }


class ExperimentalServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(
            (ROOT / "artifacts" / DEFAULT_RUN_ID / "artifact_manifest.json").read_text()
        )

    def test_loads_exact_saved_chronological_candidate_and_feature_order(self):
        service = AzurePdmExperimentalService()
        artifact = self.manifest["artifacts"]["chronological_known_machines"]
        self.assertEqual(service.artifact_sha256, artifact["sha256"])
        self.assertEqual(service.feature_columns, artifact["feature_columns"])
        self.assertEqual(service.threshold, artifact["threshold_selected_on_validation"])
        self.assertEqual(service.feature_variant, "all_features")
        frame, _ = service._prepare(valid_request(service))
        self.assertEqual(frame.columns.tolist(), artifact["feature_columns"])

    def test_loads_separate_unseen_machine_candidate_with_model_category(self):
        service = AzurePdmExperimentalService(evaluation_split="future_unseen_machines")
        self.assertIn("model", service.feature_columns)
        self.assertIn("model1", service.allowed_machine_models)
        self.assertEqual(
            service.threshold,
            self.manifest["artifacts"]["future_unseen_machines"]["threshold_selected_on_validation"],
        )

    def test_valid_vector_returns_finite_experimental_result_and_warnings(self):
        service = AzurePdmExperimentalService()
        result = service.predict(valid_request(service))
        self.assertTrue(result["experimental"])
        self.assertIn(result["predicted_class"], (0, 1))
        self.assertTrue(math.isfinite(result["raw_model_score"]))
        self.assertEqual(result["threshold"], service.threshold)
        self.assertEqual(result["target"], "A failure-table event logged in (t, t+24h]; not confirmed physical failure onset.")
        self.assertEqual(result["model"]["artifact_sha256"], service.artifact_sha256)
        self.assertTrue(any("synthetic" in warning for warning in result["warnings"]))
        self.assertTrue(any("unresolved" in warning for warning in result["warnings"]))
        self.assertTrue(result["explanation"]["available"])
        self.assertIn("not independently verified", result["warnings"][-1])
        self.assertFalse(result["probabilities_calibrated"])
        self.assertIn("not calibrated", result["explanation"]["local_decision_path"]["leaf"]["semantics"].lower())

    def test_local_path_matches_actual_preprocessed_tree_traversal_and_score(self):
        service = AzurePdmExperimentalService()
        request = valid_request(service)
        result = service.predict(request)
        local = result["explanation"]["local_decision_path"]
        frame, _ = service._prepare(request)
        transformed = service.preprocessor.transform(frame[service.feature_columns])
        tree = service.estimator.tree_
        vector = np.asarray(transformed).reshape(-1)
        node = 0
        observed = []
        while tree.children_left[node] != tree.children_right[node]:
            feature_index = int(tree.feature[node])
            went_left = vector[feature_index] <= tree.threshold[node]
            observed.append((node, feature_index, went_left))
            node = int(tree.children_left[node] if went_left else tree.children_right[node])

        self.assertEqual(node, local["leaf"]["node_id"])
        self.assertEqual(len(observed), len(local["path_nodes"]))
        for (node_id, feature_index, went_left), decision in zip(observed, local["path_nodes"]):
            with self.subTest(node=node_id):
                self.assertEqual(decision["node_id"], node_id)
                self.assertEqual(decision["transformed_feature"], service.transformed_feature_names[feature_index])
                self.assertEqual(decision["branch_taken"], "left (<= threshold)" if went_left else "right (> threshold)")
                if decision["feature_name"] != "model":
                    self.assertEqual(decision["feature_value"], request["features"][decision["feature_name"]])
                    self.assertIn("original feature scale", decision["split_condition_original_scale"])

        leaf_values = tree.value[node][0].astype(float)
        leaf_proportions = leaf_values / leaf_values.sum()
        self.assertEqual(local["leaf"]["stored_class_proportions"], leaf_proportions.tolist())
        self.assertEqual(local["leaf"]["sample_count"], int(tree.n_node_samples[node]))
        self.assertAlmostEqual(local["path_score_accounting"]["root_share_plus_path_changes"], result["raw_model_score"])
        self.assertAlmostEqual(local["leaf"]["stored_class_proportions"][1], result["raw_model_score"])
        self.assertTrue(local["path_score_accounting"]["matches_raw_model_score"])
        self.assertFalse(result["probabilities_calibrated"])
        changes_by_feature = {}
        for decision in local["path_nodes"]:
            changes_by_feature[decision["feature_name"]] = changes_by_feature.get(decision["feature_name"], 0.0) + decision["positive_class_share_change_from_parent"]
        for top_feature in local["top_path_features"]:
            self.assertAlmostEqual(
                top_feature["positive_class_share_change_along_path"],
                changes_by_feature[top_feature["feature_name"]],
            )

    def test_global_tree_importance_is_artifact_derived_and_separate_from_local(self):
        service = AzurePdmExperimentalService()
        result = service.predict(valid_request(service))
        local = result["explanation"]["local_decision_path"]
        global_summary = result["explanation"]["global_feature_importance"]
        self.assertEqual(global_summary["method"], "DecisionTreeClassifier.feature_importances_ (impurity-based)")
        self.assertNotEqual(global_summary["scope"], "local")
        for item in global_summary["ranked_features"]:
            index = service.transformed_feature_names.index(item["transformed_feature"])
            self.assertEqual(item["native_tree_importance"], float(service.estimator.feature_importances_[index]))
            self.assertTrue(item["feature_definition"])
        self.assertIn("global model-level", global_summary["scope"])
        self.assertIn("causation", " ".join(global_summary["limitations"]))
        self.assertTrue(local["path_nodes"])

    def test_unseen_machine_explanation_maps_one_hot_feature_and_preserves_schema(self):
        service = AzurePdmExperimentalService(evaluation_split="future_unseen_machines")
        request = valid_request(service)
        result = service.predict(request)
        self.assertEqual(len(service.feature_columns), 79)
        self.assertEqual(len(service.transformed_feature_names), 82)
        encoded = np.asarray(service.preprocessor.transform(
            pd.DataFrame([request["features"]], columns=service.feature_columns)
        )).reshape(-1)
        for contract in service._transformed_feature_contract:
            if contract["kind"] == "one_hot":
                index = service.transformed_feature_names.index(contract["transformed_feature"])
                self.assertEqual(encoded[index], float(contract["category"] == "model1"))
                self.assertEqual(contract["feature_name"], "model")
        for decision in result["explanation"]["local_decision_path"]["path_nodes"]:
            if decision["feature_name"] == "model":
                self.assertEqual(decision["feature_value"], "model1")
                self.assertIn("one-hot", decision["split_condition_original_scale"])
        for item in result["explanation"]["global_feature_importance"]["ranked_features"]:
            self.assertIn(item["transformed_feature"], service.transformed_feature_names)

    def test_decision_uses_saved_validation_threshold(self):
        service = AzurePdmExperimentalService()
        request = valid_request(service)
        service._explain_tree_path = lambda *_args: {"available": False, "reason": "threshold unit test"}
        service._global_importance = lambda: {"available": False, "reason": "threshold unit test"}
        service.estimator = StubEstimator(service.threshold - 1e-6)
        self.assertEqual(service.predict(request)["predicted_class"], 0)
        service.estimator = StubEstimator(service.threshold + 1e-6)
        self.assertEqual(service.predict(request)["predicted_class"], 1)

    def test_missing_malformed_nonfinite_and_unsupported_features_fail(self):
        service = AzurePdmExperimentalService()
        request = valid_request(service)
        del request["features"][service.feature_columns[0]]
        with self.assertRaisesRegex(ExperimentalInputError, "Missing engineered features"):
            service.predict(request)

        for bad in ("not-numeric", True, float("nan"), float("inf"), -float("inf")):
            request = valid_request(service)
            request["features"][service.feature_columns[0]] = bad
            with self.subTest(bad=bad), self.assertRaises(ExperimentalInputError):
                service.predict(request)

        request = valid_request(service)
        request["features"]["future_failure"] = 1
        with self.assertRaisesRegex(ExperimentalInputError, "Unsupported engineered features"):
            service.predict(request)

    def test_historical_cutoff_and_completeness_are_enforced(self):
        service = AzurePdmExperimentalService()
        for field, value in (
            ("latest_telemetry_at", "2020-01-08T00:00:00"),
            ("latest_telemetry_at", "2020-01-07T22:00:00"),
            ("history_start", "2020-01-01T01:00:00"),
        ):
            request = valid_request(service)
            request["historical_context"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ExperimentalInputError):
                service.predict(request)
        request = valid_request(service)
        request["historical_context"]["event_history_complete_before_cutoff"] = False
        with self.assertRaisesRegex(ExperimentalInputError, "event streams"):
            service.predict(request)

    def test_service_rejects_extra_request_keys_and_wrong_feature_pipeline(self):
        service = AzurePdmExperimentalService()
        request = valid_request(service)
        request["future_failures"] = []
        with self.assertRaisesRegex(ExperimentalInputError, "Unsupported request fields"):
            service.predict(request)
        request = valid_request(service)
        request["historical_context"]["feature_pipeline"] = "custom"
        with self.assertRaisesRegex(ExperimentalInputError, "feature_pipeline"):
            service.predict(request)

    def test_invalid_artifact_split_fails_descriptively(self):
        with self.assertRaises(ExperimentalArtifactError):
            AzurePdmExperimentalService(evaluation_split="locked_test")

    def test_service_only_accepts_the_audited_prompt2_run(self):
        with self.assertRaisesRegex(ExperimentalArtifactError, "Only the audited Prompt 2"):
            AzurePdmExperimentalService(run_id="run_primary24h_other")


if __name__ == "__main__":
    unittest.main()
