from __future__ import annotations

import unittest

import pandas as pd

from experiments.azure_pdm.train_primary_24h import feature_columns, selection_key


class Primary24hTests(unittest.TestCase):
    def test_feature_variants_never_include_identifiers_or_targets(self):
        frame = pd.DataFrame(columns=[
            "machineID", "datetime", "model", "volt_mean_3h", "past_errorID_error1_24h",
            "past_failure_comp1_24h", "y_24h_any", "valid_24h",
        ])
        self.assertEqual(feature_columns(frame, "all_features"), [
            "volt_mean_3h", "past_errorID_error1_24h", "past_failure_comp1_24h",
        ])
        self.assertEqual(feature_columns(frame, "without_error_history"), [
            "volt_mean_3h", "past_failure_comp1_24h",
        ])
        self.assertEqual(feature_columns(frame, "telemetry_only"), ["volt_mean_3h"])
        self.assertEqual(feature_columns(frame, "plus_model_ablation"), [
            "volt_mean_3h", "past_errorID_error1_24h", "past_failure_comp1_24h", "model",
        ])

    def test_selection_key_uses_validation_ap_and_deterministic_tie_break(self):
        higher_validation = {
            "validation_average_precision": 0.7,
            "feature_variant": "without_error_history",
            "model": "logistic",
            "test_average_precision": 0.01,
        }
        lower_validation = {
            "validation_average_precision": 0.6,
            "feature_variant": "all_features",
            "model": "decision_tree",
            "test_average_precision": 0.99,
        }
        self.assertGreater(selection_key(higher_validation), selection_key(lower_validation))
        self.assertLess(selection_key({**higher_validation, "validation_average_precision": 0.7,
                                      "feature_variant": "all_features"}),
                        selection_key({**higher_validation, "validation_average_precision": 0.7,
                                      "feature_variant": "without_error_history"}))


if __name__ == "__main__":
    unittest.main()
