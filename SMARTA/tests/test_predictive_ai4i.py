import json
import asyncio
import contextlib
import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi import HTTPException

import api
from api import ChatRequest, PredictRequest, chat, predict, predictive_model_metadata
from predictive.ai4i import (
    DATA_DIR, DEFAULT_ARCHIVE, FEATURES, FAILURE_TARGET, MODE_TARGETS,
    ModelUnavailable, Predictor, dataset_report, inspect_archive, load_dataset,
    train_model,
)


class Ai4iTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.csv_path = DATA_DIR / "ai4i2020.csv"
        if not cls.csv_path.exists() and DEFAULT_ARCHIVE.exists():
            inspect_archive(DEFAULT_ARCHIVE)
        if not cls.csv_path.exists():
            raise unittest.SkipTest("Supplied AI4I dataset archive is not available")
        cls.frame = load_dataset(cls.csv_path)

    def synthetic(self, rows=100):
        rng = np.random.default_rng(27)
        frame = pd.DataFrame({
            "UDI": np.arange(rows), "Product ID": [f"P{i}" for i in range(rows)],
            "Type": rng.choice(["L", "M", "H"], rows),
            "Air temperature [K]": rng.uniform(295, 305, rows),
            "Process temperature [K]": rng.uniform(300, 315, rows),
            "Rotational speed [rpm]": rng.integers(1000, 2000, rows),
            "Torque [Nm]": rng.uniform(20, 70, rows), "Tool wear [min]": rng.integers(0, 250, rows),
        })
        frame[FAILURE_TARGET] = np.array([0] * (rows - 20) + [1] * 20)
        for index, target in enumerate(MODE_TARGETS):
            frame[target] = np.array([0] * (rows - 24 + index) + [1] * (24 - index))
        return frame

    def test_zip_discovery_extracts_csv_without_changing_archive(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "dataset.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("nested/actual.csv", "a,b\n1,2\n")
            before = archive.read_bytes()
            result = inspect_archive(archive, root / "out")
            self.assertEqual(result.name, "actual.csv")
            self.assertEqual(archive.read_bytes(), before)

    def test_real_dataset_schema_and_quality(self):
        report = dataset_report(self.frame)
        self.assertEqual(report["rows"], 10000)
        self.assertEqual(report["duplicate_rows"], 0)
        self.assertEqual(sum(report["missing_values"].values()), 0)
        self.assertEqual(report["class_distribution"][FAILURE_TARGET], {"0": 9661, "1": 339})
        # Preserve this finding from the supplied file instead of assuming the
        # machine-failure flag is exactly the union of the five mode flags.
        self.assertEqual(
            report["target_consistency"]["machine_failure_mode_union_mismatch_rows"],
            27,
        )
        self.assertEqual(
            report["target_consistency"]["machine_failure_without_mode_label_rows"],
            9,
        )
        self.assertEqual(
            report["target_consistency"]["mode_label_without_machine_failure_rows"],
            18,
        )

    def test_identifier_and_outcome_columns_are_not_model_features(self):
        self.assertEqual(FEATURES, ["Type", "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"])
        self.assertNotIn("UDI", FEATURES)
        self.assertNotIn("Product ID", FEATURES)
        self.assertNotIn(FAILURE_TARGET, FEATURES)
        self.assertFalse(set(MODE_TARGETS) & set(FEATURES))

    def test_train_serializes_reloadable_models_and_metrics_reproduce(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "fixture.csv"
            self.synthetic().to_csv(source, index=False)
            first_dir, second_dir = Path(temp) / "one", Path(temp) / "two"
            first = train_model(source, first_dir, seed=13)
            second = train_model(source, second_dir, seed=13)
            self.assertEqual(first["machine_failure"]["random_forest"]["confusion_matrix"], second["machine_failure"]["random_forest"]["confusion_matrix"])
            self.assertEqual(first["machine_failure"]["random_forest"]["f1"], second["machine_failure"]["random_forest"]["f1"])
            model = Predictor(first_dir / "ai4i_model.joblib")
            sample = self.synthetic().iloc[0]
            output = model.predict({name: sample[name] for name in FEATURES}, "test-1")
            self.assertIn(output["machine_failure"]["classification"], (0, 1))
            self.assertEqual(output["model_version"], "ai4i-rf-v1")
            self.assertTrue((first_dir / "ai4i_metrics.json").exists())
            json.loads((first_dir / "ai4i_metrics.json").read_text())

    def test_untrained_and_invalid_requests_fail_clearly(self):
        with tempfile.TemporaryDirectory() as temp:
            predictor = Predictor(Path(temp) / "missing.joblib")
            with self.assertRaises(ModelUnavailable):
                predictor.predict({})
            predictor = Predictor()
            row = self.frame.iloc[0]
            valid = {name: row[name] for name in FEATURES}
            with self.assertRaisesRegex(ValueError, "missing"):
                predictor.predict({"Type": "M"})
            valid["extra"] = 1
            with self.assertRaisesRegex(ValueError, "unexpected"):
                predictor.predict(valid)
            valid.pop("extra")
            valid["Type"] = "X"
            with self.assertRaisesRegex(ValueError, "Type"):
                predictor.predict(valid)
            valid["Type"] = "M"
            valid["Tool wear [min]"] = 99999
            accepted = predictor.predict(valid)
            self.assertEqual(accepted["input_features"]["Tool wear [min]"], 99999.0)
            self.assertIn("Tool wear [min]", accepted["out_of_training_range"])
            with self.assertRaisesRegex(ValueError, "negative"):
                valid["Tool wear [min]"] = -1
                predictor.predict(valid)

    def test_non_numeric_and_non_finite_feature_values_are_rejected(self):
        row = self.frame.iloc[0]
        for value, message in (("not-a-number", "numeric"), (float("nan"), "finite"),
                               (float("inf"), "finite")):
            features = {name: row[name] for name in FEATURES}
            features["Air temperature [K]"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, message):
                Predictor().predict(features)

    def test_inference_accepts_finite_out_of_range_inputs_and_reports_them(self):
        row = self.frame.iloc[0]
        features = {name: row[name] for name in FEATURES}
        features["Air temperature [K]"] = 310.0
        features["Process temperature [K]"] = 299.0
        features["Rotational speed [rpm]"] = 500.0
        features["Torque [Nm]"] = 90.0
        features["Tool wear [min]"] = 300.0
        result = Predictor().predict(features)
        self.assertIn("Air temperature [K]", result["out_of_training_range"])
        self.assertIn("Process temperature [K]", result["out_of_training_range"])
        self.assertIn("Rotational speed [rpm]", result["out_of_training_range"])
        self.assertIn("Torque [Nm]", result["out_of_training_range"])
        self.assertIn("Tool wear [min]", result["out_of_training_range"])
        self.assertEqual(result["input_features"]["Process temperature [K]"], 299.0)

    def test_training_range_endpoints_are_not_warned(self):
        row = self.frame.iloc[0]
        features = {name: row[name] for name in FEATURES}
        features.update({
            "Air temperature [K]": 295.3,
            "Process temperature [K]": 305.7,
            "Rotational speed [rpm]": 1168,
            "Torque [Nm]": 3.8,
            "Tool wear [min]": 0,
        })
        result = Predictor().predict(features)
        self.assertEqual(result["out_of_training_range"], {})

    def test_physical_invalid_temperature_and_wear_are_rejected(self):
        row = self.frame.iloc[0]
        features = {name: row[name] for name in FEATURES}
        for name, value in (("Air temperature [K]", 0),
                            ("Process temperature [K]", -2),
                            ("Tool wear [min]", -0.1)):
            invalid = dict(features)
            invalid[name] = value
            with self.subTest(feature=name), self.assertRaises(ValueError):
                Predictor().predict(invalid)

    def test_real_model_predicts_and_rare_mode_remains_unsupported(self):
        report_path = Path(__file__).resolve().parents[1] / "models" / "ai4i_metrics.json"
        self.assertTrue(report_path.exists(), "Train model before running test suite")
        report = json.loads(report_path.read_text())
        self.assertFalse(report["failure_modes"]["RNF"]["supported"])
        row = self.frame.iloc[0]
        result = Predictor().predict({name: row[name] for name in FEATURES}, "test-machine")
        self.assertEqual(result["machine_id"], "test-machine")
        self.assertIn("uncalibrated", result["machine_failure"]["score_type"])
        self.assertNotIn("RNF", result["failure_modes"])
        self.assertIn("RNF", result["unsupported_failure_modes"])

    def test_saved_twf_evaluation_exposes_zero_detection_at_default_cutoff(self):
        report_path = Path(__file__).resolve().parents[1] / "models" / "ai4i_metrics.json"
        report = json.loads(report_path.read_text())
        metrics = report["failure_modes"]["TWF"]["metrics"]
        self.assertEqual(report["failure_modes"]["TWF"]["test_positive_examples"], 9)
        self.assertEqual(metrics["confusion_matrix"], [[1991, 0], [9, 0]])
        self.assertEqual(metrics["f1"], 0.0)
        # Ranking has signal, but not at the estimator's default 0.5 cutoff.
        self.assertGreater(metrics["roc_auc"], 0.9)
        self.assertGreater(metrics["pr_auc"], 0.01)

    def test_prediction_endpoint_uses_real_model_and_returns_mode_scores(self):
        request = PredictRequest(machine_id="Machine-01", features={
            "product_type": "M", "air_temperature_k": 298.1,
            "process_temperature_k": 308.6, "rotational_speed_rpm": 1551,
            "torque_nm": 42.8, "tool_wear_min": 20,
        })
        with patch("api.retrieve", return_value=[]):
            response = predict(request)
        self.assertEqual(response.prediction["model_version"], "ai4i-rf-v1")
        self.assertIn(response.prediction["machine_failure"]["classification"], (0, 1))
        self.assertEqual(response.evidence_status, "insufficient")
        self.assertFalse(response.evidence)
        self.assertIn("RNF", response.prediction["unsupported_failure_modes"])
        for mode in response.prediction["failure_modes"].values():
            self.assertGreaterEqual(mode["score"], 0)
            self.assertLessEqual(mode["score"], 1)
            self.assertEqual(mode["score_type"], "uncalibrated_random_forest_score")

    def test_prediction_api_accepts_process_temperature_below_300_kelvin(self):
        request = PredictRequest(machine_id="Below-300K", features={
            "product_type": "L", "air_temperature_k": 290.0,
            "process_temperature_k": 299.0, "rotational_speed_rpm": 1000,
            "torque_nm": 80.0, "tool_wear_min": 260.0,
        })
        with patch("api.retrieve", return_value=[]):
            response = predict(request)
        self.assertEqual(response.prediction["input_features"]["Process temperature [K]"], 299.0)
        self.assertIn("Process temperature [K]", response.prediction["out_of_training_range"])
        self.assertIn("Air temperature [K]", response.prediction["out_of_training_range"])

    def test_inference_scores_and_labels_match_loaded_estimator_outputs(self):
        bundle = Predictor().load()
        cases = [
            {
                "Type": "M",
                "Air temperature [K]": 298.1,
                "Process temperature [K]": 308.6,
                "Rotational speed [rpm]": 1551,
                "Torque [Nm]": 42.8,
                "Tool wear [min]": 20,
            },
            {
                "Type": "H",
                "Air temperature [K]": 304.5,
                "Process temperature [K]": 313.8,
                "Rotational speed [rpm]": 1168,
                "Torque [Nm]": 76.6,
                "Tool wear [min]": 253,
            },
        ]
        for features in cases:
            x = pd.DataFrame([features], columns=FEATURES)
            actual = Predictor().predict(features)
            machine = bundle["machine_failure"]
            self.assertEqual(
                actual["machine_failure"]["classification"],
                int(machine.predict(x)[0]),
            )
            self.assertAlmostEqual(
                actual["machine_failure"]["score"],
                float(machine.predict_proba(x)[0, 1]),
                places=14,
            )
            for mode, estimator in bundle["failure_modes"].items():
                self.assertEqual(
                    actual["failure_modes"][mode]["classification"],
                    int(estimator.predict(x)[0]),
                )
                self.assertAlmostEqual(
                    actual["failure_modes"][mode]["score"],
                    float(estimator.predict_proba(x)[0, 1]),
                    places=14,
                )
        ordinary = Predictor().predict(cases[0])
        edge = Predictor().predict(cases[1])
        self.assertEqual(ordinary["machine_failure"]["score"], 0.0)
        self.assertEqual(ordinary["machine_failure"]["classification"], 0)
        self.assertEqual(edge["machine_failure"]["classification"], 1)
        self.assertEqual(edge["failure_modes"]["OSF"]["classification"], 1)
        self.assertGreater(edge["failure_modes"]["OSF"]["score"], 0.5)
        self.assertEqual(edge["failure_modes"]["TWF"]["classification"], 0)
        self.assertLess(edge["failure_modes"]["TWF"]["score"], 0.5)

    def test_metadata_and_existing_chat_contract_remain_available(self):
        metadata = predictive_model_metadata()
        self.assertTrue(metadata["ready"])
        self.assertEqual(metadata["model_version"], "ai4i-rf-v1")
        self.assertIn("Air temperature [K]", metadata["feature_ranges"])

        old_status = api._corpus_status
        api._corpus_status = {"documents": 2, "maintenance_records": 1, "sources": 1}
        try:
            retrieved = [{"source": "logs.xlsx", "locator": "row 3", "text": "SMT13 fault: vibration"}]
            with patch("api.retrieve", return_value=retrieved), patch("api.generate_answer", return_value="Recorded vibration. [1]"):
                response = chat(ChatRequest(question="What happened to SMT13?"))
            self.assertIn("Recorded vibration", response.answer)
            self.assertEqual(response.sources, ["[1] logs.xlsx, row 3"])
        finally:
            api._corpus_status = old_status

    def test_chat_does_not_expose_internal_provider_errors(self):
        old_status = api._corpus_status
        api._corpus_status = {"documents": 1, "maintenance_records": 0, "sources": 1}
        try:
            with patch("api.retrieve", return_value=[{
                "source": "manual.pdf", "locator": "page 1", "text": "A safe excerpt."
            }]), patch("api.generate_answer", side_effect=RuntimeError("credential-bearing internal detail")):
                with self.assertRaises(HTTPException) as error:
                    chat(ChatRequest(question="Explain this maintenance issue"))
            self.assertEqual(error.exception.status_code, 502)
            self.assertNotIn("credential-bearing", str(error.exception.detail))
            self.assertEqual(error.exception.detail["code"], "generation_failed")
        finally:
            api._corpus_status = old_status

    def test_corpus_startup_error_is_sanitized(self):
        api._ingestion_error = None
        output = io.StringIO()

        async def enter_lifespan():
            async with api.lifespan(api.app):
                return

        with patch("api.initialize_corpus", side_effect=RuntimeError("secret connection URL")), contextlib.redirect_stdout(output):
            asyncio.run(enter_lifespan())
        self.assertEqual(api._ingestion_error, "corpus_initialization_failed")
        self.assertNotIn("secret connection URL", output.getvalue())

    def test_api_handles_untrained_model(self):
        request = PredictRequest(machine_id="test", features={
            "product_type": "M", "air_temperature_k": 298.1,
            "process_temperature_k": 308.6, "rotational_speed_rpm": 1551,
            "torque_nm": 42.8, "tool_wear_min": 20,
        })
        with patch("api._predictor.predict", side_effect=ModelUnavailable("not trained")):
            with self.assertRaises(HTTPException) as error:
                predict(request)
            self.assertEqual(error.exception.status_code, 503)
        with patch("api._predictor.metadata", side_effect=ModelUnavailable("not trained")):
            with self.assertRaises(HTTPException) as error:
                predictive_model_metadata()
            self.assertEqual(error.exception.status_code, 503)

    def test_api_rejects_blank_machine_identifier_and_extra_features(self):
        request = PredictRequest(machine_id="   ", features={
            "product_type": "M", "air_temperature_k": 298.1,
            "process_temperature_k": 308.6, "rotational_speed_rpm": 1551,
            "torque_nm": 42.8, "tool_wear_min": 20,
        })
        with self.assertRaises(HTTPException) as error:
            predict(request)
        self.assertEqual(error.exception.status_code, 422)
        with self.assertRaises(Exception):
            PredictRequest(machine_id="machine", features={
                "product_type": "M", "air_temperature_k": 298.1,
                "process_temperature_k": 308.6, "rotational_speed_rpm": 1551,
                "torque_nm": 42.8, "tool_wear_min": 20, "unexpected": 1,
            })

    def test_api_separates_prediction_from_rag_and_marks_no_machine_match(self):
        request = PredictRequest(machine_id="MACHINE-ZZ", features={
            "product_type": "M", "air_temperature_k": 298.1,
            "process_temperature_k": 308.6, "rotational_speed_rpm": 1551,
            "torque_nm": 42.8, "tool_wear_min": 20,
        })
        fake_prediction = {"machine_failure": {"classification": 1, "score": .8}, "failure_modes": {}, "model_version": "test"}
        evidence = [{"text": "Manual mentions general inspection.", "source": "manual.pdf", "locator": "page 1"}]
        with patch("api._predictor.predict", return_value=fake_prediction), patch("api.retrieve", return_value=evidence), patch("api.generate_answer", return_value="Inspect per manual. [1]"):
            result = predict(request)
        self.assertEqual(result.evidence_status, "general_only")
        self.assertEqual(result.machine_specific_evidence, [])
        self.assertEqual(result.prediction["machine_failure"]["classification"], 1)
        self.assertIn("[1]", result.recommendation)

    def test_api_identifies_exact_machine_evidence_and_returns_insufficient_when_empty(self):
        request = PredictRequest(machine_id="SMT13", features={
            "product_type": "M", "air_temperature_k": 298.1,
            "process_temperature_k": 308.6, "rotational_speed_rpm": 1551,
            "torque_nm": 42.8, "tool_wear_min": 20,
        })
        fake_prediction = {"machine_failure": {"classification": 1, "score": .8}, "failure_modes": {}, "model_version": "test"}
        specific = [{"text": "SMT13 has recorded tool wear inspection.", "source": "logs.xlsx", "locator": "row 3"}]
        with patch("api._predictor.predict", return_value=fake_prediction), patch("api.retrieve", return_value=specific), patch("api.generate_answer", return_value="Possible checks supported by the log. [1]"):
            result = predict(request)
        self.assertEqual(result.evidence_status, "machine_specific_and_general")
        self.assertEqual(len(result.machine_specific_evidence), 1)
        with patch("api._predictor.predict", return_value=fake_prediction), patch("api.retrieve", return_value=[]):
            result = predict(request)
        self.assertEqual(result.evidence_status, "insufficient")
        self.assertIn("No sufficiently supported corrective action", result.recommendation)


if __name__ == "__main__":
    unittest.main()
