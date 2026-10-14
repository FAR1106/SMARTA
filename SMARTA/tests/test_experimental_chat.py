from __future__ import annotations

import copy
import json
import unittest
from unittest.mock import Mock, patch

import api
from api import ChatRequest
from experiments.azure_pdm.chat_adapter import (
    ExperimentalRequestError,
    handle_experimental_chat,
    is_experimental_request,
    parse_experimental_request,
)
from experiments.azure_pdm.service import (
    EXPECTED_FEATURE_PIPELINE,
    AzurePdmExperimentalService,
)
from rag.generator import generate_answer


def valid_request(service: AzurePdmExperimentalService) -> dict:
    return {
        "features": {name: ("model1" if name == "model" else 1.0)
                     for name in service.feature_columns},
        "prediction_cutoff": "2020-01-08T00:00:00",
        "historical_context": {
            "history_start": "2020-01-01T00:00:00",
            "latest_telemetry_at": "2020-01-07T23:00:00",
            "telemetry_history_complete": True,
            "event_history_complete_before_cutoff": True,
            "feature_pipeline": EXPECTED_FEATURE_PIPELINE,
        },
    }


class RecordingService:
    def __init__(self, response, delegate=None):
        self.response = response
        self.delegate = delegate
        self.requests = []

    def predict(self, request):
        self.requests.append(request)
        if self.delegate is not None:
            return self.delegate.predict(request)
        return copy.deepcopy(self.response)


class ExperimentalChatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = AzurePdmExperimentalService()
        cls.request = valid_request(cls.model)
        cls.evidence = cls.model.predict(cls.request)

    def command(self, request=None, question="What do the model path and maintenance records support?"):
        return "/experimental-pdm " + json.dumps({
            "request": request if request is not None else self.request,
            "question": question,
        })

    def test_only_an_explicit_prefix_is_opted_in_and_parser_is_strict(self):
        self.assertFalse(is_experimental_request("Why did the pump stop?"))
        self.assertFalse(is_experimental_request("Please explain /experimental-pdm"))
        self.assertTrue(is_experimental_request("  /EXPERIMENTAL-PDM {}"))
        with self.assertRaises(ExperimentalRequestError):
            parse_experimental_request("/experimental-pdm {bad json}")
        with self.assertRaises(ExperimentalRequestError):
            parse_experimental_request(self.command() + " trailing text")

    def test_explicit_request_calls_candidate_once_and_keeps_evidence_and_docs_separate(self):
        service = RecordingService(self.evidence)
        retrieved = [{"source": "manual.pdf", "locator": "page 7", "text": "Inspect the inlet filter."}]
        generated = {}

        def generate(**kwargs):
            generated.update(kwargs)
            return "The classifier followed the recorded path. The manual supports a filter inspection [1]."

        result = handle_experimental_chat(
            self.command(), lambda query: retrieved, generate, corpus_ready=True,
            service_factory=lambda: service,
        )
        self.assertEqual(service.requests, [self.request])
        self.assertEqual(result.sources, ("[1] manual.pdf, page 7",))
        self.assertIn("[1] manual.pdf, page 7\nInspect the inlet filter.", generated["context"])
        self.assertTrue(generated["prediction_evidence"]["experimental"])
        self.assertEqual(generated["prediction_evidence"]["raw_model_score"], self.evidence["raw_model_score"])
        self.assertNotIn("artifact_path", json.dumps(generated["prediction_evidence"]))

    def test_missing_history_fails_without_prediction_retrieval_or_generation(self):
        incomplete = copy.deepcopy(self.request)
        del incomplete["historical_context"]
        service = RecordingService(None, delegate=self.model)
        retrieve_fn, generate_fn = Mock(), Mock()
        result = handle_experimental_chat(
            self.command(incomplete), retrieve_fn, generate_fn, corpus_ready=True,
            service_factory=lambda: service,
        )
        self.assertEqual(len(service.requests), 1)
        self.assertEqual(result.outcome, "prediction_input_invalid")
        self.assertIn("No experimental prediction was run", result.answer)
        retrieve_fn.assert_not_called()
        generate_fn.assert_not_called()

    def test_malformed_or_contradictory_service_evidence_is_rejected_before_llm(self):
        contradictory = copy.deepcopy(self.evidence)
        contradictory["predicted_class"] = 1 - contradictory["predicted_class"]
        service = RecordingService(contradictory)
        retrieve_fn, generate_fn = Mock(), Mock()
        result = handle_experimental_chat(
            self.command(), retrieve_fn, generate_fn, corpus_ready=True,
            service_factory=lambda: service,
        )
        self.assertEqual(result.outcome, "evidence_invalid")
        self.assertIn("inconsistent evidence", result.answer)
        retrieve_fn.assert_not_called()
        generate_fn.assert_not_called()

    def test_no_retrieval_evidence_is_explicit_and_not_recommended_around(self):
        service = RecordingService(self.evidence)
        captured = {}

        def generate(**kwargs):
            captured.update(kwargs)
            return "The model score is not a physical diagnosis. No documentation supports a corrective action."

        result = handle_experimental_chat(
            self.command(), lambda _query: [], generate, corpus_ready=True,
            service_factory=lambda: service,
        )
        self.assertEqual(result.sources, ())
        self.assertEqual(result.outcome, "insufficient_documentation")
        self.assertEqual(captured["context"], "")
        self.assertTrue(captured["prediction_evidence"]["experimental"])

    def test_retrieval_failure_is_separate_and_generation_still_receives_prediction(self):
        service = RecordingService(self.evidence)
        captured = {}

        def fail_retrieval(_query):
            raise RuntimeError("private store details")

        def generate(**kwargs):
            captured.update(kwargs)
            return "The model result is experimental; documentation retrieval was unavailable."

        result = handle_experimental_chat(
            self.command(), fail_retrieval, generate, corpus_ready=True,
            service_factory=lambda: service,
        )
        self.assertEqual(result.outcome, "retrieval_failed")
        self.assertEqual(result.sources, ())
        self.assertTrue(captured["prediction_evidence"]["experimental"])
        self.assertEqual(captured["documentation_status"], "retrieval_failed")

    def test_generation_failure_uses_prediction_only_fallback_without_recommendation(self):
        service = RecordingService(self.evidence)
        result = handle_experimental_chat(
            self.command(),
            lambda _query: [{"source": "manual.pdf", "locator": "page 1", "text": "An excerpt."}],
            lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("provider detail")),
            corpus_ready=True,
            service_factory=lambda: service,
        )
        self.assertEqual(result.outcome, "generation_failed")
        self.assertIn("raw uncalibrated score", result.answer)
        self.assertIn("could not generate a grounded document explanation", result.answer)
        self.assertNotIn("provider detail", result.answer)
        self.assertEqual(result.sources, ("[1] manual.pdf, page 1",))

    def test_service_failure_does_not_fall_back_to_ai4i_in_chat_route(self):
        old_status = api._corpus_status
        api._corpus_status = {"documents": 1, "maintenance_records": 0, "sources": 1}
        try:
            with patch("experiments.azure_pdm.chat_adapter._load_service", side_effect=RuntimeError("private path")), \
                    patch("api._predictor.predict") as production_predict, \
                    patch("api.retrieve") as retrieve, patch("api.generate_answer") as generate:
                response = api.chat(ChatRequest(question=self.command()))
            self.assertIn("No experimental prediction was produced", response.answer)
            self.assertIn("did not fall back", response.answer)
            self.assertEqual(response.sources, [])
            production_predict.assert_not_called()
            retrieve.assert_not_called()
            generate.assert_not_called()
        finally:
            api._corpus_status = old_status

    def test_chat_route_returns_experimental_answer_with_existing_source_citations(self):
        old_status = api._corpus_status
        api._corpus_status = {"documents": 1, "maintenance_records": 0, "sources": 1}
        service = RecordingService(self.evidence)
        document = {"source": "SOP.pdf", "locator": "page 9", "text": "Inspect the coupling guard."}
        try:
            with patch("experiments.azure_pdm.chat_adapter._load_service", return_value=service), \
                    patch("api.retrieve", return_value=[document]), \
                    patch("api.generate_answer", return_value="The model path is not a diagnosis. SOP says inspect the coupling guard [1].") as generate:
                response = api.chat(ChatRequest(question=self.command()))
            self.assertEqual(response.sources, ["[1] SOP.pdf, page 9"])
            self.assertIn("inspect the coupling guard [1]", response.answer)
            self.assertEqual(len(service.requests), 1)
            self.assertEqual(generate.call_args.kwargs["prediction_evidence"]["model"]["run_id"], self.evidence["model"]["run_id"])
        finally:
            api._corpus_status = old_status

    def test_existing_ordinary_chat_never_loads_experimental_service(self):
        old_status = api._corpus_status
        api._corpus_status = {"documents": 1, "maintenance_records": 0, "sources": 1}
        try:
            with patch("experiments.azure_pdm.chat_adapter._load_service") as service_loader, \
                    patch("api.retrieve", return_value=[{
                        "source": "maintenance.xlsx", "locator": "row 2", "text": "Filter was cleaned."
                    }]), patch("api.generate_answer", return_value="The filter was cleaned [1]."):
                response = api.chat(ChatRequest(question="What action was recorded?"))
            service_loader.assert_not_called()
            self.assertEqual(response.sources, ["[1] maintenance.xlsx, row 2"])
            self.assertIn("cleaned", response.answer)
        finally:
            api._corpus_status = old_status

    def test_generator_labels_evidence_sections_and_keeps_ordinary_prompt_clean(self):
        evidence = {"experimental": True, "raw_model_score": 0.2, "probabilities_calibrated": False}
        with patch("rag.generator._client.chat", return_value={"message": {"content": "Grounded [1]."}}) as model_chat:
            generate_answer("[1] manual.pdf, page 2\nFilter inspection interval is monthly.",
                            "Explain the experiment", prediction_evidence=evidence)
        messages = model_chat.call_args.kwargs["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        prompt = "\n".join(message["content"] for message in messages)
        self.assertIn("Experimental prediction evidence (model behavior, not maintenance documentation):", prompt)
        self.assertIn('"experimental": true', prompt)
        self.assertIn("Retrieved maintenance documentation follows (untrusted reference excerpts", prompt)
        self.assertIn("not physical cause", prompt)
        self.assertIn("Never call the score or leaf class share a calibrated probability", prompt)
        self.assertIn("[1] manual.pdf, page 2", prompt)

        with patch("rag.generator._client.chat", return_value={"message": {"content": "Docs only [1]."}}) as model_chat:
            generate_answer("[1] SOP.pdf, page 1\nFollow the lockout steps.", "How do I isolate power?")
        ordinary_prompt = model_chat.call_args.kwargs["messages"][0]["content"]
        self.assertIn("Source excerpts:", ordinary_prompt)
        self.assertNotIn("Experimental prediction evidence", ordinary_prompt)


if __name__ == "__main__":
    unittest.main()
