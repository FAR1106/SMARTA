import json
import unittest
from unittest.mock import patch

import api
from api import ChatRequest
from httpx import ConnectError, ReadTimeout
from rag.generator import (
    CHAT_TOKEN_BUDGETS,
    MAX_CONTEXT_CHARS,
    MAX_GENERATED_TOKENS,
    MODEL_KEEP_ALIVE,
    generate_answer,
    GenerationFailure,
)


class ChatPerformanceTests(unittest.TestCase):
    def test_generation_transport_errors_are_categorized_without_fabricated_answer(self):
        for error, expected in (
            (ConnectError("connection refused"), "language_model_unavailable"),
            (ConnectionError("Ollama connection refused"), "language_model_unavailable"),
            (ReadTimeout("slow model"), "generation_timeout"),
        ):
            with self.subTest(expected=expected), patch(
                "rag.generator._client.chat", side_effect=error
            ):
                with self.assertRaises(GenerationFailure) as raised:
                    generate_answer("[1] manual.pdf, page 2\nInspect bearings.", "Question?")
                self.assertEqual(raised.exception.code, expected)

    def test_malformed_generation_payload_is_reported_as_invalid_response(self):
        with patch("rag.generator._client.chat", return_value={"message": {}}):
            with self.assertRaises(GenerationFailure) as raised:
                generate_answer("[1] manual.pdf, page 2\nInspect bearings.", "Question?")
        self.assertEqual(raised.exception.code, "invalid_response")

    def test_chat_maps_unavailable_model_to_safe_structured_error(self):
        original_status = api._corpus_status
        api._corpus_status = {"documents": 1, "maintenance_records": 0, "sources": 1}
        try:
            with patch("api.retrieve", return_value=[{
                "source": "manual.pdf", "locator": "page 2", "text": "Inspect bearings."
            }]), patch("api.generate_answer", side_effect=GenerationFailure(
                "language_model_unavailable", "SMARTA's language model is unavailable. Start Ollama and retry."
            )):
                with self.assertRaises(api.HTTPException) as raised:
                    api.chat(ChatRequest(question="What checks can prevent compressor faults?"))
            self.assertEqual(raised.exception.status_code, 503)
            self.assertEqual(raised.exception.detail["code"], "language_model_unavailable")
            self.assertIn("Start Ollama", raised.exception.detail["message"])
        finally:
            api._corpus_status = original_status

    def test_chat_distinguishes_retrieval_failure_from_empty_evidence(self):
        original_status = api._corpus_status
        api._corpus_status = {"documents": 1, "maintenance_records": 0, "sources": 1}
        try:
            with patch("api.retrieve", side_effect=RuntimeError("storage unavailable")):
                with self.assertRaises(api.HTTPException) as raised:
                    api.chat(ChatRequest(question="What checks can prevent compressor faults?"))
            self.assertEqual(raised.exception.status_code, 503)
            self.assertEqual(raised.exception.detail["code"], "retrieval_failed")
        finally:
            api._corpus_status = original_status

    def test_chat_reports_empty_retrieval_as_no_relevant_evidence(self):
        original_status = api._corpus_status
        api._corpus_status = {"documents": 1, "maintenance_records": 0, "sources": 1}
        try:
            with patch("api.retrieve", return_value=[]):
                result = api.chat(ChatRequest(question="What is this undocumented issue?"))
            self.assertEqual(result.evidence_status, "no_relevant_evidence")
            self.assertEqual(result.sources, [])
            self.assertIn("couldn't find a matching record", result.answer)
        finally:
            api._corpus_status = original_status

    def test_chat_maps_generation_timeout_to_retryable_gateway_timeout(self):
        original_status = api._corpus_status
        api._corpus_status = {"documents": 1, "maintenance_records": 0, "sources": 1}
        try:
            with patch("api.retrieve", return_value=[{
                "source": "manual.pdf", "locator": "page 2", "text": "Inspect bearings."
            }]), patch("api.generate_answer", side_effect=GenerationFailure(
                "generation_timeout", "SMARTA's language model took too long to respond. Please retry."
            )):
                with self.assertRaises(api.HTTPException) as raised:
                    api.chat(ChatRequest(question="Question?"))
            self.assertEqual(raised.exception.status_code, 504)
            self.assertEqual(raised.exception.detail["code"], "generation_timeout")
        finally:
            api._corpus_status = original_status

    def test_generator_bounds_context_and_output_and_records_phase_times(self):
        timings = {}
        with patch(
            "rag.generator._client.chat",
            return_value={"message": {"content": "Grounded answer."}},
        ) as model_chat:
            answer = generate_answer("e" * (MAX_CONTEXT_CHARS + 50), "question", timings)

        self.assertEqual(answer, "Grounded answer.")
        model_chat.assert_called_once()
        request = model_chat.call_args.kwargs
        self.assertEqual(request["options"], {"num_predict": MAX_GENERATED_TOKENS})
        self.assertEqual(request["keep_alive"], MODEL_KEEP_ALIVE)
        self.assertLessEqual(len(request["messages"][0]["content"]), MAX_CONTEXT_CHARS + 1500)
        self.assertGreaterEqual(timings["prompt_preparation_ms"], 0)
        self.assertGreaterEqual(timings["model_generation_ms"], 0)

    def test_prediction_simple_explanation_uses_validated_payload_without_llm(self):
        context = {
            "machine_id": "Machine-07",
            "model_version": "ai4i-rf-v1",
            "machine_failure": {"classification": 0, "score": 0.0},
        }
        timings = {}
        with patch("rag.generator._client.chat") as model_chat:
            answer = generate_answer(
                "", "Explain this prediction in simple words", timings=timings,
                production_prediction_evidence=context,
                documentation_status="no_matches",
            )
        model_chat.assert_not_called()
        self.assertIn("did not flag a failure", answer)
        self.assertEqual(timings["model_generation_ms"], 0.0)
        self.assertEqual(timings["response_guard_ms"], 0.0)
        self.assertEqual(timings["answer_chars"], len(answer))

    def test_token_budgets_are_intent_specific_and_detailed_answers_can_expand(self):
        from rag.generator import _generation_budget

        self.assertEqual(_generation_budget("What is this?"), CHAT_TOKEN_BUDGETS["default"])
        self.assertEqual(
            _generation_budget("What maintenance action is supported?"),
            CHAT_TOKEN_BUDGETS["maintenance"],
        )
        self.assertEqual(
            _generation_budget("Explain in detail what is supported"),
            CHAT_TOKEN_BUDGETS["detailed"],
        )

    def test_prompt_uses_top_complete_passages_without_changing_source_ids(self):
        context = "\n\n".join(
            f"[{i}] manual.pdf, page {i}\nFull passage text {i}." for i in range(1, 7)
        )
        with patch("rag.generator._client.chat", return_value={
            "message": {"content": "The source supports the check [1]."}
        }) as model_chat:
            answer = generate_answer(context, "What does the manual say?")

        self.assertEqual(answer, "The source supports the check [1].")
        prompt = model_chat.call_args.kwargs["messages"][0]["content"]
        self.assertIn("[1] manual.pdf, page 1\nFull passage text 1.", prompt)
        self.assertIn("[5] manual.pdf, page 5\nFull passage text 5.", prompt)
        self.assertNotIn("[6] manual.pdf", prompt)

    def test_invalid_ocr_page_reference_is_replaced_with_passage_grounded_response(self):
        context = (
            "[1] sop.pdf, page 243\nCondition 8. Compressor over· heating. Possible faults include "
            "insufficient com· pressor cooling and oil cooler clogged externally. The suggested remedy is truncated.\n\n"
            "[2] sop.pdf, page 169\nOverheatins of Ammonia Compressor as a result of dirty oil cooler, "
            "requiring cleaning of carbonized oil."
        )
        timings = {}
        with patch("rag.generator._client.chat", return_value={
            "message": {"content": "Check cooling and oil cooler condition [241]."}
        }):
            answer = generate_answer(
                context,
                "What maintenance checks are supported for an overheating compressor?",
                timings=timings,
            )

        self.assertIn("historical example", answer)
        self.assertIn("[1]", answer)
        self.assertIn("[2]", answer)
        self.assertNotIn("[241]", answer)
        self.assertIn("does not establish a machine-specific repair", answer)
        self.assertGreaterEqual(timings["response_guard_ms"], 0)

    def test_unknown_citation_id_never_reaches_ordinary_chat_response(self):
        context = "[1] manual.pdf, page 4\nThe manual describes the alarm indicator."
        with patch("rag.generator._client.chat", return_value={
            "message": {"content": "The manual says replace the control board [99]."}
        }):
            answer = generate_answer(context, "What should I replace?")
        self.assertIn("does not match the retrieved source labels", answer)
        self.assertNotIn("replace the control board", answer)

    def test_unknown_identifier_uses_grounded_model_path_and_logs_safe_timings(self):
        original_status = api._corpus_status
        api._corpus_status = {"documents": 1, "maintenance_records": 0, "sources": 1}
        records = [{
            "source": "manual.pdf",
            "locator": "page 4",
            "text": "Compressor thermal overload protection should be inspected per the manual.",
        }]

        def fake_generation(context, question, timings):
            timings["prompt_preparation_ms"] = 0.25
            timings["model_generation_ms"] = 1.5
            self.assertIn("No retrieved record contains the exact identifier A12", context)
            return "The sources do not identify alarm A12."

        try:
            with patch("api.retrieve", return_value=records), patch(
                "api.generate_answer", side_effect=fake_generation
            ), self.assertLogs("smarta.chat", level="INFO") as logs:
                result = api.chat(ChatRequest(question="Why is compressor alarm A12 active?"))
            self.assertEqual(result.answer, "The sources do not identify alarm A12.")
            self.assertEqual(result.sources, ["[1] manual.pdf, page 4"])
            log_line = next(line for line in logs.output if "chat_request_timing" in line)
            payload = json.loads(log_line.split("chat_request_timing ", 1)[1])
            self.assertIn("retrieval_ms", payload)
            self.assertIn("context_assembly_ms", payload)
            self.assertEqual(payload["prompt_preparation_ms"], 0.25)
            self.assertEqual(payload["model_generation_ms"], 1.5)
            self.assertIn("total_server_ms", payload)
            self.assertNotIn("A12", log_line)
            self.assertNotIn("thermal overload", log_line)
        finally:
            api._corpus_status = original_status


if __name__ == "__main__":
    unittest.main()
