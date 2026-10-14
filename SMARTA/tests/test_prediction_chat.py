from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

import api
from api import ChatRequest, PredictRequest, chat, predict


def prediction_context():
    request = PredictRequest(machine_id="Machine-42", features={
        "product_type": "H", "air_temperature_k": 304.5,
        "process_temperature_k": 313.8, "rotational_speed_rpm": 1168,
        "torque_nm": 76.6, "tool_wear_min": 253,
    })
    with patch("api.retrieve", return_value=[]):
        response = predict(request)
    result = response.prediction
    return {
        "provenance": "production_ai4i",
        "experimental": False,
        "model_id": result["model_version"],
        "machine_id": result["machine_id"],
        "prediction_timestamp": result["prediction_timestamp"],
        "model_version": result["model_version"],
        "machine_failure": result["machine_failure"],
        "failure_modes": result["failure_modes"],
        "failure_mode_labels": result["failure_mode_labels"],
        "unsupported_failure_modes": result["unsupported_failure_modes"],
        "input_features": result["input_features"],
        "out_of_training_range": result["out_of_training_range"],
        "limitations": result["limitations"],
    }


class PredictionChatTests(unittest.TestCase):
    def setUp(self):
        self.old_status = api._corpus_status
        api._corpus_status = {"documents": 2, "maintenance_records": 1, "sources": 1}

    def tearDown(self):
        api._corpus_status = self.old_status

    def test_real_ai4i_result_is_validated_and_passed_separately_from_rag(self):
        evidence = {"source": "manual.pdf", "locator": "page 4", "text": "Inspect the coupling."}
        request_context = prediction_context()
        captured = {}

        def generate(**kwargs):
            captured.update(kwargs)
            machine = request_context["machine_failure"]
            status = "failure classified" if machine["classification"] else "no failure classified"
            return (
                f"For machine {request_context['machine_id']}, the classifier returned "
                f"{status} (class {machine['classification']}) with raw uncalibrated "
                f"score {machine['score']}. The manual supports inspecting the coupling [1]."
            )

        with patch("api.retrieve", return_value=[evidence]) as retrieve, patch(
            "api.generate_answer", side_effect=generate
        ):
            response = chat(ChatRequest(
                question="Explain this result and what maintenance guidance is supported?",
                prediction_context=request_context,
            ))

        self.assertEqual(response.sources, ["[1] manual.pdf, page 4"])
        self.assertIn("[1]", response.answer)
        self.assertEqual(captured["production_prediction_evidence"]["machine_id"], "Machine-42")
        self.assertEqual(captured["production_prediction_evidence"]["input_features"], request_context["input_features"])
        self.assertEqual(captured["production_prediction_evidence"]["machine_failure"], request_context["machine_failure"])
        self.assertNotIn("threshold", captured["production_prediction_evidence"])
        self.assertEqual(captured["documentation_status"], "available")
        self.assertIn("Machine-42", retrieve.call_args.args[0])

    def test_empty_or_failed_retrieval_is_explicit_and_never_recommends(self):
        context = prediction_context()
        for retrieve_config, expected_status in [
            ({"return_value": []}, "no_matches"),
            ({"side_effect": RuntimeError("private storage detail")}, "retrieval_failed"),
        ]:
            with self.subTest(expected_status=expected_status):
                captured = {}

                def generate(**kwargs):
                    captured.update(kwargs)
                    return "The classifier result is shown; no maintenance action is supported by documents."

                with patch("api.retrieve", **retrieve_config), patch(
                    "api.generate_answer", side_effect=generate
                ):
                    result = chat(ChatRequest(question="Explain this result", prediction_context=context))
                self.assertEqual(captured["documentation_status"], expected_status)
                self.assertEqual(result.sources, [])
                self.assertNotIn("private storage detail", result.answer)

    def test_malformed_and_contradictory_production_context_is_rejected(self):
        invalid = copy.deepcopy(prediction_context())
        invalid["model_version"] = "azure-pdm-experimental"
        with self.assertRaises(Exception):
            ChatRequest(question="Explain this result", prediction_context=invalid)

        invalid = copy.deepcopy(prediction_context())
        invalid["experimental"] = True
        with self.assertRaises(Exception):
            ChatRequest(question="Explain this result", prediction_context=invalid)

        invalid = copy.deepcopy(prediction_context())
        invalid["input_features"].pop("Torque [Nm]")
        with self.assertRaises(Exception):
            ChatRequest(question="Explain this result", prediction_context=invalid)

        invalid = copy.deepcopy(prediction_context())
        invalid["threshold"] = 0.25
        with self.assertRaises(Exception):
            ChatRequest(question="Explain this result", prediction_context=invalid)

        invalid = copy.deepcopy(prediction_context())
        invalid["failure_mode_labels"] = ["over-supply failure"]
        with self.assertRaises(Exception):
            ChatRequest(question="Explain this result", prediction_context=invalid)

    def test_experimental_command_rejects_production_context(self):
        from fastapi import HTTPException

        with self.assertRaises(HTTPException) as error:
            chat(ChatRequest(
                question="/experimental-pdm {}",
                prediction_context=prediction_context(),
            ))
        self.assertEqual(error.exception.status_code, 422)

    def test_generation_failure_is_sanitized_and_not_replaced_with_a_claim(self):
        from fastapi import HTTPException

        with patch("api.retrieve", return_value=[{
            "source": "manual.pdf", "locator": "page 4", "text": "Inspect the coupling."
        }]), patch("api.generate_answer", side_effect=RuntimeError("private provider detail")):
            with self.assertRaises(HTTPException) as error:
                chat(ChatRequest(
                    question="Explain this result",
                    prediction_context=prediction_context(),
                ))
        self.assertEqual(error.exception.status_code, 502)
        self.assertEqual(error.exception.detail["code"], "generation_failed")
        self.assertNotIn("private provider detail", str(error.exception.detail))

    def test_contextless_documentation_chat_remains_ordinary(self):
        with patch("api.retrieve", return_value=[{
            "source": "maintenance.xlsx", "locator": "row 8", "text": "Filter cleaned."
        }]), patch("api.generate_answer", return_value="The filter was cleaned [1].") as generate:
            result = chat(ChatRequest(question="What action was recorded?"))
        self.assertEqual(result.sources, ["[1] maintenance.xlsx, row 8"])
        self.assertNotIn("production_prediction_evidence", generate.call_args.kwargs)

    def test_generator_separates_prediction_from_retrieved_source_material(self):
        from rag.generator import generate_answer

        context = prediction_context()
        with patch("rag.generator._client.chat", return_value={
            "message": {"content": "The model classified this input. The manual supports a check [1]."}
        }) as model_chat:
            generate_answer(
                "[1] manual.pdf, page 4\nInspect the coupling.",
                "Explain this result",
                production_prediction_evidence=context,
                documentation_status="available",
            )
        prompt = "\n".join(item["content"] for item in model_chat.call_args.kwargs["messages"])
        self.assertIn("Production AI4I prediction evidence", prompt)
        self.assertIn("Retrieved maintenance documentation follows", prompt)
        self.assertIn('"score_type": "uncalibrated_random_forest_score"', prompt)
        self.assertIn("Do not call them probabilities", prompt)
        self.assertIn("[1] manual.pdf, page 4", prompt)
        self.assertNotIn('"threshold":', prompt)

    def test_generator_rejects_llm_relabeling_uncalibrated_score_as_probability(self):
        from rag.generator import generate_answer

        context = prediction_context()
        with patch("rag.generator._client.chat", return_value={
            "message": {
                "content": "The score suggests a 70% probability of failure for this machine."
            }
        }):
            answer = generate_answer(
                "", "Explain this result",
                production_prediction_evidence=context,
                documentation_status="no_matches",
            )
        self.assertIn("raw uncalibrated score", answer)
        self.assertIn("not a calibrated probability", answer)
        self.assertIn("no corrective action", answer)
        self.assertNotIn("70% probability", answer)

    def test_generator_does_not_let_rag_ignore_prediction_evidence(self):
        from rag.generator import generate_answer

        context = prediction_context()
        with patch("rag.generator._client.chat", return_value={
            "message": {
                "content": "The excerpts do not contain information about predictions."
            }
        }):
            answer = generate_answer(
                "[1] manual.pdf, page 4\nInspect the coupling.",
                "Explain this machine's prediction in simple terms",
                production_prediction_evidence=context,
                documentation_status="available",
            )
        self.assertIn("flagged a failure", answer)
        self.assertIn("Machine-42", answer)
        self.assertTrue(2 <= len([s for s in answer.split(".") if s.strip()]) <= 4)
        self.assertNotIn("Failure-mode outputs:", answer)
        self.assertIn("synthetic data", answer)
        self.assertNotIn("uncalibrated score", answer)
        self.assertNotIn("The excerpts do not contain", answer)

    def test_simple_explanation_is_plain_short_and_question_specific(self):
        from rag.generator import (
            _prediction_question_intent,
            _production_answer_is_supported,
            _safe_production_prediction_fallback,
        )

        context = prediction_context()
        context["machine_failure"] = {"classification": 0, "score": 0.0}
        question = "Explain this prediction in simple words"
        self.assertEqual(_prediction_question_intent(question), "simple_explanation")
        answer = _safe_production_prediction_fallback(context, "no_matches", question)
        self.assertTrue(2 <= len([s for s in answer.split(".") if s.strip()]) <= 4)
        self.assertIn("did not flag a failure", answer)
        self.assertIn("not proof", answer)
        self.assertIn("synthetic data", answer)
        self.assertNotIn("uncalibrated", answer)
        self.assertNotIn("Tool wear", answer)
        self.assertTrue(_production_answer_is_supported(answer, context, "", "no_matches", question))

    def test_simple_guard_accepts_valid_plain_language_without_fixed_template(self):
        from rag.generator import _production_answer_is_supported

        context = prediction_context()
        context["machine_failure"] = {"classification": 0, "score": 0.0}
        answer = (
            "Machine-42's readings did not flag a failure. That does not prove the machine is healthy. "
            "AI4I is based on synthetic training data."
        )
        self.assertTrue(_production_answer_is_supported(
            answer, context, "", "no_matches", "Explain this prediction in simple words"
        ))

    def test_operating_condition_question_gets_plain_language_structured_answer(self):
        from rag.generator import _safe_production_prediction_fallback, generate_answer

        context = prediction_context()
        context["input_features"]["Air temperature [K]"] = 310.0
        context["out_of_training_range"] = {
            "Air temperature [K]": {
                "value": 310.0,
                "training_range": [295.3, 304.5],
            }
        }
        expected = _safe_production_prediction_fallback(
            context, "available", "Which operating conditions are relevant to this result?"
        )
        for section in (
            "### What the model concluded\n\n",
            "### Relevant operating inputs\n\n",
            "### What the model output does and does not establish\n\n",
            "### Maintenance evidence\n\n",
        ):
            self.assertIn(section, expected)
        self.assertIn("Air temperature [K]: 310.0", expected)
        self.assertIn("outside the AI4I training range [295.3, 304.5]", expected)
        self.assertIn("provides no feature attribution", expected)
        self.assertNotIn("raw uncalibrated score", expected)
        self.assertEqual(expected.lower().count("synthetic ai4i"), 1)
        self.assertNotIn("probability of", expected.lower())
        self.assertNotIn("Failure-mode outputs:", expected)

        with patch("rag.generator._client.chat", return_value={
            "message": {"content": "Model result; values were used."}
        }) as model_chat:
            answer = generate_answer(
                "[1] maintenance.pdf, row 4\nThe belt was inspected.",
                "Which operating conditions are relevant to this result?",
                production_prediction_evidence=context,
                documentation_status="available",
            )
        model_chat.assert_not_called()
        self.assertEqual(answer, expected)
        self.assertIn("### Relevant operating inputs", answer)

    def test_final_conclusion_questions_get_a_concise_conclusion(self):
        from rag.generator import (
            _prediction_question_intent,
            _production_answer_is_supported,
            _safe_production_prediction_fallback,
            generate_answer,
        )

        context = prediction_context()
        questions = (
            "What is the final conclusion?",
            "So what does this mean?",
            "Give me the bottom line.",
            "Tell me the conclusion of the final prediction",
        )
        for question in questions:
            with self.subTest(question=question):
                self.assertEqual(_prediction_question_intent(question), "final_conclusion")
                answer = _safe_production_prediction_fallback(
                    context, "no_matches", question
                )
                self.assertTrue(answer.startswith("### Conclusion\n\n"))
                self.assertIn("Machine-42", answer)
                self.assertIn("flagged a failure", answer)
                self.assertIn("does not guarantee real-world safety", answer)
                self.assertIn("predict whether the machine will fail", answer)
                self.assertNotIn("uncalibrated", answer)
                self.assertNotIn("input_features", answer)
                self.assertNotIn("TWF", answer)
                self.assertNotIn("probability of", answer.lower())
                self.assertNotIn("What to do next", answer)
                self.assertTrue(_production_answer_is_supported(
                    answer, context, "", "no_matches", question
                ))

        with patch("rag.generator._client.chat") as model_chat:
            answer = generate_answer(
                "", "What is the final conclusion?",
                production_prediction_evidence=context,
                documentation_status="no_matches",
            )
        model_chat.assert_not_called()
        self.assertTrue(answer.startswith("### Conclusion\n\n"))

    def test_final_conclusion_is_concise_and_distinct_from_simple_explanation(self):
        from rag.generator import _safe_production_prediction_fallback

        context = prediction_context()
        context["machine_failure"] = {"classification": 0, "score": 0.0}
        simple = _safe_production_prediction_fallback(
            context, "no_matches", "Explain this prediction in simple words"
        )
        conclusion = _safe_production_prediction_fallback(
            context, "no_matches", "Tell me the conclusion of the final prediction"
        )
        self.assertNotEqual(simple, conclusion)
        self.assertLessEqual(len(conclusion.split()), 45)
        self.assertIn("No failure was detected by the classifier", conclusion)
        self.assertIn("does not guarantee real-world safety", conclusion)
        self.assertIn("predict whether the machine will fail", conclusion)
        self.assertNotIn("synthetic data", conclusion)

    def test_inference_and_missing_evidence_fallbacks_answer_distinct_intents(self):
        from rag.generator import _safe_production_prediction_fallback

        context = prediction_context()
        retrieved = (
            "[1] troubleshooting.pdf, page 4\n"
            "General instructions for a different equipment family; no machine identifier is given."
        )
        inference = _safe_production_prediction_fallback(
            context, "available", "What can I infer from this prediction?", retrieved
        )
        missing = _safe_production_prediction_fallback(
            context,
            "available",
            "What information is missing before a corrective action can be recommended?",
            retrieved,
        )
        self.assertNotEqual(inference, missing)
        self.assertIn("### What you can infer\n\n", inference)
        self.assertIn("not a calibrated failure probability", inference)
        self.assertIn("synthetic AI4I benchmark", inference)
        self.assertIn("### What is missing before recommending an action\n\n", missing)
        self.assertIn("\n\n- Verified current readings", missing)
        self.assertIn("timestamped trends", missing)
        self.assertIn("Confirmed symptoms", missing)
        self.assertIn("Machine-specific maintenance", missing)
        self.assertIn("AI4I product-type category", missing)
        self.assertNotIn("no relevant maintenance passages were retrieved", missing)
        self.assertIn("No repair is recommended", missing)

    def test_response_guard_rejects_wrong_score_but_accepts_non_advisory_gaps(self):
        from rag.generator import _production_answer_is_supported

        context = prediction_context()
        answer = (
            "What is missing before recommending an action? For machine Machine-42, "
            f"class 1 has raw uncalibrated score {context['machine_failure']['score']}. No current trends "
            "or confirmed symptoms are provided, so no corrective action is supported."
        )
        self.assertTrue(_production_answer_is_supported(
            answer, context, "", "no_matches",
            "What information is missing before a corrective action can be recommended?",
        ))
        fabricated = answer.replace(str(context['machine_failure']['score']), "0.91")
        self.assertFalse(_production_answer_is_supported(
            fabricated, context, "", "no_matches",
            "What information is missing before a corrective action can be recommended?",
        ))

    def test_action_question_without_evidence_never_gets_recommendation(self):
        from rag.generator import generate_answer

        context = prediction_context()
        with patch("rag.generator._client.chat", return_value={
            "message": {"content": "For Machine-42, class 1 with raw uncalibrated score 0.78. Inspect the coupling."}
        }):
            answer = generate_answer(
                "", "What corrective action should I take?",
                production_prediction_evidence=context,
                documentation_status="no_matches",
            )
        self.assertIn("no corrective action is supported", answer.lower())
        self.assertNotIn("Inspect the coupling", answer)

    def test_maintenance_action_answer_uses_and_cites_retrieved_passage_text(self):
        from rag.generator import _production_answer_is_supported, generate_answer

        context = prediction_context()
        passage = (
            "Several measurements with proper data recording are required to establish "
            "which conditions are normal. Determine the monitoring point and interval "
            "between periodic checks from experience or the manufacturer's instructions."
        )
        retrieved = f"[1] sop.pdf, page 67\n{passage}"
        classifier_result = (
            "failure classified" if context["machine_failure"]["classification"]
            else "no failure detected by the classifier"
        )
        answer = (
            "### Classifier output\n\nFor machine Machine-42, the AI4I classifier returned "
            f"{classifier_result} (class {context['machine_failure']['classification']}) "
            "with raw uncalibrated score "
            f"{context['machine_failure']['score']}. This is not a diagnosis.\n\n"
            "### Maintenance evidence\n\nAs general monitoring guidance, establish "
            "normal machine conditions using several measurements with proper data "
            "recording, and set periodic-check intervals using experience or the "
            "manufacturer's instructions [1]. This does not identify a fault or justify "
            "a Machine-42 repair."
        )
        self.assertTrue(_production_answer_is_supported(
            answer, context, retrieved, "available",
            "Based on SMARTA's maintenance records, what action is supported? Cite the relevant source.",
        ))
        unqualified = answer.replace(
            "As general monitoring guidance,", "The machine must"
        ).replace(
            "This does not identify a fault or justify a Machine-42 repair.", ""
        )
        self.assertFalse(_production_answer_is_supported(
            unqualified, context, retrieved, "available",
            "Based on SMARTA's maintenance records, what action is supported? Cite the relevant source.",
        ))

        with patch("rag.generator._client.chat", return_value={
            "message": {"content": answer}
        }) as model_chat:
            generated = generate_answer(
                retrieved,
                "Based on SMARTA's maintenance records, what action is supported? Cite the relevant source.",
                production_prediction_evidence=context,
                documentation_status="available",
            )
        model_chat.assert_not_called()
        self.assertIn("several properly recorded measurements", generated)
        self.assertIn("[1]", generated)
        self.assertIn("general maintenance guidance", generated)

        with patch("rag.generator._client.chat", return_value={
            "message": {"content": "The classifier has no failure, so review the sources."}
        }):
            grounded_fallback = generate_answer(
                retrieved,
                "Based on SMARTA's maintenance records, what action is supported? Cite the relevant source.",
                production_prediction_evidence=context,
                documentation_status="available",
            )
        self.assertIn("several properly recorded measurements", grounded_fallback)
        self.assertIn("[1]", grounded_fallback)
        self.assertIn("does not identify a fault", grounded_fallback)

    def test_troubleshooting_returns_cited_checks_without_claiming_diagnosis_or_repair(self):
        from rag.generator import (
            _prediction_question_intent, _production_answer_is_supported,
            _troubleshooting_fallback, generate_answer,
        )

        context = prediction_context()
        question = "How do I fix this failure?"
        self.assertEqual(_prediction_question_intent(question), "troubleshooting")
        retrieved = (
            "[1] sop.pdf, page 19\nEarly fault signs may be small variations. Physical senses can detect heat, cracks, vibration, odor, and strange noises.\n\n"
            "[2] sop.pdf, page 48\nAfter inspection, review equipment history and investigate the fault using appropriate instruments for accurate analysis.\n\n"
            "[3] sop.pdf, page 67\nSeveral measurements with proper data recording are required to establish normal conditions. Determine monitoring point and interval from experience or manufacturer instructions.\n\n"
            "[4] maintenance_history.xlsx, Sheet1, row 21\nMachine: SMT20. Startup failure. Recorded action: Replace Power Module."
        )
        answer = _troubleshooting_fallback(context, retrieved, "available")
        self.assertIn("### Prediction summary", answer)
        self.assertIn("### Possible causes", answer)
        self.assertIn("### Recommended diagnostic checks", answer)
        self.assertIn("### Corrective action", answer)
        self.assertIn("### Missing information", answer)
        self.assertIn("### Sources", answer)
        self.assertIn("observe", answer)
        self.assertIn("[1]", answer)
        self.assertIn("review inspection findings", answer.lower())
        self.assertIn("[2]", answer)
        self.assertIn("several properly recorded measurements", answer)
        self.assertIn("[3]", answer)
        self.assertIn("SMT20", answer)
        self.assertIn("different asset", answer)
        self.assertNotIn("Replace Power Module", answer)
        self.assertIn("No machine-specific repair is supported", answer)
        self.assertTrue(_production_answer_is_supported(
            answer, context, retrieved, "available", question
        ))

        with patch("rag.generator._client.chat", return_value={
            "message": {"content": "The classifier says class 1. Replace the module immediately [4]."}
        }) as model_chat:
            generated = generate_answer(
                retrieved, question, production_prediction_evidence=context,
                documentation_status="available",
            )
        model_chat.assert_called_once()
        self.assertIn("observe", generated)
        self.assertNotIn("Replace the module immediately", generated)
        self.assertIn("[3]", generated)

    def test_troubleshooting_general_manual_is_guidance_not_machine_specific_advice(self):
        from rag.generator import _troubleshooting_fallback

        answer = _troubleshooting_fallback(
            prediction_context(),
            "[1] manual.pdf, page 11\nEarly fault signs may include vibration. Operators should inspect for abnormal vibration and record temperature readings periodically.",
            "available",
        )
        self.assertIn("general guidance", answer)
        self.assertIn("No machine-specific repair is supported", answer)
        self.assertIn("[1]", answer)
        self.assertNotIn("is the cause", answer)

    def test_troubleshooting_without_evidence_asks_one_focused_question(self):
        from rag.generator import _troubleshooting_fallback

        answer = _troubleshooting_fallback(prediction_context(), "", "no_matches")
        self.assertIn("do not establish a cause", answer)
        self.assertIn("cannot support a diagnostic check or repair", answer)
        self.assertIn("what make/model, alarm or observed symptom", answer)
        self.assertIn("No repair is supported", answer)

    def test_troubleshooting_guard_rejects_fabricated_citation_cause_and_repair(self):
        from rag.generator import _production_answer_is_supported

        evidence = prediction_context()
        context = "[1] manual.pdf, page 2\nObserve for unusual vibration."
        good = (
            f"### Prediction summary\n\nFor Machine-42, class 1 raw uncalibrated score {evidence['machine_failure']['score']}.\n\n"
            "### Possible causes\n\nNo cause is established.\n\n"
            "### Recommended diagnostic checks\n\n1. Observe for unusual vibration [1].\n\n"
            "### Corrective action\n\nNo repair is supported.\n\n"
            "### Missing information\n\nWhat alarm is present?\n\n### Sources\n\n[1]"
        )
        question = "What should I check first?"
        self.assertTrue(_production_answer_is_supported(good, evidence, context, "available", question))
        self.assertFalse(_production_answer_is_supported(
            good.replace("[1].", "[9]."), evidence, context, "available", question
        ))
        self.assertFalse(_production_answer_is_supported(
            good.replace("No cause is established.", "The motor failed because of overheating [1]."),
            evidence, context, "available", question,
        ))
        self.assertFalse(_production_answer_is_supported(
            good.replace("No repair is supported.", "Replace the motor [1]."),
            evidence, context, "available", question,
        ))
        self.assertFalse(_production_answer_is_supported(
            good.replace("Observe for unusual vibration [1].", "Check pressure at 50 psi [1]."),
            evidence, context, "available", question,
        ))

    def test_prediction_troubleshooting_retrieval_fuses_targeted_results_with_metadata(self):
        from api import _retrieve_prediction_troubleshooting

        calls = []
        rows = [
            {"source": "sop.pdf", "locator": "page 19", "text": "signs", "metadata": {"page": 19}},
            {"source": "sop.pdf", "locator": "page 48", "text": "history", "metadata": {"page": 48}},
            {"source": "manual.pdf", "locator": "page 5", "text": "manual", "metadata": {"page": 5}},
        ]

        def retrieve_stub(query, k=6):
            calls.append(query)
            if len(calls) == 1:
                return [rows[0], rows[2]]
            if len(calls) == 2:
                return [rows[0], rows[1]]
            return [rows[1]]

        with patch("api.retrieve", side_effect=retrieve_stub):
            results = _retrieve_prediction_troubleshooting(
                "How do I fix this failure?", "How do I fix this failure? machine Machine-42 OSF"
            )
        self.assertEqual(len(calls), 3)
        self.assertEqual([(row["source"], row["locator"]) for row in results], [
            ("sop.pdf", "page 19"), ("sop.pdf", "page 48"), ("manual.pdf", "page 5")
        ])
        self.assertEqual(results[0]["metadata"], {"page": 19})

    def test_irrelevant_passages_and_source_labels_do_not_support_actions(self):
        from rag.generator import generate_answer

        context = prediction_context()
        question = "Based on SMARTA's maintenance records, what action is supported? Cite the relevant source."
        classifier_result = (
            "failure classified" if context["machine_failure"]["classification"]
            else "no failure detected by the classifier"
        )
        unsupported_answer = (
            f"For machine Machine-42, {classifier_result} "
            f"(class {context['machine_failure']['classification']}) with raw uncalibrated score "
            f"{context['machine_failure']['score']}. ### Maintenance evidence\n\n"
            "Replace the bearing and inspect the drive [1]."
        )
        for retrieved in (
            "[1] unrelated.pdf, page 7\nThe control-panel menu lists screen brightness settings.",
            "[1] sop.pdf, page 7",
        ):
            with self.subTest(retrieved=retrieved):
                with patch("rag.generator._client.chat", return_value={
                    "message": {"content": unsupported_answer}
                }):
                    answer = generate_answer(
                        retrieved, question,
                        production_prediction_evidence=context,
                        documentation_status="available",
                    )
                self.assertTrue(
                    "do not support a relevant machine-specific corrective action" in answer
                    or "No relevant maintenance passage was available" in answer
                )
                self.assertNotIn("Replace the bearing", answer)

    def test_generator_rejects_wrong_failure_mode_name_and_uncited_advice(self):
        from rag.generator import generate_answer

        context = prediction_context()
        unsafe = (
            "OSF (Over-Supply Failure) was classified.\n"
            "- Inspect the machine and replace its coupling. (manual.pdf, page 4)"
        )
        with patch("rag.generator._client.chat", return_value={
            "message": {"content": unsafe}
        }):
            answer = generate_answer(
                "[1] manual.pdf, page 4\nInspect the coupling.",
                "Explain the mode and next action",
                production_prediction_evidence=context,
                documentation_status="available",
            )
        self.assertIn("raw uncalibrated score", answer)
        self.assertNotIn("Over-Supply Failure", answer)
        self.assertNotIn("replace its coupling", answer)


if __name__ == "__main__":
    unittest.main()
