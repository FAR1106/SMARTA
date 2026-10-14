"""Explicit opt-in bridge from SMARTA chat to Azure PdM experimental evidence.

The module deliberately loads the model service only after parsing the explicit
``/experimental-pdm`` command. It is not an HTTP route and ordinary chat never
calls it.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Any, Callable, Mapping


COMMAND = "/experimental-pdm"
TARGET = "A failure-table event logged in (t, t+24h]; not confirmed physical failure onset."
SCORE_SEMANTICS = "uncalibrated classifier score; not a calibrated probability"
FEATURE_PIPELINE = "experiments.azure_pdm.pipeline.build_features"
EXPECTED_RUN_ID = "run_primary24h_20261013T171456_789146000Z"
EXPECTED_ARTIFACT_SHA256 = "560b9c60ed7367e941f38c60ebcb4f28237ca8e5450b39f896c4d89e3081d24c"
EXPECTED_DATASET_SHA256 = "7575444d4d605b1f7155ab937cf541243afe07fce1afba998bf4195e8e84e1eb"
EXPECTED_CONFIG_SHA256 = "f28d7436becf6f7386dd7a129c80a04b1914137569f865e1b890d80ae2864d7c"


class ExperimentalRequestError(ValueError):
    """An explicit experimental chat command is malformed or incomplete."""


@dataclass(frozen=True)
class PredictionEvidence:
    """Validated, prompt-safe subset of the exact experimental service result."""

    model_id: str
    run_id: str
    evaluation_split: str
    feature_variant: str
    artifact_sha256: str
    dataset_sha256: str
    config_sha256: str
    target: str
    prediction_cutoff: str
    prediction: str
    predicted_class: int
    raw_model_score: float
    threshold: float
    local_decision_path: Mapping[str, Any]
    global_feature_importance: Mapping[str, Any] | None
    historical_context: Mapping[str, Any]
    warnings: tuple[str, ...]

    @staticmethod
    def _number(value: Any, field: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ExperimentalRequestError(f"Prediction evidence has an invalid {field}.")
        number = float(value)
        if not math.isfinite(number):
            raise ExperimentalRequestError(f"Prediction evidence has an invalid {field}.")
        return number

    @classmethod
    def from_service_result(cls, result: Mapping[str, Any]) -> "PredictionEvidence":
        """Validate service output and exclude local filesystem paths from prompts."""
        if not isinstance(result, Mapping):
            raise ExperimentalRequestError("Experimental prediction evidence is malformed.")
        model = result.get("model")
        quality = result.get("input_quality")
        explanation = result.get("explanation")
        if not isinstance(model, Mapping) or not isinstance(quality, Mapping) or not isinstance(explanation, Mapping):
            raise ExperimentalRequestError("Experimental prediction evidence is missing required metadata.")
        if result.get("experimental") is not True or result.get("probabilities_calibrated") is not False:
            raise ExperimentalRequestError("Prediction evidence has contradictory experimental or calibration status.")
        if result.get("target") != TARGET or result.get("score_semantics") != SCORE_SEMANTICS:
            raise ExperimentalRequestError("Prediction evidence has an unsupported target or score definition.")
        score = cls._number(result.get("raw_model_score"), "raw score")
        threshold = cls._number(result.get("threshold"), "threshold")
        if not (0 <= score <= 1 and 0 <= threshold <= 1):
            raise ExperimentalRequestError("Prediction evidence score or threshold is outside [0, 1].")
        predicted_class = result.get("predicted_class")
        if type(predicted_class) is not int or predicted_class not in (0, 1):
            raise ExperimentalRequestError("Prediction evidence has an invalid predicted class.")
        if predicted_class != int(score >= threshold):
            raise ExperimentalRequestError("Prediction evidence class contradicts its saved threshold and score.")
        expected_label = "logged_failure_within_horizon" if predicted_class else "no_logged_failure_classified_within_horizon"
        if result.get("prediction") != expected_label:
            raise ExperimentalRequestError("Prediction evidence label contradicts its predicted class.")

        required_model = ("model_id", "run_id", "evaluation_split", "feature_variant",
                          "artifact_sha256", "dataset_sha256", "config_sha256")
        for field in required_model:
            if not isinstance(model.get(field), str) or not model[field].strip():
                raise ExperimentalRequestError(f"Prediction evidence has no valid {field}.")
        for field in ("artifact_sha256", "dataset_sha256", "config_sha256"):
            value = model[field]
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
                raise ExperimentalRequestError(f"Prediction evidence has invalid {field} provenance.")
        if (model["run_id"] != EXPECTED_RUN_ID
                or model["evaluation_split"] != "chronological_known_machines"
                or model["feature_variant"] != "all_features"
                or model["artifact_sha256"] != EXPECTED_ARTIFACT_SHA256
                or model["dataset_sha256"] != EXPECTED_DATASET_SHA256
                or model["config_sha256"] != EXPECTED_CONFIG_SHA256):
            raise ExperimentalRequestError("Prediction evidence provenance does not match the chat-enabled candidate.")
        cutoff = result.get("prediction_cutoff")
        if not isinstance(cutoff, str) or not cutoff.strip():
            raise ExperimentalRequestError("Prediction evidence has no prediction cutoff.")

        if (quality.get("historical_window_complete_as_declared") is not True
                or quality.get("latest_telemetry_strictly_precedes_cutoff") is not True
                or quality.get("latest_telemetry_cadence_hours") != 1
                or quality.get("feature_pipeline") != FEATURE_PIPELINE):
            raise ExperimentalRequestError("Prediction evidence does not satisfy the historical-input contract.")
        warnings = result.get("warnings")
        if not isinstance(warnings, list) or not all(isinstance(item, str) for item in warnings):
            raise ExperimentalRequestError("Prediction evidence warnings are missing or malformed.")
        warning_text = " ".join(warnings).lower()
        if "synthetic" not in warning_text or "not validated on company machinery" not in warning_text:
            raise ExperimentalRequestError("Prediction evidence omits the synthetic-data limitation.")

        local = explanation.get("local_decision_path")
        if (explanation.get("available") is not True or not isinstance(local, Mapping)
                or local.get("available") is not True or not isinstance(local.get("path_nodes"), list)
                or not local.get("path_nodes")):
            raise ExperimentalRequestError("Prediction evidence has no supported local tree explanation.")
        leaf = local.get("leaf")
        accounting = local.get("path_score_accounting")
        if not isinstance(leaf, Mapping) or not isinstance(accounting, Mapping):
            raise ExperimentalRequestError("Prediction evidence is missing leaf or path-score metadata.")
        proportions = leaf.get("stored_class_proportions")
        if (not isinstance(proportions, list) or len(proportions) != 2
                or not isinstance(leaf.get("class_labels"), list)
                or any(type(value) is not int for value in leaf["class_labels"])
                or leaf["class_labels"] != [0, 1]):
            raise ExperimentalRequestError("Prediction evidence has malformed leaf class statistics.")
        normalized = [cls._number(value, "leaf class share") for value in proportions]
        if any(value < 0 or value > 1 for value in normalized) or not math.isclose(sum(normalized), 1.0, abs_tol=1e-9):
            raise ExperimentalRequestError("Prediction evidence has invalid leaf class shares.")
        if (leaf.get("positive_class_share_matches_raw_score") is not True
                or not math.isclose(normalized[1], score, rel_tol=1e-12, abs_tol=1e-12)):
            raise ExperimentalRequestError("Prediction evidence leaf share contradicts the raw model score.")
        changes = [cls._number(node.get("positive_class_share_change_from_parent"), "path contribution")
                   for node in local["path_nodes"] if isinstance(node, Mapping)]
        if len(changes) != len(local["path_nodes"]):
            raise ExperimentalRequestError("Prediction evidence contains a malformed decision-path node.")
        for node in local["path_nodes"]:
            if (type(node.get("node_id")) is not int
                    or not isinstance(node.get("feature_name"), str)
                    or not isinstance(node.get("transformed_feature"), str)
                    or not isinstance(node.get("feature_definition"), str)
                    or node.get("branch_taken") not in ("left (<= threshold)", "right (> threshold)")
                    or not isinstance(node.get("split_condition_original_scale"), str)):
                raise ExperimentalRequestError("Prediction evidence contains incomplete decision-path metadata.")
            transformed_value = cls._number(node.get("transformed_value"), "transformed feature value")
            condition = node.get("transformed_split_condition")
            if not isinstance(condition, str) or " <= " not in condition:
                raise ExperimentalRequestError("Prediction evidence has a malformed transformed split condition.")
            try:
                split_threshold = float(condition.rsplit(" <= ", 1)[1])
            except ValueError as exc:
                raise ExperimentalRequestError("Prediction evidence has a malformed transformed split threshold.") from exc
            if not math.isfinite(split_threshold):
                raise ExperimentalRequestError("Prediction evidence has a non-finite transformed split threshold.")
            expected_branch = "left (<= threshold)" if transformed_value <= split_threshold else "right (> threshold)"
            if node["branch_taken"] != expected_branch:
                raise ExperimentalRequestError("Prediction evidence branch direction contradicts its transformed value and threshold.")
            if type(node.get("child_node_id")) is not int:
                raise ExperimentalRequestError("Prediction evidence has an invalid child node identifier.")
            cls._number(node.get("feature_value"), "feature value") if not isinstance(node.get("feature_value"), str) else None
        root_share = cls._number(local.get("root_positive_class_share"), "root class share")
        accounted_score = cls._number(accounting.get("root_share_plus_path_changes"), "path score")
        accounting_raw_score = cls._number(accounting.get("raw_model_score"), "path raw score")
        sample_count = leaf.get("sample_count")
        if type(sample_count) is not int or sample_count < 1:
            raise ExperimentalRequestError("Prediction evidence has an invalid leaf sample count.")
        cls._number(leaf.get("weighted_sample_count"), "weighted leaf sample count")
        if (accounting.get("matches_raw_model_score") is not True
                or not math.isclose(accounted_score, score, rel_tol=1e-10, abs_tol=1e-10)
                or not math.isclose(accounting_raw_score, score, rel_tol=1e-12, abs_tol=1e-12)
                or not 0 <= root_share <= 1
                or not math.isclose(root_share + sum(changes), score, rel_tol=1e-10, abs_tol=1e-10)):
            raise ExperimentalRequestError("Prediction evidence path accounting contradicts the raw score.")

        global_importance = explanation.get("global_feature_importance")
        if global_importance is not None:
            if (not isinstance(global_importance, Mapping) or global_importance.get("available") is not True
                    or "global model-level" not in str(global_importance.get("scope", ""))
                    or not isinstance(global_importance.get("ranked_features"), list)):
                raise ExperimentalRequestError("Prediction evidence has malformed global importance metadata.")
            for feature in global_importance["ranked_features"]:
                if not isinstance(feature, Mapping) or not isinstance(feature.get("feature_name"), str):
                    raise ExperimentalRequestError("Prediction evidence has a malformed global feature entry.")
                importance = cls._number(feature.get("native_tree_importance"), "global importance")
                if importance < 0:
                    raise ExperimentalRequestError("Prediction evidence has a negative global importance.")

        # Copy only model behavior and provenance. In particular, never pass
        # artifact_path or the caller's full engineered input vector to Ollama.
        return cls(
            model_id=model["model_id"], run_id=model["run_id"],
            evaluation_split=model["evaluation_split"], feature_variant=model["feature_variant"],
            artifact_sha256=model["artifact_sha256"], dataset_sha256=model["dataset_sha256"],
            config_sha256=model["config_sha256"], target=TARGET, prediction_cutoff=cutoff,
            prediction=expected_label, predicted_class=predicted_class, raw_model_score=score,
            threshold=threshold, local_decision_path=dict(local),
            global_feature_importance=dict(global_importance) if global_importance is not None else None,
            historical_context={
                "status": "caller-declared; not independently verified by the service",
                "complete_prior_168_hour_history_as_declared": True,
                "latest_telemetry_strictly_precedes_cutoff": True,
                "feature_pipeline": FEATURE_PIPELINE,
                "training_range_check": quality.get("training_range_check"),
            },
            warnings=tuple(warnings),
        )

    def prompt_payload(self) -> dict[str, Any]:
        return {
            "experimental": True,
            "model": {
                "model_id": self.model_id, "run_id": self.run_id,
                "evaluation_split": self.evaluation_split, "feature_variant": self.feature_variant,
                "artifact_sha256": self.artifact_sha256,
                "dataset_sha256": self.dataset_sha256, "config_sha256": self.config_sha256,
            },
            "target": self.target,
            "prediction_cutoff": self.prediction_cutoff,
            "prediction": self.prediction,
            "predicted_class": self.predicted_class,
            "raw_model_score": self.raw_model_score,
            "score_semantics": SCORE_SEMANTICS,
            "threshold": self.threshold,
            "probabilities_calibrated": False,
            "local_decision_path": dict(self.local_decision_path),
            "global_feature_importance": dict(self.global_feature_importance) if self.global_feature_importance else None,
            "historical_context": dict(self.historical_context),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class ExperimentalChatResult:
    answer: str
    sources: tuple[str, ...]
    outcome: str
    retrieved_documents: int


def is_experimental_request(question: str) -> bool:
    """Recognize only an explicit command at the start of the chat message."""
    stripped = question.lstrip()
    return stripped.lower() == COMMAND or stripped[:len(COMMAND)].lower() == COMMAND and (
        len(stripped) == len(COMMAND) or stripped[len(COMMAND)].isspace()
    )


def parse_experimental_request(question: str) -> tuple[dict[str, Any], str]:
    if not is_experimental_request(question):
        raise ExperimentalRequestError("Start the message with /experimental-pdm to request this experiment.")
    payload = question.lstrip()[len(COMMAND):].strip()
    if not payload:
        raise ExperimentalRequestError("No prediction request was provided. Add a JSON envelope with request and question fields.")
    try:
        envelope = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ExperimentalRequestError("The experimental request must contain valid JSON after /experimental-pdm.") from exc
    if not isinstance(envelope, dict) or set(envelope) != {"request", "question"}:
        raise ExperimentalRequestError("The JSON envelope must contain exactly 'request' and 'question'.")
    prediction_request = envelope["request"]
    maintenance_question = envelope["question"]
    if not isinstance(prediction_request, dict):
        raise ExperimentalRequestError("The 'request' field must contain the experimental service request object.")
    if not isinstance(maintenance_question, str) or not maintenance_question.strip():
        raise ExperimentalRequestError("The 'question' field must contain a maintenance-document question.")
    if len(maintenance_question) > 1000:
        raise ExperimentalRequestError("The maintenance-document question must be 1,000 characters or fewer.")
    return prediction_request, maintenance_question.strip()


def _load_service():
    """Import the candidate only after explicit opt-in and JSON validation."""
    from experiments.azure_pdm.service import AzurePdmExperimentalService

    return AzurePdmExperimentalService()


def _citation(index: int, item: Mapping[str, Any]) -> str:
    return f"[{index}] {item['source']}, {item['locator']}"


def _safe_fallback(evidence: PredictionEvidence, documentation_status: str) -> str:
    status = "classified a logged failure event within 24 hours" if evidence.predicted_class else "did not classify a logged failure event within 24 hours"
    answer = (
        "The experimental Azure PdM inference succeeded. The model " + status +
        f" (raw uncalibrated score {evidence.raw_model_score:.6f}; saved threshold {evidence.threshold:.6f}). "
        "This is model behavior on a synthetic benchmark, not a confirmed fault diagnosis or a calibrated probability."
    )
    if documentation_status == "available":
        answer += " Relevant document passages were retrieved, but SMARTA could not generate a grounded document explanation; review the cited sources."
    elif documentation_status in {"retrieval_failed", "corpus_unavailable"}:
        answer += " Maintenance-document retrieval was unavailable, so SMARTA cannot assess or recommend a corrective action."
    else:
        answer += " No usable maintenance documentation was retrieved, so no corrective action is supported."
    return answer


def handle_experimental_chat(
    message: str,
    retrieve_fn: Callable[[str], list[dict[str, Any]]],
    generate_fn: Callable[..., str],
    *,
    corpus_ready: bool,
    service_factory: Callable[[], Any] | None = None,
) -> ExperimentalChatResult:
    """Run one explicit prediction, then retrieve docs and generate a separated answer."""
    prediction_request, maintenance_question = parse_experimental_request(message)
    try:
        from experiments.azure_pdm.service import ExperimentalInputError
    except Exception:
        return ExperimentalChatResult(
            answer="No experimental prediction was produced because the isolated Azure candidate could not be loaded. SMARTA did not fall back to the production AI4I model.",
            sources=(), outcome="prediction_failed", retrieved_documents=0,
        )

    try:
        service = (service_factory or _load_service)()
        result = service.predict(prediction_request)
    except ExperimentalInputError as exc:
        return ExperimentalChatResult(
            answer=f"No experimental prediction was run. The supplied historical feature request is invalid or incomplete: {exc}",
            sources=(), outcome="prediction_input_invalid", retrieved_documents=0,
        )
    except Exception:
        return ExperimentalChatResult(
            answer="No experimental prediction was produced because the isolated Azure candidate could not be loaded or run. SMARTA did not fall back to the production AI4I model.",
            sources=(), outcome="prediction_failed", retrieved_documents=0,
        )

    try:
        evidence = PredictionEvidence.from_service_result(result)
    except ExperimentalRequestError:
        return ExperimentalChatResult(
            answer="The experimental service returned incomplete or inconsistent evidence. SMARTA did not use it to generate an explanation.",
            sources=(), outcome="evidence_invalid", retrieved_documents=0,
        )
    docs: list[dict[str, Any]] = []
    retrieval_failed = False
    if not corpus_ready:
        documentation_status = "corpus_unavailable"
    else:
        try:
            retrieved = retrieve_fn(maintenance_question)
            docs = [item for item in retrieved if isinstance(item, Mapping)
                    and all(isinstance(item.get(field), str) for field in ("source", "locator", "text"))]
            documentation_status = "available" if docs else "no_matches"
        except Exception:
            retrieval_failed = True
            documentation_status = "retrieval_failed"

    context = "\n\n".join(
        f"[{index}] {item['source']}, {item['locator']}\n{item['text']}"
        for index, item in enumerate(docs, 1)
    )
    citations = tuple(_citation(index, item) for index, item in enumerate(docs, 1))
    timings: dict[str, float | None] = {}
    try:
        answer = generate_fn(
            context=context,
            question=maintenance_question,
            timings=timings,
            prediction_evidence=evidence.prompt_payload(),
            documentation_status=documentation_status,
        )
        outcome = "success" if docs else "insufficient_documentation"
        if retrieval_failed:
            outcome = "retrieval_failed"
        return ExperimentalChatResult(answer=answer, sources=citations, outcome=outcome,
                                      retrieved_documents=len(docs))
    except Exception:
        return ExperimentalChatResult(
            answer=_safe_fallback(evidence, documentation_status), sources=citations,
            outcome="generation_failed", retrieved_documents=len(docs),
        )
