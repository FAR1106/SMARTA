# Experimental Azure prediction explanations in SMARTA chat

This feature is an explicit opt-in branch of the existing `POST /chat` handler. It is not a new prediction endpoint, and the Azure candidate is never selected for ordinary chat or `/predict` requests.

## Invocation

Start a chat message with `/experimental-pdm` followed by a JSON envelope with exactly two fields:

```text
/experimental-pdm {"request": {"features": {"<all exact engineered feature names>": "<finite values>"}, "prediction_cutoff": "<ISO-8601 t>", "historical_context": {"history_start": "<ISO-8601>", "latest_telemetry_at": "<ISO-8601 t-1h>", "telemetry_history_complete": true, "event_history_complete_before_cutoff": true, "feature_pipeline": "experiments.azure_pdm.pipeline.build_features"}}, "question": "What does this model path show, and what do the maintenance records support?"}
```

This is a shape example, not a ready-to-run request. The chat adapter currently selects only the default chronological known-machine candidate and requires its exact 78 engineered feature names. The service applies the saved feature order; JSON key order is not significant. The separate unseen-machine service candidate has a 79-input schema but is not selectable from chat. Obtain the complete engineered vector from the existing prior-only pipeline using at least 168 hours of history. Do not fill missing values with placeholders. A single sensor row cannot run this candidate.

The chat input displays this command syntax as an experimental opt-in hint. The current chat UI does not collect or construct the historical feature vector. A caller must supply the complete, correctly engineered request. Completeness flags are declarations supplied by that caller; the service cannot verify the source history independently.

## Processing and evidence boundaries

1. The command parser validates the explicit prefix and JSON envelope. Ordinary maintenance questions do not import or call the model service.
2. The isolated service runs once and returns the saved Prompt 2 candidate's raw uncalibrated score, validation-selected threshold decision, local fitted-tree path, optional native global importance, and artifact provenance. Missing history, invalid input, or inference failure produces a visible limitation and never falls back to AI4I.
3. A typed internal evidence contract rejects missing or contradictory model metadata before any LLM call. Filesystem artifact paths and the full 78/79-element feature vector are not sent to the LLM.
4. The user's maintenance question is sent through the current retrieval function. Retrieved passages are numbered and cited through the chat's existing source list. The LLM receives separate sections for experimental model evidence and untrusted maintenance-document excerpts. Experimental tree paths are not treated as physical causes or maintenance instructions.
5. With no retrieved documentation, the prompt explicitly requires the answer to say that available documents do not support a maintenance recommendation. Retrieval, generation, prediction, and malformed-evidence outcomes are handled separately.

The local path and global importance explain this fitted decision tree's behavior only. Its score is not calibrated. The target is a failure-table event logged in `(t, t+24h]`, not verified physical failure onset. The benchmark is synthetic, lacks company validation, and has unresolved operational alert-budget and useful-lead-time criteria. No production model, API prediction contract, or company integration is changed by this experimental branch.

## Failure behavior

- Missing or malformed command JSON: no inference is attempted; chat returns an input limitation.
- Missing or incomplete historical features: the service rejects the request; no prediction, RAG recommendation, or AI4I fallback is produced.
- Candidate load or inference failure: chat reports that no experimental prediction was produced.
- Inconsistent service evidence: evidence is rejected before retrieval or generation.
- Retrieval failure or empty results: the model result can still be explained, but the response must state that documentation is unavailable or insufficient and must not recommend a corrective action.
- LLM generation failure: a deterministic summary reports only the experimental class, raw score, threshold, and document-evidence limitation; retrieved citations remain available in the existing source panel.

## Verification commands

```bash
../.venv/bin/python -m unittest discover -s experiments/azure_pdm/tests -v
../.venv/bin/python -m unittest tests.test_experimental_chat -v
../.venv/bin/python -m unittest discover -s tests -v
cd flutter_app && flutter test && flutter analyze
```
