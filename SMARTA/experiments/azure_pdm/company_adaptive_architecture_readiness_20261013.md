# SMARTA Company-Adaptive Predictive Maintenance: Architecture and Readiness

**Status:** Phase 1 architecture/readiness assessment. This is not a production migration, model promotion, or company-data audit. The existing AI4I production model and API remain authoritative. Azure PdM remains an isolated offline experiment.

## Findings from the existing experiment

The experiment reads only the five CSV members in the supplied ZIP: `PdM_telemetry.csv`, `PdM_failures.csv`, `PdM_errors.csv`, `PdM_maint.csv`, and `PdM_machines.csv`. Its audited copy has 876,100 hourly telemetry rows for 100 machine IDs, covering 2015-01-01 06:00 through 2016-01-01 06:00. The schema includes `volt`, `rotate`, `pressure`, and `vibration`; failure labels `comp1`–`comp4`; error codes `error1`–`error5`; maintenance component codes; and static machine `model` and `age` fields. The experiment excludes machine ID and age as predictors; model category appears only in an explicit ablation. The archive matches Microsoft's synthetic benchmark schema, but its supplied ZIP has no dataset-specific license and permission to redistribute or deploy it is unverified.

The pipeline derives 3-hour, 24-hour, and 168-hour trailing telemetry summaries, deltas, and prior error/failure/maintenance counts. It shifts telemetry so all observations precede the cutoff; event windows are `[t-window,t)`. Targets are logged failure-table events in `(t,t+H]` for 1, 24, or 168 hours, with simultaneous component failures preserved as multi-label targets. Failure rows without an exact maintenance match (18) remain failure labels; maintenance-only rows are historical features and do not become target labels. These labels do not establish physical failure onset or verified repair outcomes.

Evaluation uses chronological and seeded machine-held-out splits, a 168-hour boundary purge, common eligible rows, prior/logistic/tree candidates, validation-only threshold selection, and machine-cluster bootstrap intervals. The saved-artifact audit reproduced locked-test scores and found no direct future telemetry, target, timestamp, machine ID, or machine-age feature in saved schemas. Error history carries much of the benchmark signal: for 24-hour known-machine diagnostic ablations, AP was 0.5685 using error-history alone and 0.2565 using telemetry alone; all available features scored 0.9421. This is a synthetic logged-event association, not evidence of company-machine warning capability.

The experiment's operational alert budget and minimum event recall are unset. Its probability-like outputs are uncalibrated scores. Therefore acceptance is undecided. The available evidence is in [experiment README](README.md), [configuration](config.json), [pipeline](pipeline.py), [tests](tests/test_pipeline.py), [integrity audit](reports/audit_20261013T130445_338994000Z/integrity_audit.md), and [saved experiment report](reports/run_20261013T112729_896513000Z/experiment_report.md). No company historian/CMMS export or company SQL connectivity was verified as part of this architecture assessment; company data availability, ownership, and permissions are unknown.

## Recommended architecture

Use a **versioned schema and model registry with machine-family-specific validated contracts**. Begin with separate, auditable models per family and prediction task/horizon where each family has enough confirmed events. Share a model only across families when sensor semantics, units, sampling, labels, and operating context are compatible and held-out evidence supports pooling. If data are sparse, pooled training can be evaluated with family indicators and family-stratified reporting, but it must not conceal poor performance for a family. Keep family adapters or separate models where schemas differ. Do not assume one universal feature vector or that every family can support component-level targets.

| Option | Benefits | Risks and when appropriate |
|---|---|---|
| Shared model | More examples per fit; simpler fleet-wide maintenance | Schema/label differences can create hidden domain shift; use only for semantically compatible features and validated cross-family results |
| Family-specific model | Clear feature/label semantics and family-specific thresholds | Fewer events per model; require sufficient time- and machine-held-out evidence |
| Recommended staged choice | Start family-specific contracts; compare pooled and per-family baselines in offline evaluation | Keep a family/task in “not supported” state until its data volume and acceptance gates are met |

Each registry entry should bind `machine_family`, `task`, `horizon`, `feature_schema_version`, model/preprocessing versions, ordered feature names and units, training-data lineage/hash, label definition, split and evaluation report, threshold-selection policy, calibration status, and approval state. Store raw observations with event/record time and quality metadata, then build cutoff-aware features from an immutable, time-aware view. Preprocessing is fitted on training data only and versioned with the model. Model ID alone must never imply schema compatibility.

At inference, validate the requested family/schema, required fields, units, timestamp freshness, cadence/history coverage, and documented equipment-specific validity limits. Reject malformed or unsafe inputs with an explicit reason. Missing or stale fields should produce an incomplete/unavailable assessment unless an explicitly validated missing-data policy exists; never silently substitute zero, clamp, or mix incompatible sensor schemas. Report out-of-training-distribution conditions separately from physical invalidity. Return the task and horizon, label meaning, schema/model versions, observation cutoff, raw score and calibration status, supported targets, data-quality flags, and explanation/reference IDs. Keep inference outputs distinct from RAG-generated diagnoses.

## Company-data contract

### Required before a forecasting experiment

- Stable asset identifier and machine family/type, model/variant, and effective dates for configuration changes.
- Timestamped sensor observations: canonical sensor/feature name, raw value, unit, sensor ID, event time and ingestion time/time zone, quality/validity flag, calibration and replacement history. Supply sampling cadence, expected cadence, sensor location, conversion metadata, and known clock corrections.
- Confirmed failure-event records with asset, component/failure mode, event time (physical onset if known), detection/log time, confirmation status/source, and event linkage/unique ID. Distinguish unknown onset from an exact onset.
- Operating/at-risk intervals, commissioning and retirement, run/idle state if available, downtime, and explicit data-coverage/missing intervals. These define which time windows can be labeled negative or evaluated.
- Maintenance interventions with action and component, start/end time, replacement/repair details, whether corrective or preventive, linked asset and (where known) linked failure, and verified outcome/follow-up status. A work order alone is not proof of either failure or successful repair.
- Data owner, source system and extraction timestamp, lineage/provenance, permitted purpose, privacy/security controls, license/contract rights, retention/deletion rules, and authorized access.

### Helpful but optional (must be assessed for leakage and availability at cutoff)

Alarm/error logs, production load, product/batch, ambient conditions, operating mode, shift, operator/work-order context, inspection measurements, component age since verified installation, and manufacturer maintenance limits. Include exact source timestamps and availability delays. Future work orders, post-failure measurements, and repair outcomes are not predictors for an earlier cutoff.

## Validation and acceptance gates

1. Agree each target's operational meaning: logged failure versus confirmed physical onset; machine-level versus component-level; prediction cutoff; horizon; censoring and preventive-maintenance treatment; and the action an alert enables.
2. Build as-of labels and features. Split chronologically; purge at least the maximum target horizon plus any known event-reporting delay around boundaries. Keep a locked latest-period test set. Add machine-held-out evaluation for deployment to unseen assets; where possible report a time-forward, machine-held-out cohort. No test-set feature, threshold, or model selection.
3. Compare a constant prior and simple operational/rule baseline with regularized and tree models on identical eligible rows. Fit all preprocessing on training only. Report average precision/PR-AUC alongside precision, recall, false alarms per machine-week, missed failures, event-level recall after alert deduplication, warning lead-time distribution, and coverage/data-quality exclusions. Report confidence intervals clustered by machine and/or event. Break out results by family and failure mode and disclose event counts; sparse strata cannot support strong claims.
4. If probabilities are shown, evaluate calibration on data not used to fit the model, using a time-respecting calibration design, and report calibration error/plots and uncertainty. Otherwise label outputs as uncalibrated scores.
5. Before opening the locked test result, operations and safety owners must set an alert budget per machine-week, minimum acceptable event recall, useful lead-time window, and handling of unavailable/stale data. These are currently undecided and must remain configurable until approved. High AP alone is not a deployment gate. Require a documented review of false alarms, misses, subgroup performance, monitoring, rollback, and human response procedures.

## Later integration plan (not implemented in this phase)

1. **Interpretable results:** first establish a validated model and data-quality contract. Add model-appropriate feature contributions or local explanations with versioned background data and uncertainty; state that they describe model behavior, not causal reasons or guaranteed repairs.
2. **Prediction-to-chat handoff:** pass the actual versioned inference result as structured context (asset/family, cutoff, task/horizon, target scores/status, data-quality warnings, model version). Keep chat/RAG explanations separate and never let language generation change classifier output.
3. **Source-grounded recommendations:** retrieve applicable manuals, approved procedures, and verified company maintenance records scoped to the asset/family and predicted mode. Show source title, passage, and citation for every recommendation; when sources do not support an action, say evidence is insufficient. Record source provenance and avoid presenting generated advice as verified repair outcome.

## Unresolved decisions and next milestone

Unknowns include which company machine families are in scope; whether timestamped historian and CMMS records exist and may be used; event onset/confirmation quality; sensor units, cadence, and clock quality; available exposure/operating periods; event counts by family and component; target and useful horizon; preventive-maintenance censoring; alert budget, acceptable missed-event rate and required lead time; governance, retention, and data-owner approval.

**Next implementation milestone:** obtain an authorized, read-only historian + CMMS extract for one nominated machine family, together with its data dictionary, unit/cadence and clock metadata, coverage/operating intervals, confirmed failure and linked intervention definitions, provenance and retention permissions. In a review session, agree one target/horizon and the alert-budget/recall/lead-time gates. Then implement only an isolated ingestion and data-quality/label-coverage audit; proceed to model experiments only if that audit shows adequate, temporally usable examples. Do not connect it to production inference at that milestone.
