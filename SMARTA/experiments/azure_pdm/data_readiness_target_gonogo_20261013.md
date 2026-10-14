# Azure PdM Data Readiness, Primary Target, and Go/No-Go

**Assessment date:** 2026-10-13

**Scope:** Existing isolated Azure PdM benchmark artifacts only. No production model, endpoint, UI, dataset archive, or existing run output was changed. No training was rerun for this assessment.

## Evidence reviewed

Reviewed `pipeline.py`, `config.json`, the experiment README and unit tests, the saved run `run_20261013T112729_896513000Z`, its artifact manifest and evaluation JSON, and the later saved-artifact integrity audit `audit_20261013T130445_338994000Z`. The audit's dataset SHA-256 matches the saved experiment artifact manifest (`7575444d…e1eb`); its report says locked-test average precision was reproduced to within `1e-10`. This is reuse of saved evidence, not a fresh data load or model run.

## Data and label readiness

The archive schema is five tables:

| Table | Required fields observed | Coverage/count |
|---|---|---|
| Telemetry | `datetime`, `machineID`, `volt`, `rotate`, `pressure`, `vibration` | 876,100 rows; 100 machines; 2015-01-01 06:00–2016-01-01 06:00; one-hour cadence; 8,761 rows per machine |
| Failures | `datetime`, `machineID`, `failure` (`comp1`–`comp4`) | 761 component rows; 719 distinct machine-hours; 42 hours with simultaneous component labels; 98 machines have at least one row |
| Errors | `datetime`, `machineID`, `errorID` (`error1`–`error5`) | 3,919 rows, including 303 excess same-machine/hour records that are aggregated before feature joins |
| Maintenance | `datetime`, `machineID`, `comp` | 3,286 rows; includes 2,543 exact-key records absent from the failure table and begins before telemetry coverage |
| Machines | `machineID`, `model`, `age` | 100 IDs; four static model categories; age 0–20, excluded because an as-of reference date is not defined |

The data supports **observed telemetry coverage**, not verified operating exposure: it has no explicit running/idle state, commissioning/retirement intervals, or missing-coverage intervals. Every machine has hourly rows in the supplied table, but that does not prove every machine was operating or at risk throughout the period. Failure times are rounded/logged event times; the source does not confirm physical onset. Maintenance is not proof of a failure or a verified successful repair. The 18 unmatched failure component/time rows remain valid source-table labels; 2,543 maintenance-only exact keys are not relabeled as failures.

`build_features` shifts telemetry by one observation before rolling. At decision cutoff `t`, each telemetry window ends at `t-1h`; full 168-hour telemetry history is required. Error, failure, and maintenance event counts use only `[t-window,t)`, with events at `t` excluded. Targets are generated only from failure-table rows; a future logged event exactly at `t` is excluded and the upper endpoint `t+H` is included. Target rows without full future horizon coverage are marked invalid and omitted from that target's evaluation. Machine ID and timestamp are grouping/join fields, not predictors. Machine age is excluded. Machine model is only included in a separate ablation. The saved integrity audit found no direct future telemetry/event, target, machine ID, datetime, or age feature in selected artifacts.

Error-history features contain substantial signal in this benchmark. In the saved 24-hour diagnostic ablation for known machines, error-history-only AP was 0.5685, telemetry-only AP was 0.2565, and all-available-features AP was 0.9421. This finding motivates mandatory error-history ablation and source-quality review; it does not show that errors are available, stable, or predictive in company systems.

## Recommended primary task

**Predict whether any component failure event will be logged for a machine in `(t, t+24h]`, using information available strictly before cutoff `t`.** This is a machine-level binary target formed as the OR of the four component labels. Keep component targets as secondary analyses, not the primary go/no-go target.

Why this target: it has more events and a more informative existing benchmark evaluation than the next-hour task; it also gives a clearly bounded horizon for comparing error and sensor history. Existing saved locked-test evidence (reported for audit, not used here to choose a model) includes 95 logged-event instances in the chronological test and 16 in the machine-held-out test. Sparse machine-held-out event count means the estimate is imprecise. It is an offline benchmark of future **logged** events, not prediction of physical failure onset, proof of 24 hours' actionable warning, or a task validated on company equipment.

- **Cutoff convention:** the decision is made at telemetry timestamp `t`; only data with timestamps `<t` are features. An event at `t` is neither future label nor historical feature. The target interval is open-left/closed-right: `(t,t+24h]`.
- **Label and uncertainty:** any of `comp1`–`comp4` logged in the failure table. Simultaneous component events are retained as multi-label records before OR-ing for this machine-level target. Timestamp rounding makes exact boundary and lead-time interpretation approximate.
- **Evaluation populations:** primary chronological evaluation on machines represented in earlier periods; separately report future time blocks for IDs held out entirely from train/validation. The latter tests generalization to this benchmark's unseen IDs, not to a new company/family.
- **Exposure:** labels are known only over telemetry-table coverage. No explicit at-risk/operating mask exists; do not assert full uptime or infer confirmed negatives during unobserved operational periods. Restrict eligibility to the experiment's complete-history and future-label-coverage masks and disclose this limitation.

## Existing saved evaluation (context, not selection)

The saved run uses a constant-prior reference, regularized logistic regression, and bounded decision tree; its selected candidate and threshold were selected from validation. The integrity audit reproduced selected locked-test scores. For the primary 24-hour any-component target:

| Evaluation split | Test rows' logged-event instances | Selected candidate recorded by validation | Test AP vs prior AP | Test precision / recall | Event recall | False alarms per machine-week | Missed events | Median lead among matched events |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Chronological known machines | 95 | Decision tree, base features | 0.9352 vs 0.0176 | 0.9228 / 0.9325 | 0.9592 | 0.0142 | 4 | 23 h |
| Future unseen machines | 16 | Decision tree, model-category ablation | 0.9400 vs 0.0148 | 0.9547 / 0.8776 | 1.0000 | 0.0065 | 0 | 23 h |

These are saved benchmark measurements at validation-derived thresholds; they are not new estimates or a basis for choosing the model in this assessment. Event counts are small, alert-window rows repeat around events, and a machine-cluster AP interval does not establish generalization to real assets. The long lead-time figures are relative to rounded logged timestamps and repeated 24-hour alert windows, not guaranteed operational warning. The prior AP is prevalence; a fixed 0.5 majority decision emits no alerts and misses all events. Error-history ablations are essential because event codes may encode a synthetic precursor relationship.

The experiment's config leaves `false_alarm_budget_per_machine_week` and `minimum_event_recall` null. No company-approved alert budget, minimum event recall, minimum actionable lead time, or cost trade-off is known. Scores are uncalibrated. No probability claim is supported.

## Data sufficiency decision

**Conditional go for a reproducible offline benchmark experiment on the existing synthetic archive; no-go for production, company-data claims, or a deployment recommendation.** There is enough labeled benchmark history to run the pre-existing chronological and machine-held-out comparison for this target, with measurable event counts. Sufficiency is limited by the small number of independent machines/events, missing exposure/operating-state and coverage semantics, rounded event times, synthetic process, unverified repair outcomes, no company records, and unresolved data rights. Existing results can support code-path and benchmark methodology assessment only. They cannot establish real-company performance, true failure onset, calibration, safety, or operational utility.

### Provenance and permissions

- **Verified within the repository audit:** saved dataset hash, schema/count consistency with the known Microsoft Azure PdM sample; audit/run artifacts reference the same hash.
- **Provenance assessment:** consistent with Microsoft's public synthetic benchmark. This is synthetic generated data, not company telemetry.
- **License/permission:** unknown. The supplied ZIP has no dataset-specific license manifest. An upstream repository license is not sufficient evidence that the dataset itself may be redistributed or used in a deployed product. Obtain rights clarification before redistribution/deployment.
- **Company SQL, historian, CMMS, sensors:** not checked or verified by these saved artifacts. No access claim is made.

## Unresolved assumptions and gates

1. Whether company event labels represent confirmed physical onset, detection time, or work-order logging; how preventive interventions censor outcomes.
2. What constitutes operational exposure and known-negative time; whether future data will include run state and explicit coverage intervals.
3. Whether error/alarm logs exist, are timestamped consistently, and are genuinely available at inference cutoff.
4. Per-machine-family event volume and whether family-level targets can be evaluated with meaningful uncertainty.
5. Operational alert budget per machine-week, acceptable minimum event recall, required actionable lead-time band, and intervention capacity. Keep these configurable and undecided until operations/safety owners set them.
6. For any probability output, a separate time-respecting calibration design, calibration metrics/plots, subgroup assessment, and uncertainty bounds are required. Otherwise expose only an explicitly uncalibrated score.

The next sprint must use validation only for feature/model/threshold choices and must not tune against the locked test. If no authorized company data arrives, Prompt 2 is limited to the already-scoped synthetic benchmark experiment; it must retain the no-company-generalization limitation.
