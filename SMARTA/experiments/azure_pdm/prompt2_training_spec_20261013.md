# Prompt 2 — Offline Training Specification

**Status:** Actionable specification for the next sprint prompt; no model has been retrained or promoted by this document. Keep all code and output under `experiments/azure_pdm/`. Do not modify AI4I production artifacts, API, thresholds, UI, or RAG behavior.

## Primary experiment

- **Target:** machine-level binary label `any_failure_logged_next_24h = 1` when any failure-table component event is logged for that machine in `(t, t + 24 hours]`.
- **Cutoff:** `t` is the decision timestamp. Features must be available strictly before `t`; events at `t` are excluded. The interval is open at `t`, closed at `t+24h`. Exclude rows without a full 24-hour future telemetry-label-coverage window and rows without the existing required 168-hour feature history.
- **Semantics:** logged benchmark event only; not confirmed physical failure onset, guaranteed warning, or company equipment prediction. Report component-level targets only as secondary, explicitly exploratory results.
- **Source:** exact five CSVs from the currently audited supplied ZIP and its verified SHA-256; no external data and no AI4I merge. If archive hash/schema differs, stop for an explicit new audit.

## Features allowed

Use current documented trailing-only `volt`, `rotate`, `pressure`, and `vibration` summaries/deltas plus aggregated error-history counts. Error events must be grouped by machine/code/time before feature construction and counted only in `[t-window,t)` for existing windows. Prior failures and maintenance may be included only as strictly historical events with the same cutoff, and must receive a predeclared ablation. Machine ID and datetime are join/group/split keys only. Exclude machine age. Include machine model only in a separately named ablation, never silently in the base schema. No future telemetry, current/future failure or maintenance, post-failure measurements, target-derived columns, or unverified fields. Preserve the current feature availability and data-quality masks; fit preprocessing on training rows only.

## Baselines and candidates

Use the same eligible rows and split masks for every comparison:

1. Constant-prior baseline, with prevalence/AP reported and a separately specified no-alert majority decision. Do not conflate ranking performance with a usable alert policy.
2. Simple trailing error-history baseline (e.g. at least one error in the prior 24h), declared before locked-test evaluation; include a no-error-history model ablation to quantify reliance on event codes.
3. Regularized logistic regression.
4. One bounded-depth decision tree.

Existing saved results may be summarized as reproducibility context. If fitting again, choose model/features/threshold using train and validation only. Do not add model families or tune hyperparameters on the locked test. Machine model, if ablated, is selected and reported independently on validation.

## Splits and leakage safeguards

- Reproduce the saved chronological cutoffs in `config.json` and its 168-hour purge before validation and test boundaries. Require the purge to be at least target horizon plus any documented event reporting delay; because no reporting-delay estimate exists, document that current purge only covers the 168-hour maximum experiment horizon and does not address unknown logging latency.
- Reproduce the seeded 60/20/20 machine-held-out split, stratified only by static machine model, and retain chronological time blocks within each machine group. Verify machine IDs are disjoint across train/validation/test and report event counts by split.
- For the primary 24-hour target, preserve common eligible rows across baselines/candidates. Training fit and preprocessing use training only. Validation selects candidate/feature ablation and threshold. Lock test before reading its outcomes; test is one-time reporting only.
- Re-run focused integrity checks for strict feature cutoff, `(t,t+24h]` target boundaries, correct multi-label-to-any construction, full label coverage, no identifier/target leakage, and split/purge behavior. Treat maintenance-only rows as historical maintenance only; retain unmatched failure labels.

## Required output and acceptance reporting

Create a unique run directory without overwriting prior artifacts. Save a config snapshot with dataset and config hashes, seed, task/horizon, eligibility and split counts, preprocessing and candidate parameters. Save fitted pipelines/feature manifests only to the isolated experiment artifacts directory. Save machine-readable metrics and a concise report under the isolated metrics/reports directories. Include:

- AP/PR-AUC and prevalence/prior baseline on validation and locked test; precision, recall, false positives, false alarms per machine-week, false negatives and missed-event count at a **validation-selected** threshold.
- Event-level recall after a documented alert deduplication/cooldown rule, number of events, matched and missed events, warning lead-time distribution, and evaluated exposure/coverage limitations.
- Separate known-machine chronological and unseen-machine/time-forward results. Include machine-cluster confidence intervals where feasible and warn that 100 synthetic machines and sparse held-out events limit precision.
- Error-history, telemetry-only, without-error-history, and optional model-category ablations, clearly labeled diagnostic if they use locked test. Prefer validation-only ablations for selection; any test ablation is post-selection diagnostic and cannot change the candidate.
- Calibration status. Do not claim calibrated probabilities; a calibration study is a separate later stage with time-respecting calibration data and proper calibration metrics.
- Acceptance values for alert budget/machine-week, minimum event recall, and useful lead-time range as configurable fields. Because none is operationally approved, leave them null/unresolved. Do not claim acceptance or deployment readiness based only on AP.

## Block training/evaluation if

- Archive hash or expected schema changes without a renewed audit; the supplied archive cannot be located/read; or feature/label masks cannot be reconstructed reproducibly.
- Any feature uses information at or after the cutoff, temporal purge is invalid, test rows influence preprocessing/selection/threshold, or machine split is not disjoint.
- Failure labels are silently inferred from maintenance-only rows, missing future coverage is treated as negative, or event timestamp semantics are changed without documentation.
- Required tests for boundaries, leakage, aggregation, eligibility, and purging fail.
- Production or company-data integration is proposed as part of this experimental run. Do not weaken checks to force a score or artifact.

## Prompt 2 completion gate

Prompt 2 may run the isolated, reproducible offline experiment only after confirming these inputs and gates. It must state whether results reproduce the saved run, report failures without substituting fabricated metrics, and leave production AI4I and the live application untouched. The next stage after Prompt 2 is review of benchmark findings and data readiness—not automatic promotion.
