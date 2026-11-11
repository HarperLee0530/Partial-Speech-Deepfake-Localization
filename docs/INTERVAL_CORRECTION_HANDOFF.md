# Interval correction: experiment handoff

## A. Change Summary

This implementation adds a frozen-SAL **single-round modification** experiment.
It does not implement a reinspection network or retrain SAL. All reported smoke
results are synthetic; no real-data improvement has been established.

| File | Purpose / inputs / outputs |
| --- | --- |
| `code/interval_correction/__init__.py` | Independent package and usage overview |
| `code/interval_correction/core.py` | Scores + original intervals → shared KEEP/DELETE/INSERT/SPLIT/MERGE/ADJUST pool, B rules, conflict-safe selection |
| `code/interval_correction/metrics.py` | Predictions + GT → research event metrics, fixed risk/gain labels, offline coverage/oracle diagnostics |
| `code/interval_correction/data.py` | Read-only actual SAL SQLite adapter, portable JSONL adapter, explicit source-group nested manifests |
| `code/interval_correction/model.py` | Candidate acoustic features (+ optional transcript categories) → scalar expected gain; MLP training/checkpoints |
| `code/interval_correction/cli.py` | Fixture/check/split/train/evaluate commands; comparison CSV, traces, provenance |
| `code/interval_correction/config.json` | All candidate/rule/risk settings; defaults are illustrative, NOT selected on test |
| `code/interval_correction/requirements.txt` | Standalone cached-output dependencies |
| `code/tests/test_interval_correction.py` | Synthetic interval, leakage, SQLite, semantic, training and inference tests |
| `code/semantic/analysis.py` | Existing `align()` now validates word spans even when called without `annotate()`; malformed spans previously reached fraction division |
| `docs/REPOSITORY_AUDIT_INTERVAL_CORRECTION.md` | Actual source audit, semantic findings and verification boundaries |
| `README.md` | Link to the new optional protocol; previous experiment behavior retained |

## B. Architecture

1. Read frozen SAL cache; retain original argmax intervals exactly on the saved grid.
2. Generate ONE deterministic pool from scores and original predictions. No GT,
   word categories, error tables or gain labels determine candidate locations.
3. A returns original intervals. B scores this pool using configurable evidence
   rules. C regresses risk improvement using acoustic features. D adds transcript
   category features to the SAME candidate objects; it does not alter proposals.
4. Compare each edit against KEEP=0. C/D require predicted gain strictly greater
   than `minimum_gain`. Deterministic greedy selection rejects shared source
   intervals and overlapping affected regions. All edits reference the original
   baseline; there is no second pass.
5. GT is read only by label/evaluation/explicit offline diagnostic functions.
   `features()` and network `predict()` do not access `Record.truth`.

Intervals are `[start,end)` in seconds. Overlaps are unioned; touching intervals
remain separate events. Generated edits cannot consume unreferenced original
intervals. INSERT includes low-threshold runs even if their maximum is below .5,
but only runs disjoint from original intervals are inserted. Very close disjoint
true events can still be mistakenly merged; acoustic heuristics cannot know GT.
Boundary edits use connected score runs within a configurable displacement radius.

## C. Environment Setup

Run commands from **the repository's `code` directory** in your terminal.
Python 3.10+ and PyTorch 2.2+ are supported by the implementation; execution here
was verified on Python **3.12.14**, torch **2.15.0+cpu**, numpy **2.3.5**, scipy
**1.17.0**, pytest **9.2.0**. The repository's existing SAL environment pins torch
2.2.1/numpy 1.26.4; that older environment was not exercised in this session.

```bash
python -m pip install -r interval_correction/requirements.txt
```

For CPU-only machines, install a matching CPU PyTorch wheel first. No GPU, SAL
weight, torchaudio, Whisper, or audio decoding is needed for the new standalone
commands. Existing CNN/SAL/ASR commands still need their original dependencies.
Use a separate virtual environment if you do not want to alter the SAL environment.

## D. Data Preparation

### Actual SAL cache (adapter implemented; real cache not available here)

Supply each existing run directory containing:

- `config.json`: `split` = train/dev/eval/lps and `resolution_seconds`.
- `predictions.sqlite`: table `predictions`, columns `uid`, `ordinal`, `logits`,
  `labels`, `seconds`. `logits`: raw little/native-endian float32 N×2 bytes;
  `labels`: uint8 N bytes, 0=spoof and 1=bonafide. Audio seconds are real duration.
- If `status.json` exists it must say `complete`; partial interrupted runs rejected.

The schema comes from `code/sal/pipeline.py`. Spoof score is
`softmax(logits)[0] = sigmoid(logit_0-logit_1)`, an **uncalibrated softmax score**.
A uses `argmax==0`, including ties as spoof. A is never recomputed at a tuned
threshold. PS dt=.16; LPS crossfade dt=.02. Do not resample either cache.
The scored horizon is `min(audio_seconds,N*dt)`; unscored audio tails are recorded
by `check` and excluded from event metrics rather than invented as extra frames.

Provide a verified source map (`source_groups.jsonl`), one JSON object per line:

```json
{"uid":"REAL_UID_FROM_CACHE","source_group":"VERIFIED_ORIGINAL_SOURCE_ID","split":"train"}
```

Use split `dev` for official PS dev and `test` for PS eval/LPS. Do not guess source
groups by truncating filenames. A recording with multiple shared sources requires
a connected-component group containing all related variants. Build/verify that
map externally; the code cannot infer identities from audio. Cross-split source
collisions are rejected. UID collisions across datasets must be namespaced in all
inputs consistently; loader refuses collisions instead of overwriting records.

### Portable records / mock fixtures

`records.jsonl` fields:

```json
{"uid":"example","scores":[0.1,0.7,0.2],"dt":0.16,"seconds":0.48,"original":[[0.16,0.32]],"truth":[[0.16,0.32]],"source_group":"source-example","split":"train"}
```

Scores must already be explicit spoof softmax values, not raw logits, EER margins,
CNN scores with opposite labels, or lexical frequencies. `truth` is optional for
candidate generation, but required by training and evaluation.

### D semantic inputs

Use existing `semantic.analysis.annotate_file()` output: JSONL uid, words with
text/start/end/primary_type, optional confidence and annotation metadata.
`ner_model` and `annotation_method` metadata are accepted. Word times must be
finite, sorted, nonoverlapping and inside real audio. Invalid alignment yields an
all-zero semantic vector with availability=0. Missing transcripts also fall back.
Unexpected target/outcome/prior fields are explicitly rejected.

D uses seven duration-weighted category proportions (negation, quantity, time,
person, place, organization, other), availability, covered fraction, and available
confidence mean. It does not multiply semantics into SAL scores. OOV words are
`other`, not high/low spoof probability. Existing frozen prior export is **not**
the D input for this first version. This avoids hidden label exposure from full
train priors when validation groups are drawn from that same train pool.
ASR/NER correctness on real audio still needs manual verification.

## E. Commands

### Fully executed synthetic command chain

All following paths/arguments were exercised here (with an absolute temporary
output root instead of `../result/ic_smoke`). Use fresh output names on reruns;
existing outputs intentionally are not overwritten.

```bash
python -m interval_correction.cli fixture --output ../result/ic_smoke
python -m interval_correction.cli check --input ../result/ic_smoke/records.jsonl --output ../result/ic_smoke/check.json
python -m interval_correction.cli split --input ../result/ic_smoke/records.jsonl --output ../result/ic_smoke/splits.json
python -m interval_correction.cli train --input ../result/ic_smoke/records.jsonl --manifest ../result/ic_smoke/splits.json --modes C --allow-smaller --epochs 2 --output ../result/ic_smoke/c
python -m interval_correction.cli train --input ../result/ic_smoke/records.jsonl --semantic ../result/ic_smoke/transcripts.jsonl --manifest ../result/ic_smoke/splits.json --modes D --allow-smaller --epochs 2 --output ../result/ic_smoke/d
python -m interval_correction.cli evaluate --input ../result/ic_smoke/records.jsonl --semantic ../result/ic_smoke/transcripts.jsonl --manifest ../result/ic_smoke/splits.json --allow-smaller --modes A B C D --c-checkpoint ../result/ic_smoke/c/model.pt --d-checkpoint ../result/ic_smoke/d/model.pt --offline-diagnostics --output ../result/ic_smoke/comparison
python -m pytest tests/test_interval_correction.py tests/test_semantic.py -q
```

The fixture has 20 train / 6 dev / 4 mock test recordings. `--allow-smaller` is an
explicit smoke-test exception. NEVER label these results as PartialSpoof results.

### Real cache commands: implemented adapters, NOT real-data verified

Replace `TRAIN_RUN`, `DEV_RUN`, `TEST_RUN` below with actual existing SAL run
folder names. They are placeholders, not promised repository directories.
All commands use the same repeated inputs so the saved manifest resolves IDs.
No command below triggers SAL inference or model retraining.

```bash
python -m interval_correction.cli check --input ../result/TRAIN_RUN --input ../result/DEV_RUN --input ../result/TEST_RUN --source-map ../data/source_groups.jsonl --output ../result/ic_check.json
python -m interval_correction.cli split --input ../result/TRAIN_RUN --input ../result/DEV_RUN --input ../result/TEST_RUN --source-map ../data/source_groups.jsonl --seed 888 --output ../result/ic_splits.json
python -m interval_correction.cli evaluate --input ../result/TRAIN_RUN --input ../result/DEV_RUN --input ../result/TEST_RUN --source-map ../data/source_groups.jsonl --manifest ../result/ic_splits.json --size 2000 --split dev --modes A B --offline-diagnostics --output ../result/ic_ab_dev_2k
python -m interval_correction.cli train --input ../result/TRAIN_RUN --input ../result/DEV_RUN --input ../result/TEST_RUN --source-map ../data/source_groups.jsonl --manifest ../result/ic_splits.json --size 2000 --modes C --output ../result/ic_c_2k
python -m interval_correction.cli evaluate --input ../result/TRAIN_RUN --input ../result/DEV_RUN --input ../result/TEST_RUN --source-map ../data/source_groups.jsonl --manifest ../result/ic_splits.json --size 2000 --split dev --modes A B C --c-checkpoint ../result/ic_c_2k/model.pt --offline-diagnostics --output ../result/ic_abc_dev_2k
```

After separately preparing timestamped annotated transcripts, D:

```bash
python -m interval_correction.cli train --input ../result/TRAIN_RUN --input ../result/DEV_RUN --input ../result/TEST_RUN --source-map ../data/source_groups.jsonl --semantic ../data/annotated_transcripts.jsonl --manifest ../result/ic_splits.json --size 2000 --modes D --output ../result/ic_d_2k
python -m interval_correction.cli evaluate --input ../result/TRAIN_RUN --input ../result/DEV_RUN --input ../result/TEST_RUN --source-map ../data/source_groups.jsonl --semantic ../data/annotated_transcripts.jsonl --manifest ../result/ic_splits.json --size 2000 --split dev --modes A B C D --c-checkpoint ../result/ic_c_2k/model.pt --d-checkpoint ../result/ic_d_2k/model.pt --offline-diagnostics --output ../result/ic_abcd_dev_2k
```

`evaluate --modes A B C D` generates the unified comparison table; no separate
unimplemented comparison script is needed. Repeat train/evaluate with `--size
5000` and `20000` and fresh names, using the SAME manifest and configuration.
If a full source group prevents exact recording budgets, inspect `actual` in the
manifest; only use `--allow-smaller` after explicitly deciding to report that
actual count. Insufficient available caches never silently become a 20k run.

After selecting all settings on dev, use the same evaluate command with
`--split test` and a fresh final output name. Freeze config/checkpoints first;
do not iterate on official test. PS and LPS should normally have separate final
manifests/runs so dataset-specific results are reported separately.

## F. Expected Outputs

| Step | Outputs |
| --- | --- |
| check | JSON: record counts, grids, scored duration and unscored tail duration |
| split | JSON: nested 2000/5000/20000 IDs, requested/actual counts, source groups, validation and official test IDs |
| train | `model.pt` (best dev loss), `training.jsonl`, `config.json`, `partitions.json`, `provenance.json` |
| evaluate | `metrics.csv`, `predictions.jsonl` with original and corrected intervals and edit traces, `operations.json`, `failure_cases.json` (first 100 worsened cases), `run_log.json`, `config.json`, `partitions.json`, `provenance.json` |
| explicit offline diagnostics | `OFFLINE_DIAGNOSTICS_NOT_PREDICTIONS.json` with coverage and oracle bounds |

Runtime covers candidate generation and correction, excludes SAL/ASR inference,
IO, checkpoint loading and offline oracle computation. A has no candidate-generation
cost. Every B/C/D row accounts for the same pool-generation time. All timing is
CPU wall time and should not be mistaken for full pipeline runtime.

## G. Experiment Settings

- Source recording groups are indivisible. Official dev is used if supplied;
  otherwise `--validation-groups` (default 100) reserves train groups BEFORE the
  nested size prefixes. Official test is never manufactured from train.
- Seed 888. Save concrete IDs; compare C/D using exactly the same size/manifest.
  No automatic threshold search exists. Tune only on train/dev and save the config.
- Candidate defaults: main .5, low .3, internal valley .2; merge gap .2s;
  maximum boundary movement .32s; minimum new interval .02s. B deletion threshold
  .6 with isolation gap .4s; merge support .35; insertion support .4.
- Acoustic input: **68 dimensions** = 6 operation flags + 14 summary features +
  16 sampled score values + 16 before-mask values + 16 after-mask values.
  D adds 10 transcript features, total **78 dimensions**. No SAL embedding export
  is needed. Normalization fits ONLY network training candidates.
- MLP 68/78 → 64 → 32 → 1, ReLU, Adam lr .001, Smooth L1, batch 256, max 40 epochs,
  early stopping patience 6 on dev Smooth L1. CLI exposes epochs/patience/batch/lr.
- Gain = risk(before) − risk(after). Risk = missed events + false-positive events
  + .1 × false-positive seconds + Σmatched(1−tIoU). All weights configurable.
  Label samples retain positive, negative, zero and KEEP cases.
- Fixed one-to-one tIoU≥.5 assignment maximizes eligible match count then summed
  IoU, same principle as existing visualization diagnostics. Unmatched GT counts
  as missed (includes match conflicts, not only zero overlap). Unmatched prediction
  counts as FP event. FP duration is predicted time outside the union of all GT,
  including excess duration of matched events. This is explicitly
  `research_event_tiou_v1`, NOT SAL's official frame EER/F1.
- Short event = GT duration ≤.32s, recall under the same matching. Undefined mean
  IoU/short recall/coverage/harm rates use null. Empty-empty P/R/F1 is defined 0.
- Harmful Modification Rate: negative true fixed-baseline gain among SELECTED
  individual edits. Also export harmful_recording_rate for combined correction
  risk increases. These are evaluation labels, not prediction features.
- C/D minimum positive gain .01; KEEP=0 exactly, regardless of network error.
  B heuristics are evidence utilities, not expected risk gains, and use >0.
- Candidate Coverage: fraction of GT events with at least one original/proposed
  interval reaching matching tIoU, ignoring joint conflict feasibility. It is a
  local pool diagnostic, not achieved recall. Offline oracle enumerates compatible
  subsets when ≤14 non-KEEP candidates; larger pools report the valid but loose
  upper bound risk(before), separately named. Single-edit oracle is a feasible
  lower bound. Neither oracle is a deployable result.

## H. Known Limitations

- No real audio, released weights, SAL SQLite runs or Whisper/NER models existed
  in this session. No true dataset training, ASR transcription or performance
  improvement is verified. The legacy full-suite dependency environment differs.
- Source-map truth must be externally audited, especially multiple-source splices.
- Existing README discusses future reinspection and full-train sharing; this new
  optional protocol uses isolated validation and has no reinspection network.
  Do not fit existing full-train semantic priors on modifier validation labels.
- D lexical categories are optional heuristic features, not a language model and
  not calibrated spoof probabilities. Real alignment/category quality is untested.
- Candidate support is limited to saved grid and proposal types. Expansion cannot
  resolve sub-grid boundaries; missed runs below low threshold have no INSERT.
- Single-round greedy conflict selection is deterministic, not a globally optimal
  expected-gain optimizer. Even individually beneficial edits can interact through
  global event matching; combined harm is separately measured.
- Training materializes feature rows in memory; profile candidate counts/RAM on 2k
  before 20k. Offline exact oracle is optional and potentially expensive.
- Existing SAL source is Windows/CUDA oriented (`msvcrt` and `.cuda()`); the new
  cached-output implementation does not import it and works on CPU.
- No automatic hyperparameter tuning, no full two-stage system, no claims of
  statistical significance. Repeat seeds and source-group bootstrap before claims.

## I. Teammate Checklist

- [ ] Fetch the feature branch; review the PR before merging.
- [ ] Create environment and pass the two targeted pytest files.
- [ ] Run synthetic fixture → check → split → train C/D → unified evaluate.
- [ ] Obtain existing SAL train/dev/eval/LPS caches; do not rerun five-hour inference unnecessarily.
- [ ] Verify source-map connected groups and official split identities.
- [ ] Run cache check, inspect unscored tails, verify logits/0=spoof on sample rows.
- [ ] Save one manifest; inspect actual nested counts for 2k/5k/20k.
- [ ] Start A/B dev; inspect edit traces, coverage, wrong merges and harmful changes.
- [ ] Train C on 2k; verify positive/negative/zero label counts and dev early stopping.
- [ ] Manually inspect ASR timestamps/NER on a sample; train D with the same pool.
- [ ] Select settings on dev; save configs, hashes, IDs and rationale.
- [ ] Scale to 5k/20k only if runtime/RAM and validation results justify it.
- [ ] Freeze all settings, then run each official test once and label it `test`.
- [ ] Report PS/LPS separately, include harm/runtime and no oracle as a method.
- [ ] Return metrics, logs, checkpoint and failure examples; do not commit datasets/weights.

### Copyable task for teammate's Codex

> Read `docs/INTERVAL_CORRECTION_HANDOFF.md` and the audit first. Work on
> `feature/interval-correction-gain`. Run the targeted pytest and synthetic chain
> before touching real data. Find the actual existing SAL run directories and
> verify their SQLite/config schema; do not invent paths or rerun SAL unnecessarily.
> Require an externally verified uid/source-group/split map. Save the nested size
> manifest. Run A/B and then C on 2k with independent dev; only then add annotated
> ASR category inputs for D using exactly the same pool/config/IDs. Do not use
> test labels or full-train outcome priors to build input features or choose
> thresholds. Preserve original predictions and old results. Return actual commands,
> counts, logs, metrics CSV, failures, limitations and checkpoint hashes. Do not
> claim improvement before real evaluation, and do not start 5k/20k automatically.

### Validation scope for this delivery

The final standalone command `python -m pytest tests/test_interval_correction.py
tests/test_semantic.py -q` passes **38 tests**. A broader attempt had 44 passes
and one existing CNN/SAL regression blocked by `sal.pipeline` importing the
Windows-only `msvcrt` module on Linux. That does not certify the full legacy suite;
run it in the original Windows environment. The old test also requires the tests
directory on the Python import path. No full-data or ASR run was performed.
