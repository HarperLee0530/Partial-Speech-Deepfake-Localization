# Repository audit: interval correction

Audited GitHub main at `dd0fdb3440e35ff08812861ff895a2d2d7aa30e3`
(2026-11-11). The session initially contained no Git repository, audio, weights,
or result caches. Files were retrieved through the authorized GitHub connection
into an isolated source snapshot; this is not access to the user's VS Code or
uncommitted local changes. Remote commits are based directly on the audited main
commit. No original results or member work was overwritten.

Recent main commits inspected: dd0fdb3 (full training/CNN diagnostics), e1e3e21
(merge fixed10 sharing), 2c7d9d2 (fixed10 sharing). Default branch is main;
feature branch is `feature/interval-correction-gain`.

## Actual code inspected

| Source | Existing behavior / consequence |
| --- | --- |
| `README.md`, `code/README.txt`, dependency/config files | Existing CNN/SAL/semantic full/fixed10 protocol, no implemented candidate/gain module |
| `code/project_config.py` | Config-relative adjacent data/result roots; caches and weights excluded from repository |
| `code/sal/pipeline.py`, `full_data.py`, `fixed10.py` | Public frozen checkpoint inference and separate optional SAL training; actual SQLite schema and config metadata |
| `code/vendor/SAL/src/models/net/model.py` | SSL sequence binary head, 20ms encoder scale pooling to requested resolution; output cached as logits, not probability |
| `code/vendor/SAL/src/data/partialspoof.py`, `llama_partialspoof.py` | Existing datasets and saved-grid target handling |
| `code/vendor/SAL/src/evaluator/evaluator.py` | Official frame EER/F1; F1's positive class is bonafide, not the research event F1 |
| `code/cnn/baseline.py`, config and existing test modules | CNN-BiGRU: 16kHz, hop=320 (20ms); INTERNAL label encoding 1=spoof, opposite SAL; do not feed CNN logits to the SAL adapter |
| `code/semantic/analysis.py`, `full_train_priors.py`, config/README, tests | Local ASR, annotation, word overlap diagnostics, smoothed priors, held-out/group-exclusion exports |
| `visualization/diagnose_saved_predictions.py` | Saved-grid intervals and Hungarian matching maximize eligible match count before summed tIoU |

SAL model load: `load_model()` reads adjacent `data/models/SAL_WavLM.ckpt` and
`wavlm_large.pt`, uses eval mode and inference_mode in prediction. New code never
loads either model. Actual outputs are adjacent `result/<run>/predictions.sqlite`
with run-local config, metrics, status. No actual run folder is checked into Git.
The new adapter consumes this existing format read-only.

PS is evaluated at .16s, LPS crossfade at .02s; `prepare_lps_labels()` floors
boundaries onto the 20ms grid and writes 0 for spoof. Prediction class is binary
argmax, 0=spoof; ties choose 0. Cached scores are raw float32 logits. The derived
softmax spoof score is uncalibrated, not a verified true probability. Audio duration
clips the final grid cell; some dataset paths leave a short unscored tail.

## Semantic review findings

| Severity | Actual path / finding | Action / status |
| --- | --- | --- |
| Medium bug | `code/semantic/analysis.py:align`: direct callers could bypass timestamp validation and reach overlap/duration division with reversed/zero spans | Fixed: call existing `validate_record` at entry; regression test. Existing annotated valid-input behavior unchanged |
| High protocol risk | `code/semantic/full_train_priors.py:freeze_full/export_features`: full-train statistics, even leave-own-group-out, can contain labels of a new modifier validation subset drawn from train | Existing full-train mode kept unchanged. D first version accepts transcript categories only, rejects outcome/prior tables. Fit new priors ONLY within modifier training groups if added later |
| Medium limitation | `analysis.py:transcribe`: genuine local faster-whisper adapter exists, requests word timestamps, no downloads, records ASR model/runtime; model/real audio absent here | Cannot certify actual ASR execution. Existing transcription is fail-fast; no synthetic transcript is presented as real ASR output |
| Medium limitation | `analysis.py:validate_record/align`: validates order, nonoverlap, finiteness; correctly marks partially/unscored words; timestamps are ASR estimates, not forced alignment guarantees | D checks bounds/order and falls back to unavailable semantics on invalid alignment; manual word/audio spot checks remain required |
| Low design limitation | `analysis.py:annotate`: English lexicons for negation/quantity/time; digit matches quantity; optional spaCy NER for person/place/org, priority-based primary type | Existing rules retained. Without NER many names/places are `other`; ambiguous words such as May and contextual negation need empirical review |
| Existing safeguard | `analysis.py:fit_priors/lookup`: rate denominators, smoothing, minimum-count masks, lexical→type→acoustic fallback; rates not raw word frequency | Verified in static code and synthetic semantic tests; no change needed |
| Existing safeguard | `analysis.py:analyze/freeze`: analysis is retrospective, frozen legacy priors require train provenance and explicit correction-training membership | Keep; GT diagnostics must never be D inference inputs |
| Existing safeguard | `full_train_priors.py:export_features`: train features exclude own group; non-train IDs/group metadata checked; model exposure explicitly in-sample | Keep, but missing external source identities cannot be inferred and statistical count exclusion is not model out-of-fold training |
| Existing safeguard | No direct multiplication of semantic features into SAL real/fake scores found in inspected modules | New D concatenates acoustic and category features, predicts gain; never modifies SAL score probabilities |
| Medium reproducibility limitation | ASR model identifiers/config/runtime exist, but real models, word outputs and per-recording failure recovery were unavailable | New run provenance hashes input transcript/cache/config/manifest/checkpoints; deployment ASR recovery remains outside this implementation |

## Implementation / validation boundary

A preserves baseline intervals. B is a configurable score-evidence rule baseline.
C is an acoustic MLP; D adds a validated transcript-category interface and can train
on synthetic/valid supplied word features. All use the identical GT-free pool.
No full two-stage reinspection system was added. The event risk and metrics are
explicitly named research diagnostics rather than official SAL frame metrics.

Targeted new interval tests and existing semantic tests passed (37 tests before
final compatibility additions). Synthetic CLI fixture/check/split, C and D two-epoch
training, checkpoint safe reload and unified A/B/C/D dev evaluation were executed.
See handoff for final verification command. No performance inference should be
made from deterministic fixture metrics. Full repository testing initially failed
at collection due to missing torchaudio, then soundfile; additional environment
checks and final outcome are reported separately in the final delivery.

Final broader regression attempt: 44 passed and one existing CNN/SAL test failed
because `sal.pipeline` unconditionally imports Windows-only `msvcrt` on this Linux
host (after resolving its test import path). This old platform requirement was
not changed. The standalone correction and semantic tests do not require it.
Real ASR, official metric execution with SAL dependencies, full Windows suite and
real-data training remain unverified. No fake `msvcrt` shim was used.
