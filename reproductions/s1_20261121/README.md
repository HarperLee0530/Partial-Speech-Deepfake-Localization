# S1 reproducibility package (research candidate)

**Status:** S1 is an evaluated research candidate, **not** the project's current main baseline. This branch preserves the original 20-Epoch A0/S1 comparison; it does not replace `main` or choose a new checkpoint using EVAL.

S1 changes exactly one coefficient in the A0 training objective:

| Term | A0 | S1 |
|---|---:|---:|
| 20ms BCE on final logits | 1.0 | 1.0 |
| Main 160ms BCE on mean-of-eight final 20ms logits | 1.0 | **0.0** |
| Auxiliary 160ms BCE on the separate coarse output | 0.5 | 0.5 |
| Boundary loss | 0.1 | 0.1 |
| Consistency loss | 0.05 | 0.05 |

The WavLM Base Plus 6/9/12 frozen features, trainable global layer fusion, CNN–BiGRU, two-resolution network, attention/gate, data order, augmentation, optimizer, inference, and DEV model selection are unchanged. S1 still uses coarse supervision through the auxiliary branch. C1/C2 are disabled.

## Frozen 20-Epoch results

| Seed | Model | Best DEV epoch | DEV Event-F1@IoU0.5 | Frozen DEV logit threshold | EVAL Event-F1@IoU0.5 |
|---:|---|---:|---:|---:|---:|
| 666 | A0 | 11 | 83.9126% | 0.18779975175857547 | 72.0979% |
| 666 | S1 | 17 | **84.6821%** | -0.6446576714515685 | **73.4801%** |
| 667 | A0 | 19 | 84.2982% | 0.021871685981750492 | 73.6296% |
| 667 | S1 | 19 | **84.9790%** | -0.712909162044525 | **73.7267%** |

DEV S1−A0: +0.7695 and +0.6807 percentage points. EVAL S1−A0: +1.3822 and +0.0971 percentage points. EVAL was used in earlier research and is **not a wholly untouched blind test**. Seed667 EVAL gain is close to zero; two seeds cannot establish general stability. Full precision and error analyses are in [`evaluation/`](evaluation/).

## Contents

- [`artifacts/`](artifacts/): A0/S1 best checkpoints, original `complete.json`, `protocol.json`, 21-row `history.json`, DEV logits; A0 epoch0 checkpoints for initialization checks. Training logs and all intermediate epochs remain on the original server and are not bundled.
- [`data_protocol/`](data_protocol/): fixed TRAIN/DEV UID manifests, audit and cache metadata hashes. No source audio, fine-label arrays, pretrained WavLM weights, or 29GB feature cache are redistributed.
- [`evaluation/`](evaluation/): frozen four-model EVAL report, complete metrics, paired comparisons, uncertainty intervals, audit and the actual evaluation implementation. That implementation contains original server paths and requires adaptation to a different workspace without altering evaluation semantics.
- [`ENVIRONMENT.md`](ENVIRONMENT.md): actual AutoDL container hardware, quotas, software versions and limits of hardware identification.
- [`REPRODUCE.md`](REPRODUCE.md): prerequisites, code entry points, exact training commands and validation rules.
- [`verify_package.py`](verify_package.py): read-only SHA256 and result metadata checks.

The model implementation reused from the parent commit is in `experiments/wavlm_ablation_ab_20261119/code/cnn_ps/`. The actual seed666 S1 entry point is `experiments/cross_resolution_supervision/trainer.py`; seed667 uses `experiments/cross_resolution_multiseed/trainer.py` with the same `supervision.py`. The original A0 seed666 entry point is `experiments/event_aware_c/trainer.py`. All are copied from the actual server files; `protocol.json` records their SHA256. `experiments/event_aware_c/data_audit.py` and `prepare_cache.py` are included for data/cache preparation.

Run `python3 reproductions/s1_20261121/verify_package.py` after checkout to validate transferred files. It does not train or score EVAL.
