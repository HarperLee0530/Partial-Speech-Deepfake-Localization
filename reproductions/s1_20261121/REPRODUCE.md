# Reproduction instructions

## Inputs and protocol

Use the fixed reduced PartialSpoof TRAIN 6345 and DEV 2484 recordings, with the exact UID order in `data_protocol/manifests/` and the pinned source list `experiments/wavlm_ablation_ab_20261119/code/manifests/ps_fixed_ids.json`. Do not substitute the complete corpus, generate fine labels from coarse labels, or use EVAL to choose thresholds or checkpoints. `experiments/event_aware_c/data_audit.py` extracts native 20ms and 160ms TRAIN/DEV labels from the official **v1.2 segment-label archive**, after verifying the archive hash; the expected label hashes are in `data_protocol/labels/provenance.json` and the audit file. Original dataset audio/label access and WavLM weights must be obtained separately under their respective terms. The 29GB frozen feature cache is not in Git; exact numerical reproduction requires rebuilding the cache to the SHA256 hashes in `protocol.json` or obtaining the original verified cache.

Frozen encoder: `microsoft/wavlm-base-plus`, revision `4c66d4806a428f2e922ccfa1a962776e232d487b`, model SHA256 `3bb273a6ace99408b50cfc81afdbb7ef2de02da2eab0234e18db608ce692fe51`; layers 6/9/12, float32, original stride320/receptive400/sidepad40 grid. `data_protocol/cache_metadata/` and each original `protocol.json` specify expected reference/layer cache hashes, shapes and UID digests. Rebuild using `experiments/event_aware_c/data_audit.py` and `prepare_cache.py` and the shared `cnn_ps` cache code; these scripts enforce the original manifests and hashes and may need local path configuration. Preserve any mismatch rather than asserting equivalence.

Training: seed666/667, 20 formal Epochs, batch8, 794 batches/Epoch, 15,880 optimizer steps; AdamW lr 0.001, weight decay 0.0001, global norm clip5; clean/noise20dB/EQ views rotate by Epoch. Frozen WavLM features, learnable global layer fusion, CNN–BiGRU and dual-resolution architecture. DEV selection: each checkpoint gets its own threshold under real-duration FPR≤5%; maximize Event-F1@IoU0.5 over Epoch0–20, earliest tie. Thresholds act on raw 20ms logits with `score >= threshold`.

## Original training entry points

The commands below show the **original code path** and require a verified cache and GPU. They are documentation; this branch upload did not rerun training. Run in an isolated NNDL workspace whose `repo/` is this branch. The scripts require absolute paths under that workspace and a dedicated `.venv`. Never point `--run` at an existing result directory.

```bash
workspace=/path/to/nndl_partial_speech
cd "$workspace"
gpu_uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1)
export CUDA_VISIBLE_DEVICES="$gpu_uuid"
export PYTHONPATH="$workspace/repo/experiments/cross_resolution_supervision:$workspace/repo/experiments/wavlm_ablation_ab_20261119/code:$workspace/repo/experiments/event_aware_c"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
mkdir -p "$workspace/logs/s1_reproduction" "$workspace/experiments/cross_resolution_supervision"
```

For **seed666**, reproduce the paired A0 with the original `experiments/event_aware_c/trainer.py` using the same verified cache, then run the original S1 entry point. A0 and S1 outputs must go to separate new directories:

```bash
"$workspace/.venv/bin/python" -u "$workspace/repo/experiments/event_aware_c/trainer.py" \
  --workspace "$workspace" --cache "$workspace/cache/fixed_ps" \
  --run "$workspace/experiments/event_aware_c/A0_666_reproduction" \
  --log "$workspace/logs/s1_reproduction/A0_666.log" \
  --arm A0 --seed 666 --gpu-uuid "$gpu_uuid"
```

Then run S1:

```bash
"$workspace/.venv/bin/python" -u "$workspace/repo/experiments/cross_resolution_supervision/trainer.py" \
  --workspace "$workspace" --cache "$workspace/cache/fixed_ps" \
  --run "$workspace/experiments/cross_resolution_supervision/S1_666_reproduction" \
  --log "$workspace/logs/s1_reproduction/S1_666.log" \
  --arm S1 --seed 666 --gpu-uuid "$gpu_uuid"
```

For **seed667**, the actual multiseed trainer explicitly checks that fresh S1 initialization is tensor-identical to `experiments/cross_resolution_supervision/A0_667_formal/epochs/epoch_00.pt`. Train A0_667 with the same trainer first. The original S1 code expects the A0 output directory to have that exact name; do not use it if it already exists. Alternatively, in a fresh workspace restore the bundled A0 epoch0 checkpoint to that exact relative path after SHA256 verification if only reproducing S1. Then run S1 into a new directory:

```bash
"$workspace/.venv/bin/python" -u "$workspace/repo/experiments/cross_resolution_multiseed/trainer.py" \
  --workspace "$workspace" --cache "$workspace/cache/fixed_ps" \
  --run "$workspace/experiments/cross_resolution_supervision/A0_667_formal" \
  --log "$workspace/logs/s1_reproduction/A0_667.log" \
  --arm A0 --seed 667 --gpu-uuid "$gpu_uuid"
```

```bash
"$workspace/.venv/bin/python" -u "$workspace/repo/experiments/cross_resolution_multiseed/trainer.py" \
  --workspace "$workspace" --cache "$workspace/cache/fixed_ps" \
  --run "$workspace/experiments/cross_resolution_supervision/S1_667_reproduction" \
  --log "$workspace/logs/s1_reproduction/S1_667.log" \
  --arm S1 --seed 667 --gpu-uuid "$gpu_uuid"
```

The multiseed trainer has a historical `seed=667` argument restriction; do not silently alter it and claim the implementation hash still matches. Seed666 and seed667 use different **entry-point file hashes**, both stored in their own `protocol.json`; their S1 loss implementation hash is identical. The paired A0/S1 comparison is fair within each seed. The seed666 A0 output was originally in `experiments/event_aware_c/plan_c_666/A0_666_formal/`.

## Verify delivered artifacts

`python3 reproductions/s1_20261121/verify_package.py` checks checkpoint, DEV logits, protocol and implementation hashes and validates stored metadata. For real metric recomputation, use the official fixed DEV fine labels and the original `cnn_ps.metrics20` evaluation. Published EVAL metrics used frozen DEV thresholds; the original evaluation source in `evaluation/` contains the old absolute server root, so path relocation is required on another machine. Do not change its matching or threshold rules while relocating.

The package omits intermediate Epoch checkpoints, raw audio, fine label arrays and feature arrays. Consequently it supports exact provenance checks and inference from best checkpoints once the verified cache exists, but **cannot by itself rerun training or independently score DEV/EVAL**. Distinct CUDA environments may produce small numerical differences; compare hashes when possible and otherwise report numeric deviation without replacing the frozen historical results.
