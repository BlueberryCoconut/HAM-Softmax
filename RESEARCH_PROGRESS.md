# HAM-Softmax Research Progress and Server Handoff

Snapshot date: 2026-09-29 (local Windows workspace)

This file records evidence for handoff. Live server state always takes precedence over this snapshot.

## Repository state

- Repository: `D:\新大语音科研\HAM-Softmax`
- Base commit: `7060764` (`update`), branch `main` tracking `origin/main`.
- The worktree is intentionally dirty. Existing modifications include `datasets.py`, `main.py`, and `utils.py`, plus untracked ES-GMM code, launchers, data lists, logs, and utilities.
- Do not use destructive Git cleanup or overwrite these files. The server handoff must include the working-tree versions, not only the upstream commit.

## Dataset and local environment

- Local VoxCeleb1 root: `D:\新大语音科研\voxceleb1`.
- Previously verified local counts: 148,642 development utterances and 4,874 test WAV/TXT files.
- Local Python: `C:\Users\18268\anaconda3\envs\YZH\python.exe`.
- Local GPU: RTX 4070 Laptop GPU. This path and hardware information are Windows-only and must not be copied into a Linux launch command.
- Training uses VoxCeleb1 and, when `--augment` is enabled, MUSAN/RIR augmentation data. Server paths remain unknown until verified live.

## Established HAM-Softmax results

### HAM-Softmax + MUSAN without ES-GMM

| Condition | Status | EER (%) | minDCF |
|---|---|---:|---:|
| clean | complete | 2.415 | 0.183 |
| sym03 (symmetric label noise 0.3) | complete | 7.496 | 0.442 |
| asym02 (asymmetric label noise 0.2) | complete | 5.303 | 0.356 |

### HAM-Softmax + MUSAN + ES-GMM

| Condition | Status | EER (%) | minDCF (0.05) |
|---|---|---:|---:|
| sym03 | complete | 3.403 | 0.221 |
| asym02 | complete | 3.516 | 0.273 |
| clean | earlier snapshot was incomplete; re-audit original logs before publishing | - | - |

These runs are distinct from the newer AAM+ES-GMM runs below.

## Current AAM+ES-GMM evidence

Training entry point: `main_es_gmm.py`

Common intended arguments from `run_vox1k_aam_esgmm.ps1`:

```text
--dataset v1
--batch_size 64
--n_cpu 0
--max_epoch 100
--test_step 5
--warmup_epochs 5
--m 0.2
--s 30
--loss_type aam
--augment
--seed 42
```

### sym03

- Experiment directory: `exps/v1_esgmm_aam_m=0.2_s=30.0_augment_aam_esgmm_vox1k_sym03_100`
- Completed Epoch 100.
- Final evaluation recorded 2026-09-24: EER 3.239, minDCF(0.05) 0.223, bestEER 3.206.
- `model_0100.model` exists; a later evaluation-only resume created `model_final.model`.
- Historical stdout from the evaluation-only run printed `train_list: data/v1_clean.txt`, while `train.log` records `train_list=data\vox1k_sym_0.3.txt`. Treat the train log plus the actual list argument as stronger evidence, but audit the mismatch before reproducing.

### asym02: interrupted/partial run to audit and resume

- Experiment directory: `exps/v1_esgmm_aam_m=0.2_s=30.0_augment_aam_esgmm_vox1k_asym02_100`
- Latest complete checkpoint observed locally: `model/model_0052.model`.
- `model_0052.model` size: 63,588,285 bytes, matching adjacent numbered checkpoints.
- Epoch 52 completed at 2026-09-26 21:25:42.
- The log then performed ES-GMM selection and entered `Epoch 53` at 2026-09-26 21:46:17, but no `model_0053.model` was found. Therefore Epoch 53 is not considered complete.
- Latest recorded periodic metric at Epoch 50: ACC 93.613, EER 3.409, minDCF 0.254, bestEER 3.286.
- Historical stdout for a run with an asym02 postfix printed `train_list: data/v1_clean.txt`. This is a blocking configuration discrepancy: inspect `run_vox1k_aam_esgmm.ps1`, the actual list files, `main_es_gmm.py`, and any resume command before continuing.
- If the configuration is validated and the experiment directory is transferred intact, current `main_es_gmm.py` should discover `model_0052.model` and begin at Epoch 53. Verify this behavior in a short controlled launch before leaving a long-running job unattended.

## Server snapshot supplied by the user

- `50030` maps to host `x1`; all eight RTX 3090 GPUs were occupied at the time checked.
- `50040` maps to host `x2`.
- On x2, physical GPU 6 was the only apparently free GPU: 0% utilization, 2 MiB used, and no compute process at the time checked.
- This is not a reservation. Re-run `nvidia-smi` immediately before launch.
- x2 previously reported system load above 96, so CPU/data-loader and storage contention must be checked even if GPU 6 remains free.

## Required server audit before any resume

1. Confirm the transferred working tree contains the local modified/untracked code required by the experiment.
2. Locate the Linux Python/Conda environment and verify PyTorch sees CUDA.
3. Locate VoxCeleb1, MUSAN, and RIR data on the server; verify expected structure and representative files.
4. Compare the contents of `data/vox1k_asym_0.2.txt`, `data/vox1k_sym_0.3.txt`, and `data/v1_clean.txt`; resolve the train-list discrepancy.
5. Check `ps`, `pgrep -af`, `nvidia-smi`, and recent logs for an existing copy of this experiment.
6. Verify physical GPU 6 is still free. If using `CUDA_VISIBLE_DEVICES=6`, use logical `cuda:0` inside the process.
7. Verify `model_0052.model` exists, has the expected size, is stable, and loads successfully.
8. Produce the exact Linux resume command and expected log/checkpoint paths for review before execution.
9. Launch inside `tmux` or another approved persistent mechanism only after the audit passes.
10. After launch, confirm PID, command line, log growth, checkpoint target, VRAM allocation, GPU utilization, and absence of a duplicate process.

## Definition of a successful continuation

- The process remains alive after the initial data/model load.
- The log explicitly reports resuming from `model_0052.model` and starting Epoch 53.
- Training batches advance, not merely allocate VRAM.
- GPU 6 shows expected memory and compute activity without affecting other users.
- The next complete checkpoint is written with a stable, expected size.
- The exact command and evidence are appended to this document.
