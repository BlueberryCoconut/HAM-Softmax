# HAM-Softmax Research Operating Rules

## Safety first

- Treat this repository as a research workspace with valuable uncommitted changes. Do not reset, clean, checkout over, delete, or bulk-rewrite existing files.
- Before starting, resuming, stopping, or replacing a training job, inspect the live process list, its full command line, GPU ownership, recent log growth, and checkpoints.
- Never start a duplicate of a healthy training process.
- Never stop another user's process or use a GPU that already has a compute process or substantial allocated memory.
- On the current shared server, GPU availability is live state. Recheck it immediately before every launch; a previously free GPU is not a reservation.
- Report the evidence and exact proposed command before making a consequential training-state change.

## Checkpoint and result rules

- Resume only from the newest complete checkpoint. A log line entering the next epoch does not prove that epoch completed.
- Compare checkpoint sizes with adjacent known-good checkpoints and check that the file is stable before loading it.
- Do not present a smoke test, a partial epoch, or a published paper metric as a locally reproduced final result.
- Preserve original logs and checkpoints. Write new run output to a new timestamped log or an explicitly agreed continuation log.
- After a launch or resume, record the command, PID, physical GPU, visible CUDA index, checkpoint, log path, start time, and first evidence of progress in `RESEARCH_PROGRESS.md`.

## GPU conventions

- When launching with `CUDA_VISIBLE_DEVICES=<physical index>`, code inside the process should normally use `cuda:0`.
- A GPU at 0% utilization is not necessarily free; allocated VRAM and compute-process ownership are decisive.
- Prefer PyTorch DistributedDataParallel for multi-GPU work only when the code explicitly supports it. Do not assume that listing several GPUs enables multi-GPU training.

## Project facts and portability

- The local Windows dataset root is outside this repository at `D:\新大语音科研\voxceleb1`; the server path is intentionally not assumed.
- The local Windows Python interpreter is `C:\Users\18268\anaconda3\envs\YZH\python.exe`; do not reuse that path on Linux.
- Windows `.ps1` launchers are historical evidence and are not directly executable on Linux. Translate their arguments into a reviewed Linux command or shell script.
- Before server training, locate and validate the server's VoxCeleb1, MUSAN, and RIR paths and verify required file counts/samples rather than editing paths blindly.
- `main_es_gmm.py` automatically discovers `model_0*.model` in the target experiment directory and resumes from the lexicographically latest numbered checkpoint.

## Required checks for the current handoff

- Read `RESEARCH_PROGRESS.md` before proposing work.
- The latest local AAM+ES-GMM asym02 run name and its apparent data condition disagree in historical stdout: the run/postfix says `asym02`, while one stdout header says `train_list: data/v1_clean.txt`. The current `train.log` later records ES-GMM behavior consistent with a noisy run, but this discrepancy must be resolved from the actual command, source, and list contents before resuming on the server.
- Do not claim that a Windows process is still alive from old logs. Server process state must be checked live.
