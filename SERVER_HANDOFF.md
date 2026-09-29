# Server Handoff Checklist

## What to transfer

The authoritative source snapshot is the Git branch `research/server-handoff-20260929`. Verify the commit and list hashes against `RESEARCH_PROGRESS.md` after transfer.

Transfer the working-tree code and small configuration/list files, including:

- `AGENTS.md`
- `RESEARCH_PROGRESS.md`
- `SERVER_HANDOFF.md`
- all project Python source files needed by `main_es_gmm.py`
- `data/vox1k_sym_0.3.txt`
- `data/vox1k_asym_0.2.txt`
- `data/v1_clean.txt`
- the relevant launch scripts as historical argument references
- for asym02 continuation, the experiment log plus `model_0052.model` and enough adjacent checkpoint metadata to validate it
- `environment.yml` and `server_audit.sh`

Do not blindly upload the local VoxCeleb1 or augmentation archives. Prefer the server's existing datasets after verifying their paths and contents.

Run `bash server_audit.sh` from the repository root for the first read-only server inventory. It does not launch or stop training.

## First instruction for server Codex

Paste the following after starting Codex from the transferred repository root:

```text
Read AGENTS.md, RESEARCH_PROGRESS.md, and SERVER_HANDOFF.md completely. You are taking over a transferred Windows research workspace on Linux. First perform a read-only audit; do not launch, stop, rename, delete, or modify training state yet.

Confirm the current host, repository path, Git/worktree state, Python/Conda environments, PyTorch/CUDA availability, dataset paths, augmentation paths, live processes, full process command lines, GPU ownership, logs, and checkpoints. Pay special attention to the mismatch where some AAM+ES-GMM sym03/asym02 stdout headers reported data/v1_clean.txt. Determine the actual intended and effective train list from source, list contents, experiment logs, and launch arguments before proposing a resume.

For the asym02 continuation, verify whether model_0052.model is the newest complete checkpoint and whether Epoch 53 was incomplete. Recheck physical GPU 6 immediately; 0% utilization alone is not enough, so inspect VRAM and compute processes. If CUDA_VISIBLE_DEVICES=6 will be used, plan for logical cuda:0 inside the process.

Return: (1) evidence-backed current status, (2) blockers or discrepancies, (3) the exact proposed Linux resume command, (4) the persistent tmux/logging plan, and (5) post-launch verification commands. Do not execute the resume until I explicitly approve the proposed command.
```
## Approval instruction after the audit

Only after reviewing the audit, use:

```text
The audit is approved. Execute only the exact reviewed resume plan. Start it in a persistent tmux session, record the PID, full command, physical and logical GPU IDs, checkpoint, log path, and start time in RESEARCH_PROGRESS.md. Then verify process liveness, log progress, VRAM, GPU utilization, and that no duplicate process exists. Stop and report instead of improvising if the observed state differs from the approved plan.
```

