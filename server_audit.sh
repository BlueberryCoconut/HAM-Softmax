#!/usr/bin/env bash
set -u

printf '=== host ===\n'
hostname
date --iso-8601=seconds
uname -a

printf '\n=== repository ===\n'
git status --short --branch
git log -3 --oneline --decorate

printf '\n=== python candidates ===\n'
command -v python || true
command -v python3 || true
conda env list 2>/dev/null || true

printf '\n=== CUDA and GPUs ===\n'
nvidia-smi --query-gpu=index,name,uuid,utilization.gpu,memory.used,memory.total --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv 2>/dev/null || true

printf '\n=== possible duplicate jobs ===\n'
pgrep -af 'main(_es_gmm)?\.py' || true

printf '\n=== required list files ===\n'
sha256sum data/v1_clean.txt data/vox1k_sym_0.3.txt data/vox1k_asym_0.2.txt
wc -l data/v1_clean.txt data/vox1k_sym_0.3.txt data/vox1k_asym_0.2.txt

printf '\n=== configured data roots ===\n'
printf 'VOXCELEB1_ROOT=%s\n' "${VOXCELEB1_ROOT:-<unset>}"
printf 'AUGMENTATION_DATA_ROOT=%s\n' "${AUGMENTATION_DATA_ROOT:-<unset>}"

printf '\nAudit only: this script does not start or stop training.\n'
