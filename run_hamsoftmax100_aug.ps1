$ErrorActionPreference = 'Stop'

$python = 'C:\Users\18268\anaconda3\envs\YZH\python.exe'

Write-Output "===== PREPARE AUGMENTATION DATA $(Get-Date -Format o) ====="
& $python prepare_augmentation_data.py
if ($LASTEXITCODE -ne 0) {
    throw "Augmentation data preparation failed with exit code $LASTEXITCODE"
}

Write-Output "===== START HAM-Softmax-100-AUG $(Get-Date -Format o) ====="
& $python main.py `
    --dataset v1 `
    --lambda_1 0 `
    --lambda_2 1 `
    --h_C 3 `
    --h_m 0.2 `
    --h_s 30 `
    --batch_size 64 `
    --n_cpu 0 `
    --max_epoch 100 `
    --test_step 5 `
    --seed 42 `
    --postfix quick100 `
    --augment

if ($LASTEXITCODE -ne 0) {
    throw "HAM-Softmax-100-AUG failed with exit code $LASTEXITCODE"
}

Write-Output "===== DONE HAM-Softmax-100-AUG $(Get-Date -Format o) ====="
