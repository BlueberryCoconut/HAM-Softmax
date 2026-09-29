$ErrorActionPreference = 'Stop'

$python = 'C:\Users\18268\anaconda3\envs\YZH\python.exe'

Write-Output "===== START H-Softmax-100 $(Get-Date -Format o) ====="
& $python main.py `
    --dataset v1 `
    --lambda_1 0 `
    --lambda_2 1 `
    --h_C 5 `
    --h_m 0 `
    --h_s 30 `
    --batch_size 128 `
    --n_cpu 4 `
    --max_epoch 100 `
    --test_step 5 `
    --seed 42 `
    --postfix quick100_noaug

if ($LASTEXITCODE -ne 0) {
    throw "H-Softmax-100 failed with exit code $LASTEXITCODE"
}

Write-Output "===== DONE H-Softmax-100 $(Get-Date -Format o) ====="
