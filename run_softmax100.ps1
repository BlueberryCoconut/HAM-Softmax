$ErrorActionPreference = 'Stop'

$python = 'C:\Users\18268\anaconda3\envs\YZH\python.exe'

Write-Output "===== START Softmax-100 $(Get-Date -Format o) ====="
& $python main.py `
    --dataset v1 `
    --loss_type ce `
    --lambda_1 1 `
    --lambda_2 0 `
    --batch_size 128 `
    --n_cpu 4 `
    --max_epoch 100 `
    --test_step 5 `
    --seed 42 `
    --postfix quick5_noaug

if ($LASTEXITCODE -ne 0) {
    throw "Softmax-100 failed with exit code $LASTEXITCODE"
}

Write-Output "===== DONE Softmax-100 $(Get-Date -Format o) ====="
