$ErrorActionPreference = 'Stop'

$python = 'C:\Users\18268\anaconda3\envs\YZH\python.exe'
$common = @(
    'main.py',
    '--dataset', 'v1',
    '--batch_size', '128',
    '--n_cpu', '4',
    '--max_epoch', '5',
    '--test_step', '5',
    '--seed', '42',
    '--postfix', 'quick5_noaug'
)

$experiments = @(
    @('--loss_type', 'ce', '--lambda_1', '1', '--lambda_2', '0'),
    @('--lambda_1', '0', '--lambda_2', '1', '--h_C', '5', '--h_s', '30', '--h_m', '0'),
    @('--loss_type', 'am', '--lambda_1', '1', '--lambda_2', '0', '--m', '0.2', '--s', '30'),
    @('--lambda_1', '0', '--lambda_2', '1', '--h_C', '3', '--h_s', '30', '--h_m', '0.2')
)

$names = @('Softmax', 'H-Softmax', 'AM-Softmax', 'HAM-Softmax')

for ($i = 0; $i -lt $experiments.Count; $i++) {
    Write-Output "===== START $($names[$i]) $(Get-Date -Format o) ====="
    & $python @common @($experiments[$i])
    if ($LASTEXITCODE -ne 0) {
        throw "$($names[$i]) failed with exit code $LASTEXITCODE"
    }
    Write-Output "===== DONE $($names[$i]) $(Get-Date -Format o) ====="
}

Write-Output "===== ALL EXPERIMENTS COMPLETE $(Get-Date -Format o) ====="
