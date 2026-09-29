# Python/tqdm writes normal progress messages to stderr. PowerShell must not
# convert those messages into terminating NativeCommandError exceptions.
$ErrorActionPreference = 'Continue'

$repo = 'D:\新大语音科研\HAM-Softmax'
$python = 'C:\Users\18268\anaconda3\envs\YZH\python.exe'
$runs = @(
    @{ List = 'data\vox1k_sym_0.3.txt';  Postfix = 'vox1k_sym03_quick100' },
    @{ List = 'data\vox1k_asym_0.2.txt'; Postfix = 'vox1k_asym02_quick100' }
)

Set-Location -LiteralPath $repo

foreach ($run in $runs) {
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $stdout = Join-Path $repo ("ham_{0}_{1}_stdout.log" -f $run.Postfix, $stamp)
    $stderr = Join-Path $repo ("ham_{0}_{1}_stderr.log" -f $run.Postfix, $stamp)

    & $python main.py `
        --dataset v1 `
        --train_list $run.List `
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
        --postfix $run.Postfix `
        --augment `
        1> $stdout 2> $stderr

    if ($LASTEXITCODE -ne 0) {
        throw "Training failed for $($run.Postfix) with exit code $LASTEXITCODE. See $stderr"
    }
}
