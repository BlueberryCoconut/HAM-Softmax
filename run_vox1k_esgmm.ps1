$ErrorActionPreference = 'Continue'

$repo = 'D:\新大语音科研\HAM-Softmax'
$python = 'C:\Users\18268\anaconda3\envs\YZH\python.exe'
$runs = @(
    @{ List = 'data\vox1k_sym_0.3.txt';  Postfix = 'esgmm_vox1k_sym03_100' },
    @{ List = 'data\vox1k_asym_0.2.txt'; Postfix = 'esgmm_vox1k_asym02_100' },
    @{ List = 'data\v1_clean.txt';        Postfix = 'esgmm_v1_clean_100' }
)

Set-Location -LiteralPath $repo
foreach ($run in $runs) {
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $stdout = Join-Path $repo ("esgmm_{0}_{1}_stdout.log" -f $run.Postfix, $stamp)
    $stderr = Join-Path $repo ("esgmm_{0}_{1}_stderr.log" -f $run.Postfix, $stamp)
    & $python main_es_gmm.py `
        --dataset v1 `
        --train_list $run.List `
        --batch_size 64 `
        --n_cpu 0 `
        --max_epoch 100 `
        --test_step 5 `
        --warmup_epochs 5 `
        --m 0.2 `
        --s 30 `
        --seed 42 `
        --postfix $run.Postfix `
        1> $stdout 2> $stderr
    if ($LASTEXITCODE -ne 0) {
        throw "ES-GMM training failed for $($run.Postfix) with exit code $LASTEXITCODE. See $stderr"
    }
}
