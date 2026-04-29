@echo off
REM Batch script to run backbone sweep on Windows PowerShell
REM 
REM Usage:
REM   .\scripts\backbone_sweep.ps1
REM 
REM This script runs multiple backbones with 1 seed each for quick comparison

$backbones = @(
    "swinv2_base_window8_256",
    "convnextv2_base",
    "maxvit_rmlp_tiny_rw_256"
)

$seed = 42
$epochs = 1500
$patch_size = 256

Write-Host "========================================" -ForegroundColor Green
Write-Host "PET Backbone Sweep (Quick)" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Backbones to test: $($backbones -join ', ')" -ForegroundColor Yellow
Write-Host "Seed: $seed" -ForegroundColor Yellow
Write-Host "Epochs: $epochs" -ForegroundColor Yellow
Write-Host ""

$results = @()

foreach ($backbone in $backbones) {
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Testing: $backbone" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    
    $output_dir = "backbone_$($backbone)_seed_$seed"
    $cmd = "python main.py --backbone $backbone --seed $seed --epochs $epochs --output_dir $output_dir --patch_size $patch_size"
    
    Write-Host "Command: $cmd" -ForegroundColor White
    Write-Host ""
    
    & cmd /c $cmd
    
    if ($LASTEXITCODE -eq 0) {
        # Try to extract best MAE from log
        $log_file = "outputs\SHA\$output_dir\run_log.txt"
        if (Test-Path $log_file) {
            $log_content = Get-Content $log_file
            $best_mae_match = $log_content | Select-String -Pattern "best mae:[\s]+([0-9.]+)" | Select-Object -First 1
            if ($best_mae_match) {
                $best_mae = $best_mae_match.Matches[0].Groups[1].Value
                $results += @{
                    backbone = $backbone
                    status = "SUCCESS"
                    best_mae = $best_mae
                }
                Write-Host "✓ $backbone - Best MAE: $best_mae" -ForegroundColor Green
            }
        }
    } else {
        $results += @{
            backbone = $backbone
            status = "FAILED"
            best_mae = "N/A"
        }
        Write-Host "✗ $backbone - Training failed" -ForegroundColor Red
    }
    
    Write-Host ""
}

Write-Host "========================================" -ForegroundColor Green
Write-Host "Backbone Sweep Summary" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

foreach ($result in $results) {
    if ($result.status -eq "SUCCESS") {
        Write-Host "$($result.backbone) : MAE=$($result.best_mae)" -ForegroundColor Green
    } else {
        Write-Host "$($result.backbone) : FAILED" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Pick the backbone with lowest MAE" -ForegroundColor Yellow
Write-Host "2. Run ensemble with multiple seeds: python scripts/run_backbone_seeds.py --backbone <best> --seeds 42 7 13 99 1234" -ForegroundColor Yellow
Write-Host "3. Evaluate ensemble: python scripts/ensemble_evaluate.py --backbone <best> --checkpoints ..." -ForegroundColor Yellow
Write-Host ""
