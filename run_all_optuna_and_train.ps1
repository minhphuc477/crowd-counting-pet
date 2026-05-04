# Optuna search + training for all backbones
$backbones = @(
  'convnext_tiny',
  'convnext_base',
  'convnextv2_tiny',
  'convnextv2_base',
  'swinv2_tiny',
  'swinv2_base',
  'maxvit_tiny',
  'maxvit_small',
  'maxvit_rmlp_tiny',
  'fastvit_tiny',
  'fastvit_small',
  'efficientvit_tiny',
  'efficientvit_small',
  'efficientnetv2_tiny',
  'efficientnetv2_small',
  'mobilenetv4_small',
  'mobilenetv4_hybrid',
  'hgnetv2_tiny',
  'hgnetv2_small',
  'pvtv2_b0',
  'pvtv2_b1',
  'edgenext_tiny',
  'edgenext_small',
  'repvit_tiny',
  'repvit_small'
)

foreach ($backbone in $backbones) {
  Write-Host "========================================" -ForegroundColor Green
  Write-Host "Starting Optuna search for $backbone..." -ForegroundColor Yellow
  Write-Host "========================================" -ForegroundColor Green
  python scripts/optuna_search.py --backbone $backbone --trials 20 --seeds 42 7 13 --output_dir results
  
  Write-Host "========================================" -ForegroundColor Green
  Write-Host "Starting training for $backbone..." -ForegroundColor Yellow
  Write-Host "========================================" -ForegroundColor Green
  python main.py --backbone $backbone --epochs 150 --patch_size 256 --seed 42 --output_dir results/$backbone/final_train
  
  Write-Host "Completed $backbone`n" -ForegroundColor Cyan
}

Write-Host "All backbones completed!" -ForegroundColor Green
