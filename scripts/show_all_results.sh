#!/bin/bash
# Quick script to show all training results on Ubuntu

echo "=========================================="
echo "PET Training Results Summary"
echo "=========================================="
echo ""

# Check if outputs/SHA exists
if [ ! -d "outputs/SHA" ]; then
    echo "❌ Error: outputs/SHA directory not found"
    echo ""
    echo "Available directories:"
    if [ -d "outputs" ]; then
        ls -la outputs/
    else
        echo "No outputs directory found"
    fi
    exit 1
fi

# Show folder structure
echo "📁 Outputs Directory Structure:"
echo ""
du -sh outputs/SHA/* 2>/dev/null | sort -h

echo ""
echo "=========================================="
echo "📊 Extracting Training Metrics"
echo "=========================================="
echo ""

# For each subdirectory in outputs/SHA
for dir in outputs/SHA/*/; do
    dirname=$(basename "$dir")
    logfile="${dir}run_log.txt"
    
    if [ -f "$logfile" ]; then
        echo "🔍 $dirname"
        
        # Extract best MAE (look for the pattern)
        best_mae=$(grep -i "best mae:" "$logfile" | tail -1 | grep -oP 'best mae:\s+\K[0-9.]+' || echo "N/A")
        best_epoch=$(grep -i "best epoch:" "$logfile" | tail -1 | grep -oP 'best epoch:\s+\K[0-9]+' || echo "N/A")
        best_threshold=$(grep -i "best threshold:" "$logfile" | tail -1 | grep -oP 'best threshold:\s+\K[0-9.]+' || echo "N/A")
        
        # Extract backbone
        backbone=$(grep "backbone=" "$logfile" | head -1 | grep -oP "backbone='?\K[^'\"]+(?=['\"])" || echo "N/A")
        
        # Count total lines (approximation of training progress)
        lines=$(wc -l < "$logfile")
        
        echo "   Backbone:    $backbone"
        echo "   Best MAE:    $best_mae"
        echo "   Best Epoch:  $best_epoch"
        echo "   Threshold:   $best_threshold"
        echo "   Log Lines:   $lines"
        echo ""
    fi
done

echo "=========================================="
echo "✅ Full Summary Table:"
echo "=========================================="
echo ""

# Create a simple table
printf "%-50s %-30s %-12s %-12s\n" "Run" "Backbone" "Best MAE" "Best Epoch"
printf "%-50s %-30s %-12s %-12s\n" "---" "---" "---" "---"

for dir in outputs/SHA/*/; do
    dirname=$(basename "$dir")
    logfile="${dir}run_log.txt"
    
    if [ -f "$logfile" ]; then
        best_mae=$(grep -i "best mae:" "$logfile" | tail -1 | grep -oP 'best mae:\s+\K[0-9.]+' || echo "N/A")
        best_epoch=$(grep -i "best epoch:" "$logfile" | tail -1 | grep -oP 'best epoch:\s+\K[0-9]+' || echo "N/A")
        backbone=$(grep "backbone=" "$logfile" | head -1 | grep -oP "backbone='?\K[^'\"]+(?=['\"])" || echo "unknown")
        
        printf "%-50s %-30s %-12s %-12s\n" "$dirname" "$backbone" "$best_mae" "$best_epoch"
    fi
done

echo ""
echo "=========================================="
echo "📋 Command to view full JSON summary:"
echo "=========================================="
echo ""
echo "python scripts/summary_all_runs.py"
echo ""
echo "Or to view a specific run's log:"
echo "cat outputs/SHA/<run-name>/run_log.txt"
echo ""
