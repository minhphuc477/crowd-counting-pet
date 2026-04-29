#!/usr/bin/env python3
"""
Extract and display all training results from outputs folder.

Usage:
    python scripts/summary_all_runs.py

This will print a table of all training runs with their best MAE, best epoch, and other metrics.
"""

import os
import json
import glob
from pathlib import Path
from collections import defaultdict


def extract_best_mae_from_log(log_file):
    """Extract best_mae and best_epoch from run_log.txt"""
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Look for patterns like: best mae: 59.24, best epoch: 1250
        lines = content.split('\n')
        best_mae = None
        best_epoch = None
        best_threshold = None
        
        for line in reversed(lines):  # Search from end for latest values
            if 'best mae:' in line.lower():
                try:
                    # Parse: "best mae: 59.24, best epoch: 1250, best threshold: 0.5"
                    parts = line.split(',')
                    for part in parts:
                        if 'best mae' in part.lower():
                            best_mae = float(part.split(':')[1].strip())
                        elif 'best epoch' in part.lower():
                            best_epoch = int(part.split(':')[1].strip())
                        elif 'best threshold' in part.lower():
                            best_threshold = float(part.split(':')[1].strip())
                    if best_mae is not None:
                        break
                except:
                    pass
        
        return best_mae, best_epoch, best_threshold
    except:
        return None, None, None


def extract_backbone_from_args(log_file):
    """Extract backbone name from args in log"""
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Look for backbone='swinv2_base_window8_256'
        for line in content.split('\n'):
            if "backbone=" in line:
                try:
                    start = line.find("backbone='") + len("backbone='")
                    end = line.find("'", start)
                    if start > 0 and end > start:
                        return line[start:end]
                except:
                    pass
        return None
    except:
        return None


def main():
    outputs_base = Path('./outputs/SHA')
    
    if not outputs_base.exists():
        print(f"Error: {outputs_base} does not exist")
        print("Available outputs paths:")
        if Path('./outputs').exists():
            for item in Path('./outputs').iterdir():
                print(f"  - {item}")
        return
    
    print("\n" + "="*100)
    print("PET TRAINING RESULTS SUMMARY")
    print("="*100 + "\n")
    
    # Collect all results
    results = []
    
    # Find all run directories
    for run_dir in sorted(outputs_base.iterdir()):
        if run_dir.is_dir():
            log_file = run_dir / 'run_log.txt'
            
            if log_file.exists():
                best_mae, best_epoch, best_threshold = extract_best_mae_from_log(log_file)
                backbone = extract_backbone_from_args(log_file)
                
                results.append({
                    'run_name': run_dir.name,
                    'backbone': backbone or 'unknown',
                    'best_mae': best_mae,
                    'best_epoch': best_epoch,
                    'best_threshold': best_threshold,
                    'log_path': str(log_file),
                })
    
    if not results:
        print("No training results found in outputs/SHA/")
        return
    
    # Sort by backbone, then by MAE
    results.sort(key=lambda x: (x['backbone'], x['best_mae'] if x['best_mae'] else float('inf')))
    
    # Print table
    print(f"{'Run Name':<50} {'Backbone':<30} {'Best MAE':<12} {'Best Epoch':<12} {'Threshold':<12}")
    print("-" * 100)
    
    current_backbone = None
    for result in results:
        if result['backbone'] != current_backbone:
            current_backbone = result['backbone']
            print(f"\n{result['backbone']}")
            print("-" * 100)
        
        mae_str = f"{result['best_mae']:.2f}" if result['best_mae'] is not None else "N/A"
        epoch_str = str(result['best_epoch']) if result['best_epoch'] is not None else "N/A"
        thresh_str = f"{result['best_threshold']:.3f}" if result['best_threshold'] is not None else "N/A"
        
        print(f"{result['run_name']:<50} {result['backbone']:<30} {mae_str:<12} {epoch_str:<12} {thresh_str:<12}")
    
    print("\n" + "="*100)
    print("SUMMARY BY BACKBONE")
    print("="*100 + "\n")
    
    # Group by backbone
    by_backbone = defaultdict(list)
    for result in results:
        if result['best_mae'] is not None:
            by_backbone[result['backbone']].append(result['best_mae'])
    
    # Print stats by backbone
    print(f"{'Backbone':<40} {'Best':<12} {'Worst':<12} {'Avg':<12} {'Runs':<10}")
    print("-" * 100)
    
    for backbone in sorted(by_backbone.keys()):
        maes = sorted(by_backbone[backbone])
        best = min(maes)
        worst = max(maes)
        avg = sum(maes) / len(maes)
        
        print(f"{backbone:<40} {best:<12.2f} {worst:<12.2f} {avg:<12.2f} {len(maes):<10}")
    
    print("\n" + "="*100)
    print("BEST SINGLE RUN")
    print("="*100 + "\n")
    
    best_result = min([r for r in results if r['best_mae'] is not None], key=lambda x: x['best_mae'])
    print(f"Run Name:    {best_result['run_name']}")
    print(f"Backbone:    {best_result['backbone']}")
    print(f"Best MAE:    {best_result['best_mae']:.2f}")
    print(f"Best Epoch:  {best_result['best_epoch']}")
    print(f"Threshold:   {best_result['best_threshold']:.3f}")
    print(f"Log Path:    {best_result['log_path']}")
    
    print("\n" + "="*100)
    print("FULL OUTPUT (JSON)")
    print("="*100 + "\n")
    
    # Print as JSON for easy parsing
    output_data = {
        'summary_by_backbone': {
            backbone: {
                'best_mae': min(maes),
                'worst_mae': max(maes),
                'avg_mae': sum(maes) / len(maes),
                'num_runs': len(maes),
                'all_maes': sorted(maes),
            }
            for backbone, maes in by_backbone.items()
        },
        'all_runs': results,
        'best_run': best_result,
    }
    
    print(json.dumps(output_data, indent=2, default=str))
    
    # Also save to file
    output_file = Path('./outputs/SHA/SUMMARY.json')
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"\n✓ Summary also saved to: {output_file}")


if __name__ == '__main__':
    main()
