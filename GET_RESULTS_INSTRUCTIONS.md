# How to Get Training Results Summary

Your training was done on Ubuntu, so use these scripts to extract and share the results with me.

## Quick Method (Fast, Text Output)

Run on your Ubuntu device:

```bash
cd /path/to/PET
bash scripts/show_all_results.sh
```

This will print a table of all your training runs with:
- Run folder name
- Backbone used
- Best MAE achieved
- Best epoch
- Directory sizes

**Copy and paste the output to share with me.**

---

## Detailed Method (Full JSON Report)

For comprehensive results:

```bash
cd /path/to/PET
python scripts/summary_all_runs.py
```

This will:
1. Print a formatted table of all runs
2. Show summary statistics by backbone
3. Display the best single run
4. Output full JSON data
5. Save results to `outputs/SHA/SUMMARY.json`

**Copy and paste the output, OR download the JSON file.**

---

## What Information I'll See

The scripts will extract:

```
Run Name:        backbone_name_seed_42
Backbone:        maxvit_small_tf_224
Best MAE:        59.24
Best Epoch:      1250
Threshold:       0.50
Directory Size:  2.5 GB
```

This tells me:
- Which backbones you've tested
- Which one performed best
- Whether training converged (best_epoch tells if it peaked early or late)
- Approximate number of epochs trained

---

## Example Output

When you run it, you'll see something like:

```
Run Name                                           Backbone                   Best MAE     Best Epoch   Threshold   
===================================================================================================================

maxvit_small_tf_224
---------------------------------------------------
maxvit_small_tf_224_seed_42                        maxvit_small_tf_224        59.24        1250         0.500
maxvit_small_tf_224_seed_7                         maxvit_small_tf_224        58.91        1180         0.490

===================================================================================================================
SUMMARY BY BACKBONE
===================================================================================================================

Backbone                                           Best         Worst        Avg          Runs      
maxvit_small_tf_224                               58.91        59.24        59.07        2
```

---

## What to Share With Me

Copy the full output and share it. I need:

1. **For quick check:** Run `bash scripts/show_all_results.sh` and paste output
2. **For full analysis:** Run `python scripts/summary_all_runs.py` and paste output

Or upload `outputs/SHA/SUMMARY.json` if it's easier.

---

## Troubleshooting

**If you get "command not found: bash":**
Use bash explicitly:
```bash
python scripts/summary_all_runs.py
```

**If you get "No training results found":**
Check that your `outputs/SHA` directory has subdirectories with `run_log.txt` files:
```bash
ls -la outputs/SHA/
```

**If paths are wrong:**
Make sure you're running from the PET root directory:
```bash
cd /path/to/PET
pwd  # Verify you're in the right place
python scripts/summary_all_runs.py
```

---

## Next Steps After Sharing Results

Once you share the output, I'll:
1. See which backbone(s) you've tested
2. Identify the best single backbone
3. Understand current MAE range
4. Know what data path to use for multi-seed ensemble
5. Create proper ensemble runs for you

So **go run one of these scripts now on your Ubuntu device and paste the output here!** ✨
