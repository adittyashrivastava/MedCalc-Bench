#!/bin/bash

# MedCalc-Bench Parallel Processing Script for Babel
# This script partitions the first 100 examples and launches parallel jobs

set -e  # Exit on any error

echo "🚀 MedCalc-Bench Parallel Processing Setup"
echo "=========================================="

# Configuration
MAX_EXAMPLES=100
NUM_PARTITIONS=4
MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
PROMPT="zero_shot"
ENABLE_ATTENTION="--enable_attention_analysis"

# Slurm job configuration
PARTITION="debug"
GPUS_PER_JOB=1
CPUS_PER_JOB=4
MEMORY_PER_JOB="16G"
TIME_LIMIT="01:00:00"

echo "Configuration:"
echo "  Max examples: $MAX_EXAMPLES"
echo "  Number of partitions: $NUM_PARTITIONS"
echo "  Model: $MODEL"
echo "  Prompt style: $PROMPT"
echo "  Attention analysis: $([ -n "$ENABLE_ATTENTION" ] && echo "enabled" || echo "disabled")"
echo "  Slurm partition: $PARTITION"
echo ""

# Step 1: Partition the dataset
echo "📊 Step 1: Partitioning dataset..."
python partition_dataset.py \
    --dataset ../dataset/test_data.csv \
    --num_partitions $NUM_PARTITIONS \
    --max_examples $MAX_EXAMPLES \
    --output_dir partitions

if [ ! -f "partitions/partition_info.txt" ]; then
    echo "❌ Error: Partition info file not created"
    exit 1
fi

echo "✅ Dataset partitioned successfully"
echo ""

# Step 2: Create job scripts and submit them
echo "🎯 Step 2: Creating and submitting parallel jobs..."

# Create jobs directory
mkdir -p jobs
mkdir -p logs

# Read partition information and create job scripts
JOB_IDS=()

for i in $(seq 0 $((NUM_PARTITIONS-1))); do
    PARTITION_ID=$(printf "p%02d" $i)
    
    # Calculate start and end indices for this partition
    EXAMPLES_PER_PARTITION=$(( (MAX_EXAMPLES + NUM_PARTITIONS - 1) / NUM_PARTITIONS ))
    START_IDX=$((i * EXAMPLES_PER_PARTITION))
    END_IDX=$(( (i + 1) * EXAMPLES_PER_PARTITION ))
    
    # Don't exceed max examples
    if [ $END_IDX -gt $MAX_EXAMPLES ]; then
        END_IDX=$MAX_EXAMPLES
    fi
    
    # Skip if start index is beyond max examples
    if [ $START_IDX -ge $MAX_EXAMPLES ]; then
        continue
    fi
    
    JOB_SCRIPT="jobs/job_${PARTITION_ID}.sh"
    
    echo "Creating job script: $JOB_SCRIPT (rows $START_IDX-$((END_IDX-1)))"
    
    # Create the job script
    cat > "$JOB_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=medcalc_${PARTITION_ID}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:${GPUS_PER_JOB}
#SBATCH --cpus-per-task=${CPUS_PER_JOB}
#SBATCH --mem=${MEMORY_PER_JOB}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=logs/medcalc_${PARTITION_ID}_%j.out
#SBATCH --error=logs/medcalc_${PARTITION_ID}_%j.err

echo "🔧 Job ${PARTITION_ID} starting on node: \$(hostname)"
echo "📊 Processing rows $START_IDX to $((END_IDX-1))"
echo "🕐 Start time: \$(date)"

# Navigate to evaluation directory
cd /home/hrangara/MedCalc/MedCalc-Bench/evaluation

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate medCalcEnv

# Verify environment activation
if [ "\$CONDA_DEFAULT_ENV" != "medCalcEnv" ]; then
    echo "❌ Failed to activate medCalcEnv conda environment"
    exit 1
fi

echo "✅ Environment activated: \$CONDA_DEFAULT_ENV"
echo "🐍 Python version: \$(python --version)"
echo "📍 Working directory: \$(pwd)"

# Run the evaluation for this partition
echo "🚀 Starting MedCalc-Bench evaluation..."
python run.py \\
    --model "$MODEL" \\
    --prompt "$PROMPT" \\
    $ENABLE_ATTENTION \\
    --start_idx $START_IDX \\
    --end_idx $END_IDX \\
    --partition_id "$PARTITION_ID"

EXIT_CODE=\$?

echo "🕐 End time: \$(date)"
if [ \$EXIT_CODE -eq 0 ]; then
    echo "✅ Job ${PARTITION_ID} completed successfully"
else
    echo "❌ Job ${PARTITION_ID} failed with exit code \$EXIT_CODE"
fi

exit \$EXIT_CODE
EOF

    # Make the job script executable
    chmod +x "$JOB_SCRIPT"
    
    # Submit the job
    echo "Submitting job for partition $PARTITION_ID..."
    JOB_ID=$(sbatch "$JOB_SCRIPT" | awk '{print $4}')
    JOB_IDS+=($JOB_ID)
    
    echo "  Job ID: $JOB_ID"
    echo "  Log files: logs/medcalc_${PARTITION_ID}_${JOB_ID}.{out,err}"
    
done

echo ""
echo "✅ All jobs submitted successfully!"
echo ""
echo "📋 Job Summary:"
echo "  Number of jobs: ${#JOB_IDS[@]}"
echo "  Job IDs: ${JOB_IDS[*]}"
echo "  Examples per job: ~$EXAMPLES_PER_PARTITION"
echo ""

# Step 3: Provide monitoring commands
echo "🔍 Monitoring Commands:"
echo "  Check job status:    squeue -u \$USER"
echo "  Check specific jobs: squeue -j $(IFS=,; echo "${JOB_IDS[*]}")"
echo "  Cancel all jobs:     scancel $(IFS=' '; echo "${JOB_IDS[*]}")"
echo "  View logs:           tail -f logs/medcalc_p*_*.out"
echo ""

# Step 4: Create monitoring script
MONITOR_SCRIPT="monitor_jobs.sh"
cat > "$MONITOR_SCRIPT" << EOF
#!/bin/bash
# Monitor parallel MedCalc-Bench jobs

echo "🔍 MedCalc-Bench Job Status"
echo "=========================="
squeue -u \$USER -o "%.10i %.12j %.8T %.10M %.6D %R"

echo ""
echo "📊 Job Details:"
for job_id in ${JOB_IDS[*]}; do
    echo "  Job \$job_id: \$(squeue -j \$job_id -h -o "%T %M" 2>/dev/null || echo "COMPLETED/NOT_FOUND")"
done

echo ""
echo "📁 Output Files:"
ls -la outputs/*partition*.jsonl 2>/dev/null || echo "  No partition output files found yet"

echo ""
echo "🔍 Recent Log Activity:"
find logs -name "medcalc_p*_*.out" -newer logs 2>/dev/null | head -5 | while read logfile; do
    echo "  \$logfile: \$(tail -1 "\$logfile" 2>/dev/null || echo "empty")"
done
EOF

chmod +x "$MONITOR_SCRIPT"

echo "📊 Created monitoring script: $MONITOR_SCRIPT"
echo "   Run: ./$MONITOR_SCRIPT"
echo ""

# Step 5: Create results merger script
MERGE_SCRIPT="merge_results.py"
cat > "$MERGE_SCRIPT" << 'EOF'
#!/usr/bin/env python3
"""
Merge parallel processing results into a single file
"""

import pandas as pd
import glob
import os
import argparse

def merge_partition_results(pattern="outputs/*partition*.jsonl", output_file=None):
    """Merge all partition result files into a single file"""
    
    # Find all partition files
    partition_files = glob.glob(pattern)
    
    if not partition_files:
        print(f"❌ No partition files found matching pattern: {pattern}")
        return
    
    print(f"📁 Found {len(partition_files)} partition files:")
    for f in sorted(partition_files):
        print(f"  {f}")
    
    # Read and combine all files
    all_results = []
    total_rows = 0
    
    for file_path in sorted(partition_files):
        try:
            df = pd.read_json(file_path, lines=True)
            all_results.append(df)
            total_rows += len(df)
            print(f"  ✅ {file_path}: {len(df)} rows")
        except Exception as e:
            print(f"  ❌ {file_path}: Error - {e}")
    
    if not all_results:
        print("❌ No valid partition files to merge")
        return
    
    # Combine all dataframes
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Sort by Row Number to maintain order
    combined_df = combined_df.sort_values('Row Number').reset_index(drop=True)
    
    # Generate output filename if not provided
    if output_file is None:
        # Extract model and prompt info from first partition file
        first_file = os.path.basename(partition_files[0])
        # Remove partition suffix: model_prompt_partition_pXX.jsonl -> model_prompt.jsonl
        base_name = first_file.replace('_partition_p00', '').replace('_partition_p01', '').replace('_partition_p02', '').replace('_partition_p03', '')
        if '_partition_' in base_name:
            # More generic removal
            parts = base_name.split('_partition_')
            base_name = parts[0] + '.jsonl'
        output_file = f"outputs/merged_{base_name}"
    
    # Save merged results
    combined_df.to_json(output_file, orient='records', lines=True)
    
    print(f"\n✅ Merged results saved to: {output_file}")
    print(f"📊 Total rows: {total_rows}")
    print(f"🎯 Unique calculators: {combined_df['Calculator ID'].nunique()}")
    print(f"📝 Unique notes: {combined_df['Note ID'].nunique()}")
    
    # Show accuracy summary if available
    if 'Result' in combined_df.columns:
        correct = (combined_df['Result'] == 'Correct').sum()
        total = len(combined_df)
        accuracy = correct / total * 100 if total > 0 else 0
        print(f"🎯 Overall Accuracy: {correct}/{total} ({accuracy:.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge parallel processing results")
    parser.add_argument("--pattern", default="outputs/*partition*.jsonl", 
                       help="Pattern to match partition files")
    parser.add_argument("--output", default=None,
                       help="Output file name (auto-generated if not specified)")
    
    args = parser.parse_args()
    merge_partition_results(args.pattern, args.output)
EOF

chmod +x "$MERGE_SCRIPT"

echo "🔗 Created results merger script: $MERGE_SCRIPT"
echo "   Run after jobs complete: python $MERGE_SCRIPT"
echo ""

echo "🎉 Parallel processing setup complete!"
echo ""
echo "Next steps:"
echo "1. Monitor jobs: ./$MONITOR_SCRIPT"
echo "2. Wait for completion (check with: squeue -u \$USER)"
echo "3. Merge results: python $MERGE_SCRIPT"
echo "4. Check final results in outputs/merged_*.jsonl" 