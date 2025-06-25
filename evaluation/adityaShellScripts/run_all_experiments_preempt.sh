#!/bin/bash
#SBATCH --job-name=medcalc-llama3-8b-preempt
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=medcalc_experiments_preempt_%j.out
#SBATCH --error=medcalc_experiments_preempt_%j.err

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Initialize conda for bash shell
eval "$(conda shell.bash hook)"

# Load environment variables
source ../../startup.sh

# Activate conda environment
conda activate medcalc-env

# Verify GPU availability
nvidia-smi

# Model name
MODEL="meta-llama/Meta-Llama-3-8B-Instruct"

# Array of prompt types
PROMPTS=("zero_shot" "direct_answer" "one_shot")

# Run experiments for each prompt type
for PROMPT in "${PROMPTS[@]}"; do
    echo "========================================="
    echo "Starting experiment: $PROMPT"
    echo "Time: $(date)"
    echo "========================================="

    # Run the experiment
    python ../run.py --model "$MODEL" --prompt "$PROMPT"

    # Check if the experiment completed successfully
    if [ $? -eq 0 ]; then
        echo "✅ Successfully completed: $PROMPT"
    else
        echo "❌ Failed: $PROMPT"
        # Continue with other experiments even if one fails
    fi

    echo "Completed experiment: $PROMPT at $(date)"
    echo ""
done

echo "========================================="
echo "All experiments completed!"
echo "End Time: $(date)"
echo "========================================="

# List the generated files
echo "Generated result files:"
ls -la ../results/results_*Llama-3-8B-Instruct*.json

echo ""
echo "Generated output files:"
ls -la ../outputs/meta-llama_Meta-Llama-3-8B-Instruct*.jsonl

# Print job statistics
echo ""
echo "Job Statistics:"
echo "Job ID: $SLURM_JOB_ID"
echo "Total Runtime: $SECONDS seconds"