#!/bin/bash
#SBATCH --job-name=medcalc_debug_150_20250617_003005
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=logs/medcalc_debug_150_20250617_003005_%j.out
#SBATCH --error=logs/medcalc_debug_150_20250617_003005_%j.err

# Print job information
echo "======================================"
echo "SLURM Job Information"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Started at: $(date)"
echo "Number of examples: 150"
echo "======================================"
echo ""

# Change to the evaluation directory
cd /home/hrangara/MedCalc/MedCalc-Bench/evaluation

# Run the debug script with appropriate parameters
if [ -n "" ]; then
    echo "🚀 Running debug script with custom output directory..."
    ./run_debug.sh "" 150
else
    echo "🚀 Running debug script with default output directory..."
    ./run_debug.sh "" 150
fi

echo ""
echo "======================================"
echo "Job completed at: $(date)"
echo "======================================"
