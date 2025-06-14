#!/bin/bash
#SBATCH --job-name=medcalc_p03
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/medcalc_p03_%j.out
#SBATCH --error=logs/medcalc_p03_%j.err

echo "🔧 Job p03 starting on node: $(hostname)"
echo "📊 Processing rows 75 to 99"
echo "🕐 Start time: $(date)"

# Navigate to evaluation directory
cd /home/hrangara/MedCalc/MedCalc-Bench/evaluation

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate medCalcEnv

# Verify environment activation
if [ "$CONDA_DEFAULT_ENV" != "medCalcEnv" ]; then
    echo "❌ Failed to activate medCalcEnv conda environment"
    exit 1
fi

echo "✅ Environment activated: $CONDA_DEFAULT_ENV"
echo "🐍 Python version: $(python --version)"
echo "📍 Working directory: $(pwd)"

# Run the evaluation for this partition
echo "🚀 Starting MedCalc-Bench evaluation..."
python run.py \
    --model "meta-llama/Meta-Llama-3-8B-Instruct" \
    --prompt "zero_shot" \
    --enable_attention_analysis \
    --start_idx 75 \
    --end_idx 100 \
    --partition_id "p03"

EXIT_CODE=$?

echo "🕐 End time: $(date)"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Job p03 completed successfully"
else
    echo "❌ Job p03 failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
