#!/bin/bash

# MedCalc-Bench SLURM Debug Job Submission Script
# This script submits a SLURM job to run debug experiments
# Usage: ./run_slurm_debug.sh [num_examples] [output_directory]
# If no num_examples is provided, defaults to 10
# If no output_directory is provided, uses default timestamped directory

set -e  # Exit on any error

# Set default number of examples
NUM_EXAMPLES=${1:-10}
OUTPUT_DIR=${2:-""}

echo "🚀 MedCalc-Bench SLURM Debug Job Submission"
echo "==========================================="
echo "📊 Number of examples: $NUM_EXAMPLES"
if [ -n "$OUTPUT_DIR" ]; then
    echo "📁 Output directory: $OUTPUT_DIR"
else
    echo "📁 Output directory: Using timestamped default"
fi
echo ""

# Generate a unique job name with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="medcalc_debug_${NUM_EXAMPLES}_${TIMESTAMP}"

echo "🏷️  Job name: $JOB_NAME"

# Create the SLURM script
SLURM_SCRIPT="slurm_job_${JOB_NAME}.sh"

cat > "$SLURM_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/${JOB_NAME}_%j.out
#SBATCH --error=logs/${JOB_NAME}_%j.err

# Print job information
echo "======================================"
echo "SLURM Job Information"
echo "======================================"
echo "Job ID: \$SLURM_JOB_ID"
echo "Job Name: \$SLURM_JOB_NAME"
echo "Node: \$SLURMD_NODENAME"
echo "Started at: \$(date)"
echo "Number of examples: $NUM_EXAMPLES"
echo "======================================"
echo ""

# Change to the evaluation directory
cd /home/hrangara/MedCalc/MedCalc-Bench/evaluation

# Run the debug script with appropriate parameters
if [ -n "$OUTPUT_DIR" ]; then
    echo "🚀 Running debug script with custom output directory..."
    ./run_debug.sh "$OUTPUT_DIR" $NUM_EXAMPLES
else
    echo "🚀 Running debug script with default output directory..."
    ./run_debug.sh "" $NUM_EXAMPLES
fi

echo ""
echo "======================================"
echo "Job completed at: \$(date)"
echo "======================================"
EOF

# Create logs directory if it doesn't exist
mkdir -p logs

# Make the SLURM script executable
chmod +x "$SLURM_SCRIPT"

echo "📝 Created SLURM script: $SLURM_SCRIPT"
echo ""

# Submit the job
echo "📤 Submitting job to SLURM..."
JOB_ID=$(sbatch "$SLURM_SCRIPT" | awk '{print $4}')

echo "✅ Job submitted successfully!"
echo "🆔 Job ID: $JOB_ID"
echo "🏷️  Job Name: $JOB_NAME"
echo ""
echo "📊 Job Details:"
echo "   - Partition: general"
echo "   - GPU: 1"
echo "   - CPUs: 4"
echo "   - Memory: 32G"
echo "   - Time Limit: 8 hours"
echo "   - Examples to process: $NUM_EXAMPLES"
echo ""
echo "📝 Monitor job with:"
echo "   squeue -u \$USER"
echo "   squeue -j $JOB_ID"
echo ""
echo "📄 Check logs:"
echo "   tail -f logs/${JOB_NAME}_${JOB_ID}.out"
echo "   tail -f logs/${JOB_NAME}_${JOB_ID}.err"
echo ""
echo "❌ Cancel job if needed:"
echo "   scancel $JOB_ID"

# Clean up the temporary SLURM script after a delay (optional)
# Uncomment the following lines if you want auto-cleanup
# echo ""
# echo "🧹 SLURM script will be cleaned up in 60 seconds..."
# (sleep 60 && rm -f "$SLURM_SCRIPT" && echo "🗑️  Cleaned up $SLURM_SCRIPT") & 