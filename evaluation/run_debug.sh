#!/bin/bash

# MedCalc-Bench Debug Run Script
# This script runs a debug/test version with limited data for quick testing
# Usage: ./run_debug.sh

set -e  # Exit on any error

echo "🔧 MedCalc-Bench Debug Run"
echo "=========================="
echo "📅 Started at: $(date)"
echo ""

# Activate conda environment
echo "🐍 Activating medCalcEnv conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate medCalcEnv

# Verify environment activation
if [[ "$CONDA_DEFAULT_ENV" == "medCalcEnv" ]]; then
    echo "✅ Environment activated successfully: $CONDA_DEFAULT_ENV"
    echo "🐍 Python path: $(which python)"
    echo "📦 Package paths found: $(python -c "import sys; print(len([p for p in sys.path if 'medCalcEnv' in p]))")"
else
    echo "❌ Failed to activate medCalcEnv environment"
    exit 1
fi

echo ""
echo "🚀 Starting debug run with attention analysis..."
echo "📊 Running on limited dataset (first 10 rows) for quick testing"
echo "🔍 Model: meta-llama/Meta-Llama-3-8B-Instruct"
echo "💡 Prompt: zero_shot"
echo "👁️  Attention analysis: ENABLED"
echo ""

# Run the debug version
python run.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --prompt zero_shot \
    --enable_attention_analysis \
    --debug_run

echo ""
echo "✅ Debug run completed at: $(date)"
echo "📁 Check outputs/ directory for results"
echo "👁️  Check outputs/attention_analysis/ for attention visualizations" 