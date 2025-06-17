#!/bin/bash

# MedCalc-Bench Full Run Script
# This script runs the complete evaluation on the full dataset
# Usage: ./run_full.sh

set -e  # Exit on any error

echo "🚀 MedCalc-Bench Full Run"
echo "========================="
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
echo "🔥 Starting FULL run with attention analysis and ATTRIEVAL..."
echo "📊 Running on COMPLETE dataset - this will take significant time!"
echo "🔍 Model: meta-llama/Meta-Llama-3-8B-Instruct"
echo "💡 Prompt: zero_shot"
echo "👁️  Attention analysis: ENABLED"
echo "🎯 ATTRIEVAL fact retrieval: ENABLED"
echo ""
echo "⚠️  WARNING: This is a full run and may take hours to complete!"
echo "💡 Consider running this on a GPU compute node for better performance"
echo ""

# Ask for confirmation
read -p "🤔 Are you sure you want to proceed with the full run? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Full run cancelled by user"
    exit 1
fi

echo "✅ Proceeding with full run..."
echo ""

# Run the full version
python run.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --prompt zero_shot \
    --enable_attention_analysis \
    --enable_attrieval

echo ""
echo "🎉 Full run completed at: $(date)"
echo "📁 Check outputs/ directory for results"
echo "👁️  Check outputs/attention_analysis/ for attention visualizations"
echo "🎯 Check outputs/attrieval_analysis/ for ATTRIEVAL fact retrieval analysis"
echo "📊 Full evaluation results with ATTRIEVAL analysis now available" 