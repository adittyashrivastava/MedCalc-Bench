#!/bin/bash

# MedCalc-Bench Debug Run Script
# This script runs a debug/test version with limited data for quick testing
# Usage: ./run_debug.sh [output_directory] [num_examples] [prompt_type]
# If no output directory is provided, defaults to /data/user_data/hrangara/experiments-{num_examples}/{timestamp}/outputs
# If no num_examples is provided, defaults to 10
# If no prompt_type is provided, defaults to zero_shot

set -e  # Exit on any error

# Set default values
NUM_EXAMPLES=${2:-10}
PROMPT_TYPE=${3:-"zero_shot"}

# Validate prompt type
if [[ ! "$PROMPT_TYPE" =~ ^(zero_shot|one_shot|direct_answer)$ ]]; then
    echo "❌ Error: Invalid prompt type '$PROMPT_TYPE'"
    echo "Valid options: zero_shot, one_shot, direct_answer"
    exit 1
fi

echo "🔧 MedCalc-Bench Debug Run"
echo "=========================="
echo "📅 Started at: $(date)"
echo "📊 Number of examples: $NUM_EXAMPLES"
echo "💡 Prompt type: $PROMPT_TYPE"
echo ""

# Generate timestamp for directory naming
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Set output directory - use provided argument or default
if [ -z "$1" ]; then
    OUTPUT_DIR="/data/user_data/hrangara/experiments-${NUM_EXAMPLES}/${TIMESTAMP}/outputs"
else
    OUTPUT_DIR="$1"
fi

echo "📁 Output directory: $OUTPUT_DIR"

# Create output directory structure
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/llm_results"
mkdir -p "$OUTPUT_DIR/attention_results"
mkdir -p "$OUTPUT_DIR/attrieval_results"

echo "✅ Created output directory structure"
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
echo "🚀 Starting debug run with attention analysis and ATTRIEVAL..."
echo "📊 Running on limited dataset (first $NUM_EXAMPLES rows) for testing"
echo "🔍 Model: meta-llama/Meta-Llama-3-8B-Instruct"
echo "💡 Prompt: $PROMPT_TYPE"
echo "👁️  Attention analysis: ENABLED"
echo "🎯 ATTRIEVAL fact retrieval: ENABLED"
echo ""

# Run the debug version with custom output directory and number of examples
python ../run.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --prompt "$PROMPT_TYPE" \
    --enable_attention_analysis \
    --enable_attrieval \
    --debug_run \
    --num_examples "$NUM_EXAMPLES" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "✅ Debug run completed at: $(date)"
echo "📁 Check $OUTPUT_DIR/llm_results/ for LLM results"
echo "👁️  Check $OUTPUT_DIR/attention_results/ for attention visualizations"
echo "🎯 Check $OUTPUT_DIR/attrieval_results/ for ATTRIEVAL fact retrieval analysis"