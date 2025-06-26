#!/bin/bash
#SBATCH --job-name=medcalc---model_name
#SBATCH --partition=Qwen/Qwen2.5-7B-Instruct
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=10
#SBATCH --time=--num_examples
#SBATCH --output=medcalc_--model_name_%j.out
#SBATCH --error=medcalc_--model_name_%j.err

# Print job information
echo "========================================="
echo "MedCalc-Bench Experiment"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "Model: --model_name"
echo "Partition: Qwen/Qwen2.5-7B-Instruct"
echo "Time Limit: --num_examples"
echo "Memory: 10"
if [ -n "--enable_attention --enable_attrieval --enable_formula_catalogue --output_dir outputs/" ]; then
    echo "Additional Arguments: --enable_attention --enable_attrieval --enable_formula_catalogue --output_dir outputs/"
fi
echo "========================================="

# Set up conda environment
export PATH="/home/ashriva3/miniconda3/bin:$PATH"
source /home/ashriva3/miniconda3/etc/profile.d/conda.sh

# Load environment variables if startup.sh exists
if [ -f "../../startup.sh" ]; then
    echo "Loading environment variables from startup.sh..."
    source ../../startup.sh
else
    echo "startup.sh not found, skipping..."
fi

# Activate conda environment
echo "Activating medcalc-env..."
conda activate medcalc-env

# Verify environment
echo "Python path: $(which python)"
echo "Conda environment: $CONDA_DEFAULT_ENV"

# Check and install all required dependencies
echo "Checking and installing dependencies..."

# Core data science packages
python -c "import pandas; print('✅ pandas available')" 2>/dev/null || {
    echo "❌ pandas missing, installing..."
    pip install pandas
}

python -c "import numpy; print('✅ numpy available')" 2>/dev/null || {
    echo "❌ numpy missing, installing..."
    pip install numpy
}

# ML/AI packages
python -c "import torch; print('✅ torch available')" 2>/dev/null || {
    echo "❌ torch missing, installing..."
    pip install torch
}

python -c "import transformers; print('✅ transformers available')" 2>/dev/null || {
    echo "❌ transformers missing, installing..."
    pip install transformers
}

# OpenAI and related packages
python -c "import openai; print('✅ openai available')" 2>/dev/null || {
    echo "❌ openai missing, installing..."
    pip install openai
}

python -c "import tiktoken; print('✅ tiktoken available')" 2>/dev/null || {
    echo "❌ tiktoken missing, installing..."
    pip install tiktoken
}

python -c "import huggingface_hub; print('✅ huggingface_hub available')" 2>/dev/null || {
    echo "❌ huggingface_hub missing, installing..."
    pip install huggingface_hub
}

# Other required packages
python -c "import tqdm; print('✅ tqdm available')" 2>/dev/null || {
    echo "❌ tqdm missing, installing..."
    pip install tqdm
}

python -c "import accelerate; print('✅ accelerate available')" 2>/dev/null || {
    echo "❌ accelerate missing, installing..."
    pip install accelerate
}

# Install attention_viz if not available
python -c "import attention_viz; print('✅ attention_viz available')" 2>/dev/null || {
    echo "❌ attention_viz missing, installing from local directory..."
    pip install -e /home/ashriva3/codebase/attention_viz
    python -c "import attention_viz; print('✅ attention_viz installed successfully')" 2>/dev/null || {
        echo "❌ Failed to install attention_viz, but continuing anyway..."
    }
}

echo "✅ Dependencies check completed."

# Verify GPU availability
echo "GPU Information:"
nvidia-smi

# Model configuration
MODEL="--model_name"

# Additional arguments for run.py
ADDITIONAL_ARGS="--enable_attention --enable_attrieval --enable_formula_catalogue --output_dir outputs/"

# Array of prompt types
PROMPTS=("zero_shot" "direct_answer" "one_shot")

echo ""
echo "========================================="
echo "Starting experiments for model: $MODEL"
echo "Prompt types: ${PROMPTS[@]}"
if [ -n "$ADDITIONAL_ARGS" ]; then
    echo "Additional arguments: $ADDITIONAL_ARGS"
fi
echo "========================================="

# Run experiments for each prompt type
for PROMPT in "${PROMPTS[@]}"; do
    echo ""
    echo "========================================="
    echo "Starting experiment: $PROMPT"
    echo "Time: $(date)"
    echo "========================================="

    # Build the command with additional arguments
    CMD="python run.py --model \"$MODEL\" --prompt \"$PROMPT\""
    if [ -n "$ADDITIONAL_ARGS" ]; then
        CMD="$CMD $ADDITIONAL_ARGS"
    fi

    echo "Running command: $CMD"

    # Run the experiment
    eval $CMD

    # Check if the experiment completed successfully
    if [ $? -eq 0 ]; then
        echo "✅ Successfully completed: $PROMPT"
    else
        echo "❌ Failed: $PROMPT"
        echo "Exit code: $?"
        # Continue with other experiments even if one fails
    fi

    echo "Completed experiment: $PROMPT at $(date)"
done

echo ""
echo "========================================="
echo "All experiments completed!"
echo "End Time: $(date)"
echo "========================================="

# Create model-safe filename pattern for listing files
MODEL_PATTERN=$(echo "--model_name" | sed 's/\//_/g')

# List the generated files
echo "Generated output files:"
ls -la ../outputs/${MODEL_PATTERN}_*.jsonl 2>/dev/null || echo "No output files found in outputs/"

echo ""
echo "Generated result files:"
ls -la ../results/results_*${MODEL_PATTERN}*.json 2>/dev/null || echo "No result files found in results/"

# Print summary statistics
echo ""
echo "========================================="
echo "Job Summary"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Model: --model_name"
echo "Total Runtime: $SECONDS seconds"
echo "Node: $SLURMD_NODENAME"
echo "End Time: $(date)"

# Check if all experiments produced output files
echo ""
echo "Experiment Status Check:"
for PROMPT in "${PROMPTS[@]}"; do
    if [ -f "../outputs/${MODEL_PATTERN}_${PROMPT}.jsonl" ]; then
        LINES=$(wc -l < "../outputs/${MODEL_PATTERN}_${PROMPT}.jsonl")
        echo "✅ $PROMPT: $LINES lines generated"
    else
        echo "❌ $PROMPT: No output file found"
    fi
done

echo "========================================="
