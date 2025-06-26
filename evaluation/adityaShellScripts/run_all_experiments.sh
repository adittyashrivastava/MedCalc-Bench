#!/bin/bash

# Check if model parameter is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_name> [partition] [time_limit] [memory] [options...]"
    echo ""
    echo "Examples:"
    echo "  $0 Qwen/Qwen2.5-7B-Instruct"
    echo "  $0 Qwen/Qwen2.5-14B-Instruct general 24:00:00 64G"
    echo "  $0 meta-llama/Meta-Llama-3-8B-Instruct general 12:00:00 32G --enable_attention_analysis --enable_formula_catalogue"
    echo "  $0 meta-llama/Meta-Llama-3-8B-Instruct general 12:00:00 32G --debug_run --num_examples 50"
    echo ""
    echo "Supported models:"
    echo "  - Qwen/Qwen2.5-7B-Instruct"
    echo "  - Qwen/Qwen2.5-14B-Instruct"
    echo "  - Qwen/Qwen2.5-32B-Instruct"
    echo "  - Qwen/Qwen2.5-72B-Instruct"
    echo "  - Qwen/Qwen2-7B-Instruct"
    echo "  - meta-llama/Meta-Llama-3-8B-Instruct"
    echo "  - meta-llama/Meta-Llama-3-70B-Instruct"
    echo "  - mistralai/Mistral-7B-Instruct-v0.2"
    echo "  - mistralai/Mixtral-8x7B-Instruct-v0.1"
    echo "  - epfl-llm/meditron-70b"
    echo ""
    echo "Parameters:"
    echo "  model_name  : Required - HuggingFace model name"
    echo "  partition   : Optional - SLURM partition (default: general)"
    echo "  time_limit  : Optional - Job time limit (default: 12:00:00)"
    echo "  memory      : Optional - Memory allocation (default: 32G)"
    echo ""
    echo "Optional run.py flags (pass after the 4 main parameters):"
    echo "  --enable_attention_analysis  : Enable attention analysis"
    echo "  --enable_attrieval          : Enable ATTRIEVAL analysis"
    echo "  --enable_formula_catalogue  : Enable medical formula catalogue"
    echo "  --debug_run                 : Run in debug mode (limited examples)"
    echo "  --num_examples N            : Number of examples to process in debug mode"
    echo "  --output_dir DIR            : Custom output directory"
    echo ""
    echo "Example with all options:"
    echo "  $0 meta-llama/Meta-Llama-3-8B-Instruct general 12:00:00 32G --enable_attention_analysis --enable_attrieval --enable_formula_catalogue --debug_run --num_examples 100 --output_dir /custom/path"
    exit 1
fi

# Parse command line arguments
MODEL="$1"
PARTITION="${2:-general}"
TIME_LIMIT="${3:-12:00:00}"
MEMORY="${4:-32G}"

# Collect additional arguments (run.py flags)
shift 4 2>/dev/null || shift $#  # Remove first 4 args if they exist, otherwise remove all
ADDITIONAL_ARGS="$@"

# Extract model name for job naming (replace slashes and special chars)
MODEL_SAFE=$(echo "$MODEL" | sed 's/[^a-zA-Z0-9._-]/_/g')
MODEL_SHORT=$(echo "$MODEL" | sed 's/.*\///' | sed 's/[^a-zA-Z0-9._-]/_/g')

# Create jobs directory if it doesn't exist
mkdir -p ../jobs

# Create SLURM script content
cat > ../jobs/medcalc_experiment_${MODEL_SAFE}.sh << EOF
#!/bin/bash
#SBATCH --job-name=medcalc-${MODEL_SHORT}
#SBATCH --partition=${PARTITION}
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=${MEMORY}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=medcalc_${MODEL_SAFE}_%j.out
#SBATCH --error=medcalc_${MODEL_SAFE}_%j.err

# Print job information
echo "========================================="
echo "MedCalc-Bench Experiment"
echo "========================================="
echo "Job ID: \$SLURM_JOB_ID"
echo "Job Name: \$SLURM_JOB_NAME"
echo "Node: \$SLURMD_NODENAME"
echo "Start Time: \$(date)"
echo "Working Directory: \$(pwd)"
echo "Model: ${MODEL}"
echo "Partition: ${PARTITION}"
echo "Time Limit: ${TIME_LIMIT}"
echo "Memory: ${MEMORY}"
if [ -n "${ADDITIONAL_ARGS}" ]; then
    echo "Additional Arguments: ${ADDITIONAL_ARGS}"
fi
echo "========================================="

# Set up conda environment
export PATH="/home/ashriva3/miniconda3/bin:\$PATH"
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
echo "Python path: \$(which python)"
echo "Conda environment: \$CONDA_DEFAULT_ENV"

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
MODEL="${MODEL}"

# Additional arguments for run.py
ADDITIONAL_ARGS="${ADDITIONAL_ARGS}"

# Array of prompt types
PROMPTS=("zero_shot" "direct_answer" "one_shot")

echo ""
echo "========================================="
echo "Starting experiments for model: \$MODEL"
echo "Prompt types: \${PROMPTS[@]}"
if [ -n "\$ADDITIONAL_ARGS" ]; then
    echo "Additional arguments: \$ADDITIONAL_ARGS"
fi
echo "========================================="

# Run experiments for each prompt type
for PROMPT in "\${PROMPTS[@]}"; do
    echo ""
    echo "========================================="
    echo "Starting experiment: \$PROMPT"
    echo "Time: \$(date)"
    echo "========================================="

    # Build the command with additional arguments
    CMD="python run.py --model \"\$MODEL\" --prompt \"\$PROMPT\""
    if [ -n "\$ADDITIONAL_ARGS" ]; then
        CMD="\$CMD \$ADDITIONAL_ARGS"
    fi

    echo "Running command: \$CMD"

    # Run the experiment
    eval \$CMD

    # Check if the experiment completed successfully
    if [ \$? -eq 0 ]; then
        echo "✅ Successfully completed: \$PROMPT"
    else
        echo "❌ Failed: \$PROMPT"
        echo "Exit code: \$?"
        # Continue with other experiments even if one fails
    fi

    echo "Completed experiment: \$PROMPT at \$(date)"
done

echo ""
echo "========================================="
echo "All experiments completed!"
echo "End Time: \$(date)"
echo "========================================="

# Create model-safe filename pattern for listing files
MODEL_PATTERN=\$(echo "${MODEL}" | sed 's/\//_/g')

# List the generated files
echo "Generated output files:"
ls -la ../outputs/\${MODEL_PATTERN}_*.jsonl 2>/dev/null || echo "No output files found in outputs/"

echo ""
echo "Generated result files:"
ls -la ../results/results_*\${MODEL_PATTERN}*.json 2>/dev/null || echo "No result files found in results/"

# Print summary statistics
echo ""
echo "========================================="
echo "Job Summary"
echo "========================================="
echo "Job ID: \$SLURM_JOB_ID"
echo "Model: ${MODEL}"
echo "Total Runtime: \$SECONDS seconds"
echo "Node: \$SLURMD_NODENAME"
echo "End Time: \$(date)"

# Check if all experiments produced output files
echo ""
echo "Experiment Status Check:"
for PROMPT in "\${PROMPTS[@]}"; do
    if [ -f "../outputs/\${MODEL_PATTERN}_\${PROMPT}.jsonl" ]; then
        LINES=\$(wc -l < "../outputs/\${MODEL_PATTERN}_\${PROMPT}.jsonl")
        echo "✅ \$PROMPT: \$LINES lines generated"
    else
        echo "❌ \$PROMPT: No output file found"
    fi
done

echo "========================================="
EOF

# Make the generated script executable
chmod +x ../jobs/medcalc_experiment_${MODEL_SAFE}.sh

echo "========================================="
echo "SLURM Job Script Generated"
echo "========================================="
echo "Script: ../jobs/medcalc_experiment_${MODEL_SAFE}.sh"
echo "Model: $MODEL"
echo "Partition: $PARTITION"
echo "Time Limit: $TIME_LIMIT"
echo "Memory: $MEMORY"
if [ -n "${ADDITIONAL_ARGS}" ]; then
    echo "Additional Arguments: ${ADDITIONAL_ARGS}"
fi
echo ""
echo "To submit the job:"
echo "  sbatch ../jobs/medcalc_experiment_${MODEL_SAFE}.sh"
echo ""
echo "To check job status:"
echo "  squeue -u $USER"
echo ""
echo "To check job details:"
echo "  scontrol show job <job_id>"
echo ""
echo "To cancel job:"
echo "  scancel <job_id>"
echo "========================================="