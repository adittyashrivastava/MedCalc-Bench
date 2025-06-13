# MedCalc-Bench Run Scripts

This directory contains two convenient scripts for running MedCalc-Bench experiments with attention analysis enabled.

## Prerequisites

- Conda environment `medCalcEnv` must be created and configured with all dependencies
- Run the scripts from the `evaluation/` directory

## Scripts

### 🔧 Debug Run Script (`run_debug.sh`)

**Purpose**: Quick testing and debugging with limited data

**Usage**:
```bash
./run_debug.sh
```

**Features**:
- Processes only the first 10 rows of the dataset
- Enables attention analysis
- Uses meta-llama/Meta-Llama-3-8B-Instruct model
- Uses zero_shot prompting
- Fast execution for testing purposes

### 🚀 Full Run Script (`run_full.sh`)

**Purpose**: Complete evaluation on the entire dataset

**Usage**:
```bash
./run_full.sh
```

**Features**:
- Processes the complete dataset
- Enables attention analysis
- Uses meta-llama/Meta-Llama-3-8B-Instruct model
- Uses zero_shot prompting
- Includes confirmation prompt before starting
- May take hours to complete

## Environment Setup

Both scripts automatically:
1. Activate the `medCalcEnv` conda environment
2. Verify successful activation
3. Display environment information
4. Run the appropriate command

## Output

Results are saved to:
- `outputs/` - Main evaluation results
- `outputs/attention_analysis/` - Attention visualization files
  - `calc_{calculator_id}_note_{note_id}/` - Individual entry directories
  - `basic_attention.html` - Interactive attention visualizations
  - `attention_stats.json` - Attention analysis statistics

## GPU Requirements

⚠️ **Important**: For optimal performance, especially for the full run, consider running on a GPU compute node rather than login nodes.

### Getting a Debug Node on Babel

To run on a GPU compute node instead of the login node:

1. **Start an Interactive Session on debug**:
   ```bash
   # Ask for 1 GPU (debug allows up to 2), 4 CPUs, 16 GB RAM, 1 hour:
   salloc --partition=debug --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=01:00:00 \
     srun --pty bash
   ```
   
2. **Wait for resource allocation**: As soon as this grants you resources, your prompt will change—you're now on a compute node.

3. **Navigate to the evaluation directory**:
   ```bash
   cd /home/hrangara/MedCalc/MedCalc-Bench/evaluation
   ```

4. **Run your script**:
   ```bash
   ./run_debug.sh    # or ./run_full.sh
   ```

This approach will provide GPU access and avoid the CUDA/memory errors that occur on login nodes.

## Troubleshooting

If you encounter issues:
1. Ensure the `medCalcEnv` conda environment exists and is properly configured
2. Check that you're in the `evaluation/` directory when running scripts
3. For CUDA/memory errors on login nodes, try running on GPU compute nodes
4. Check the script output for detailed error messages

## Manual Execution

You can also run the commands manually:

**Debug run**:
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate medCalcEnv
python run.py --model meta-llama/Meta-Llama-3-8B-Instruct --prompt zero_shot --enable_attention_analysis --debug_run
```

**Full run**:
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate medCalcEnv
python run.py --model meta-llama/Meta-Llama-3-8B-Instruct --prompt zero_shot --enable_attention_analysis
``` 