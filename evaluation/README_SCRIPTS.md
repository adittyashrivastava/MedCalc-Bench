# MedCalc-Bench Run Scripts

This directory contains convenient scripts for running MedCalc-Bench experiments with attention analysis enabled, including both single-instance and parallel processing options.

## Prerequisites

- Conda environment `medCalcEnv` must be created and configured with all dependencies
- Run the scripts from the `evaluation/` directory
- Hugging Face token is automatically configured via `hf_config.py`

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

### ⚡ Parallel Processing Scripts (NEW!)

**Purpose**: Process datasets in parallel across multiple Babel compute nodes for faster execution

#### Quick Start - Parallel Processing for First 100 Examples

```bash
# 1. Test the setup (recommended first)
./test_parallel.sh

# 2. Run parallel processing
./run_parallel.sh

# 3. Monitor jobs
./monitor_jobs.sh

# 4. After completion, merge results
python merge_results.py
```

#### Parallel Processing Components

**`partition_dataset.py`** - Dataset partitioning utility
```bash
python partition_dataset.py --max_examples 100 --num_partitions 4
```

**`run_parallel.sh`** - Main parallel processing script
- Partitions first 100 examples into 4 chunks (25 examples each)
- Submits 4 parallel Slurm jobs to debug partition
- Each job gets 1 GPU, 4 CPUs, 16GB RAM, 1 hour time limit
- Automatically creates monitoring and merging scripts

**`test_parallel.sh`** - Test parallel setup without submitting jobs
```bash
./test_parallel.sh  # Verify everything works before submitting
```

**`monitor_jobs.sh`** - Monitor running parallel jobs (auto-generated)
```bash
./monitor_jobs.sh  # Check job status and progress
```

**`merge_results.py`** - Combine parallel results (auto-generated)
```bash
python merge_results.py  # Merge partition results into single file
```

#### Customizing Parallel Processing

Edit `run_parallel.sh` to modify:
- `MAX_EXAMPLES`: Number of examples to process (default: 100)
- `NUM_PARTITIONS`: Number of parallel jobs (default: 4)
- `MODEL`: Model to use (default: meta-llama/Meta-Llama-3-8B-Instruct)
- `PROMPT`: Prompt style (default: zero_shot)
- `ENABLE_ATTENTION`: Enable/disable attention analysis
- Slurm parameters (partition, resources, time limits)

#### Manual Parallel Execution

You can also run individual partitions manually:
```bash
# Process rows 0-24 (partition p00)
python run.py --model meta-llama/Meta-Llama-3-8B-Instruct --prompt zero_shot \
  --enable_attention_analysis --start_idx 0 --end_idx 25 --partition_id p00

# Process rows 25-49 (partition p01)  
python run.py --model meta-llama/Meta-Llama-3-8B-Instruct --prompt zero_shot \
  --enable_attention_analysis --start_idx 25 --end_idx 50 --partition_id p01
```

## Environment Setup

All scripts automatically:
1. Activate the `medCalcEnv` conda environment
2. Configure Hugging Face token from `hf_config.py`
3. Verify successful activation
4. Display environment information
5. Run the appropriate command

## Output

Results are saved to:
- `outputs/` - Main evaluation results
  - Single runs: `{model}_{prompt}.jsonl`
  - Parallel runs: `{model}_{prompt}_partition_{id}.jsonl`
  - Merged results: `merged_{model}_{prompt}.jsonl`
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

### Parallel Processing Advantages

The parallel processing approach offers several benefits:
- **Speed**: 4x faster processing with 4 parallel jobs
- **Resource Efficiency**: Better utilization of available GPU nodes
- **Fault Tolerance**: If one job fails, others continue
- **Scalability**: Easy to adjust number of partitions
- **Monitoring**: Built-in job monitoring and progress tracking

## Troubleshooting

If you encounter issues:
1. Ensure the `medCalcEnv` conda environment exists and is properly configured
2. Check that you're in the `evaluation/` directory when running scripts
3. For CUDA/memory errors on login nodes, try running on GPU compute nodes
4. Check the script output for detailed error messages
5. For parallel processing issues:
   - Run `./test_parallel.sh` first to verify setup
   - Check job logs in `logs/` directory
   - Use `./monitor_jobs.sh` to check job status
   - Verify partition files exist in `partitions/` directory

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

**Parallel partition run**:
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate medCalcEnv
python run.py --model meta-llama/Meta-Llama-3-8B-Instruct --prompt zero_shot --enable_attention_analysis --start_idx 0 --end_idx 25 --partition_id p00
``` 