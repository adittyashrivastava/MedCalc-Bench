# MedCalc-Bench Run Scripts

This directory contains convenient scripts for running MedCalc-Bench experiments with attention analysis enabled, including both single-instance, SLURM job submission, and parallel processing options.

## Prerequisites

- Conda environment `medCalcEnv` must be created and configured with all dependencies
- Run the scripts from the `evaluation/` directory
- Hugging Face token is automatically configured via `hf_config.py`

## Scripts

### 🔧 Debug Run Script (`run_debug.sh`) - **UPDATED**

**Purpose**: Quick testing and debugging with configurable number of examples

**Usage**:
```bash
# Run with default 10 examples and timestamped output directory
./run_debug.sh

# Run with custom output directory and default 10 examples
./run_debug.sh /path/to/custom/output

# Run with default timestamped directory and 50 examples
./run_debug.sh "" 50

# Run with custom directory and 100 examples
./run_debug.sh /path/to/custom/output 100
```

**Features**:
- **Configurable examples**: Process any number of examples (default: 10)
- **Custom output directories**: Specify output location or use timestamped defaults
- **Organized output structure**: Separate folders for LLM results and attention analysis
- **Unique identifiers**: Each result uniquely identified by calculator_id, note_id, and row_number
- Enables attention analysis
- Uses meta-llama/Meta-Llama-3-8B-Instruct model
- Uses zero_shot prompting

**Output Structure**:
```
/data/user_data/hrangara/experiments-{num_examples}/{timestamp}/outputs/
├── llm_results/
│   └── meta-llama_Meta-Llama-3-8B-Instruct_zero_shot.jsonl
└── attention_results/
    ├── calc_1_note_1_row_1/
    ├── calc_1_note_2_row_2/
    └── ...
```

### 🎯 SLURM Debug Job Script (`run_slurm_debug.sh`) - **NEW**

**Purpose**: Submit debug experiments as SLURM jobs for better resource management

**Usage**:
```bash
# Submit job with default 10 examples and timestamped output
./run_slurm_debug.sh

# Submit job with 25 examples
./run_slurm_debug.sh 25

# Submit job with 50 examples and custom output directory
./run_slurm_debug.sh 50 /data/user_data/hrangara/my_experiment
```

**Features**:
- **SLURM job submission**: Runs on dedicated compute nodes
- **Resource allocation**: 1 GPU, 4 CPUs, 32GB RAM, 8-hour time limit
- **General partition**: Uses 'general' partition for longer jobs
- **Comprehensive logging**: Organized log files in `logs/` directory
- **Job monitoring**: Built-in commands for status checking
- **Unique job naming**: Jobs named with timestamp and example count

**Job Configuration**:
- Partition: `general`
- Resources: 1 GPU, 4 CPUs, 32G memory
- Time limit: 8 hours
- Log files: `logs/medcalc_debug_{num_examples}_{timestamp}_{job_id}.{out,err}`

**Monitoring Commands** (provided after job submission):
```bash
# Check job status
squeue -u $USER
squeue -j {JOB_ID}

# View logs in real-time
tail -f logs/medcalc_debug_{num_examples}_{timestamp}_{job_id}.out
tail -f logs/medcalc_debug_{num_examples}_{timestamp}_{job_id}.err

# Cancel job if needed
scancel {JOB_ID}
```

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

### ⚡ Parallel Processing Scripts

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

## Comprehensive Usage Examples

### 🎯 Common Workflows

#### **Quick Test (10 examples)**:
```bash
# Direct execution
./run_debug.sh

# SLURM job submission
./run_slurm_debug.sh
```

#### **Small Experiment (25 examples)**:
```bash
# Direct execution with custom directory
./run_debug.sh /data/user_data/hrangara/small_test 25

# SLURM job submission
./run_slurm_debug.sh 25
```

#### **Medium Experiment (100 examples)**:
```bash
# SLURM job submission (recommended for 100+ examples)
./run_slurm_debug.sh 100 /data/user_data/hrangara/medium_experiment

# Or use parallel processing for faster execution
./run_parallel.sh  # Processes 100 examples across 4 parallel jobs
```

#### **Large Experiment (500+ examples)**:
```bash
# Modify run_parallel.sh settings and use parallel processing
# Edit MAX_EXAMPLES=500 and NUM_PARTITIONS=10 in run_parallel.sh
./run_parallel.sh
```

### 📊 Output Organization Examples

#### **Default Timestamped Output**:
```bash
./run_debug.sh "" 50
# Creates: /data/user_data/hrangara/experiments-50/20241201_143022/outputs/
```

#### **Custom Organized Output**:
```bash
./run_slurm_debug.sh 100 /data/user_data/hrangara/llama3_baseline_experiment
# Creates: /data/user_data/hrangara/llama3_baseline_experiment/llm_results/
#          /data/user_data/hrangara/llama3_baseline_experiment/attention_results/
```

### 🔧 Advanced Usage with run.py

#### **Custom Model and Prompt**:
```bash
python run.py \
    --model meta-llama/Meta-Llama-3-70B-Instruct \
    --prompt one_shot \
    --enable_attention_analysis \
    --debug_run \
    --num_examples 50 \
    --output_dir /data/user_data/hrangara/llama3_70b_oneshot
```

#### **Specific Range Processing**:
```bash
python run.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --prompt zero_shot \
    --enable_attention_analysis \
    --start_idx 100 \
    --end_idx 200 \
    --partition_id custom_range \
    --output_dir /data/user_data/hrangara/rows_100_200
```

## Environment Setup

All scripts automatically:
1. Activate the `medCalcEnv` conda environment
2. Configure Hugging Face token from `hf_config.py`
3. Verify successful activation
4. Display environment information
5. Run the appropriate command

## Output

Results are saved to organized directory structures:

### **LLM Results** (`llm_results/`):
- `{model}_{prompt}.jsonl` - Main evaluation results
- Each entry includes unique identifiers and timestamps
- Cross-references to attention analysis directories

### **Attention Analysis** (`attention_results/`):
- `calc_{calculator_id}_note_{note_id}_row_{row_number}/` - Individual entry directories
- `basic_attention.html` - Interactive attention visualizations
- `attention_heatmap.png` - Static attention heatmaps
- `layer_comparison.png` - Multi-layer attention comparison
- `essential_attention_data.npz` - Compressed attention weights
- `attention_report.md` - Comprehensive analysis report
- `attention_summary.json` - Metadata and file listing

## GPU Requirements

⚠️ **Important**: For optimal performance, especially for larger experiments, use SLURM job submission rather than login nodes.

### Recommended Resource Usage

- **1-25 examples**: Direct execution on login node acceptable
- **26-100 examples**: Use `./run_slurm_debug.sh` for better resource management
- **100+ examples**: Use parallel processing with `./run_parallel.sh`
- **500+ examples**: Modify parallel processing settings for optimal performance

### Getting a Debug Node on Babel (Alternative to SLURM)

To run on a GPU compute node interactively:

1. **Start an Interactive Session**:
   ```bash
   salloc --partition=debug --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=01:00:00 \
     srun --pty bash
   ```
   
2. **Navigate and run**:
   ```bash
   cd /home/hrangara/MedCalc/MedCalc-Bench/evaluation
   ./run_debug.sh /data/user_data/hrangara/interactive_session 50
   ```

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
3. For CUDA/memory errors on login nodes, use SLURM job submission
4. Check the script output for detailed error messages
5. For SLURM jobs, check log files in `logs/` directory
6. For parallel processing issues:
   - Run `./test_parallel.sh` first to verify setup
   - Check job logs in `logs/` directory
   - Use `./monitor_jobs.sh` to check job status
   - Verify partition files exist in `partitions/` directory

## Manual Execution

You can also run the commands manually:

**Debug run with custom parameters**:
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate medCalcEnv
python run.py --model meta-llama/Meta-Llama-3-8B-Instruct --prompt zero_shot \
  --enable_attention_analysis --debug_run --num_examples 25 \
  --output_dir /data/user_data/hrangara/manual_run
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