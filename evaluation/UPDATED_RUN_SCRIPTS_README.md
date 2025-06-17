# Updated Run Scripts with ATTRIEVAL Support

All run scripts have been updated to include ATTRIEVAL (Attention-guided Retrieval for Long-Context Reasoning) analysis alongside the existing attention analysis capabilities.

## 🔄 Updated Scripts

### 1. Debug Run Script (`run_debug.sh`)
**Purpose**: Quick testing with limited data
**Changes**:
- ✅ Added `--enable_attrieval` flag
- ✅ Creates `attrieval_results/` output directory
- ✅ Updated console output messages
- ✅ Updated completion messages

**Usage**:
```bash
# Basic debug run (default: 10 examples)
./run_debug.sh

# Custom number of examples
./run_debug.sh /custom/output/path 25

# Default path with custom examples
./run_debug.sh "" 5
```

**Output Structure**:
```
/data/user_data/hrangara/experiments-{NUM_EXAMPLES}/{TIMESTAMP}/outputs/
├── llm_results/                    # LLM evaluation results
├── attention_results/              # Attention visualizations  
└── attrieval_results/              # 🆕 ATTRIEVAL fact retrieval analysis
    └── calc_{ID}_note_{ID}_row_{N}/
        ├── attrieval_results.json
        ├── attrieval_analysis_report.md
        ├── top_facts_summary.json
        └── attrieval_attention_data.npz
```

### 2. Parallel Run Script (`run_parallel.sh`)
**Purpose**: Distributed processing across multiple GPU nodes
**Changes**:
- ✅ Added `ENABLE_ATTRIEVAL="--enable_attrieval"` configuration variable
- ✅ Updated configuration display to show ATTRIEVAL status
- ✅ All parallel jobs now include ATTRIEVAL analysis

**Usage**:
```bash
# Run parallel processing with ATTRIEVAL
./run_parallel.sh

# Monitor parallel jobs
./monitor_jobs.sh

# Merge results after completion
python merge_results.py
```

**Configuration** (in script):
```bash
MAX_EXAMPLES=100
NUM_PARTITIONS=4
MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
PROMPT="zero_shot"
ENABLE_ATTENTION="--enable_attention_analysis"
ENABLE_ATTRIEVAL="--enable_attrieval"          # 🆕 NEW
```

### 3. Full Run Script (`run_full.sh`)
**Purpose**: Complete evaluation on the entire dataset
**Changes**:
- ✅ Added `--enable_attrieval` flag
- ✅ Updated console messages to mention ATTRIEVAL
- ✅ Updated completion messages with ATTRIEVAL output paths

**Usage**:
```bash
./run_full.sh
# Will prompt for confirmation before starting
```

**Output**:
```
outputs/
├── [model_name]_[prompt_style].jsonl      # Main LLM results
├── attention_analysis/                    # Attention visualizations
└── attrieval_analysis/                    # 🆕 ATTRIEVAL results
```

## 🚀 Quick Start Examples

### Test ATTRIEVAL with 1 Example
```bash
./run_debug.sh "" 1
```

### Run Small Parallel Test
```bash
# Edit run_parallel.sh to set MAX_EXAMPLES=10
./run_parallel.sh
```

### Full Production Run
```bash
./run_full.sh
# Confirm when prompted
```

## 📊 What's New in Each Script

### Debug Script (`run_debug.sh`)

**Before**:
```bash
python run.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --prompt zero_shot \
    --enable_attention_analysis \
    --debug_run \
    --num_examples "$NUM_EXAMPLES" \
    --output_dir "$OUTPUT_DIR"
```

**After**:
```bash
python run.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --prompt zero_shot \
    --enable_attention_analysis \
    --enable_attrieval \                    # 🆕 NEW
    --debug_run \
    --num_examples "$NUM_EXAMPLES" \
    --output_dir "$OUTPUT_DIR"
```

### Parallel Script (`run_parallel.sh`)

**Before**:
```bash
python run.py \\
    --model "$MODEL" \\
    --prompt "$PROMPT" \\
    $ENABLE_ATTENTION \\
    --start_idx $START_IDX \\
    --end_idx $END_IDX \\
    --partition_id "$PARTITION_ID"
```

**After**:
```bash
python run.py \\
    --model "$MODEL" \\
    --prompt "$PROMPT" \\
    $ENABLE_ATTENTION \\
    $ENABLE_ATTRIEVAL \\                    # 🆕 NEW
    --start_idx $START_IDX \\
    --end_idx $END_IDX \\
    --partition_id "$PARTITION_ID"
```

### Full Script (`run_full.sh`)

**Before**:
```bash
python run.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --prompt zero_shot \
    --enable_attention_analysis
```

**After**:
```bash
python run.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --prompt zero_shot \
    --enable_attention_analysis \
    --enable_attrieval                      # 🆕 NEW
```

## 🎯 ATTRIEVAL Analysis Output

Each script now generates ATTRIEVAL analysis for every processed entry:

### Files Generated Per Entry:
1. **`attrieval_results.json`**
   - Comprehensive algorithm results
   - Retrieved facts with attention scores
   - Configuration parameters
   - Execution metadata

2. **`attrieval_analysis_report.md`**
   - Human-readable analysis
   - Input summary (context, question, CoT)
   - Top retrieved facts with explanations
   - Attention insights

3. **`top_facts_summary.json`**
   - Quick access summary
   - Top retrieved facts
   - Calculator/note metadata
   - Timestamp information

4. **`attrieval_attention_data.npz`**
   - Compressed attention weights
   - Fact scores and retriever tokens
   - Efficient storage format

### Example Entry Directory:
```
attrieval_results/calc_1_note_001_row_15/
├── attrieval_results.json           # Comprehensive results
├── attrieval_analysis_report.md     # Human-readable report
├── top_facts_summary.json           # Quick summary
└── attrieval_attention_data.npz     # Compressed data
```

## 🔧 Configuration & Customization

### ATTRIEVAL Parameters
The scripts use these default ATTRIEVAL settings:
```python
AttrievelConfig(
    layer_fraction=0.25,      # Use last 25% of layers
    top_k=50,                 # Top 50 tokens per CoT token
    frequency_threshold=0.99, # Filter attention sinks
    max_facts=10             # Retrieve top 10 facts
)
```

### Model Compatibility
- ✅ **Supported**: Mistral, Llama, local transformer models
- ❌ **Not Supported**: OpenAI models (GPT-3.5, GPT-4)

When using unsupported models, ATTRIEVAL is automatically disabled with a warning.

## ⚡ Performance Considerations

### Memory Usage
ATTRIEVAL adds memory overhead for attention processing:
- **Debug runs**: Minimal impact (1-10 examples)
- **Parallel runs**: Distributed across nodes
- **Full runs**: Monitor memory usage on large datasets

### Runtime Impact
- **Additional time per entry**: ~10-30 seconds (depending on model size)
- **Storage overhead**: ~1-5 MB per entry (compressed)
- **Parallel benefit**: ATTRIEVAL runs alongside attention analysis

### Optimization Tips
1. **Start with debug runs** to verify everything works
2. **Use parallel processing** for large datasets
3. **Monitor GPU memory** during full runs
4. **Consider smaller models** for initial testing

## 🚫 Troubleshooting

### Common Issues
1. **"ATTRIEVAL analysis failed"**
   - Check model compatibility (not OpenAI)
   - Verify sufficient GPU memory

2. **Missing output directories**
   - Ensure script permissions: `chmod +x *.sh`
   - Check disk space in output location

3. **Parallel job failures**
   - Check individual job logs in `logs/` directory
   - Verify Slurm configuration

### Debug Commands
```bash
# Test with minimal example
./run_debug.sh "" 1

# Check ATTRIEVAL integration
python test_attrieval_integration.py

# Monitor parallel jobs
./monitor_jobs.sh
```

## 📈 Expected Benefits

With ATTRIEVAL now integrated into all run scripts:

1. **Better Analysis**: Understand which facts models focus on during medical reasoning
2. **Improved Debugging**: Identify when models miss important clinical information  
3. **Enhanced Transparency**: Make model reasoning more interpretable
4. **Research Insights**: Compare fact retrieval patterns across models

## 🎉 Getting Started

1. **Quick Test**:
   ```bash
   ./run_debug.sh "" 1
   ```

2. **Small Scale**:
   ```bash
   ./run_debug.sh "" 10
   ```

3. **Production**:
   ```bash
   ./run_parallel.sh  # or ./run_full.sh
   ```

All scripts now provide comprehensive medical reasoning analysis with both attention mechanisms and fact retrieval insights! 