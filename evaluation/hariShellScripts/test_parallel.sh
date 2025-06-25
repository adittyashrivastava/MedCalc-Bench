#!/bin/bash

# Test script for MedCalc-Bench parallel processing setup
# This script tests the partitioning and job creation without submitting to Slurm

set -e

echo "🧪 Testing MedCalc-Bench Parallel Processing Setup"
echo "================================================="

# Configuration (same as run_parallel.sh)
MAX_EXAMPLES=100
NUM_PARTITIONS=4

echo "Configuration:"
echo "  Max examples: $MAX_EXAMPLES"
echo "  Number of partitions: $NUM_PARTITIONS"
echo ""

# Activate conda environment
echo "🔧 Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate medCalcEnv

# Verify environment activation
if [ "$CONDA_DEFAULT_ENV" != "medCalcEnv" ]; then
    echo "❌ Failed to activate medCalcEnv conda environment"
    exit 1
fi

echo "✅ Environment activated: $CONDA_DEFAULT_ENV"
echo ""

# Test 1: Partition the dataset
echo "📊 Test 1: Partitioning dataset..."
python ../partition_dataset.py \
    --dataset ../../dataset/test_data.csv \
    --num_partitions $NUM_PARTITIONS \
    --max_examples $MAX_EXAMPLES \
    --output_dir test_partitions

if [ ! -f "test_partitions/partition_info.txt" ]; then
    echo "❌ Error: Partition info file not created"
    exit 1
fi

echo "✅ Dataset partitioned successfully"
echo ""

# Test 2: Show partition information
echo "📋 Test 2: Partition Information"
cat test_partitions/partition_info.txt
echo ""

# Test 3: Test the modified run.py with partition arguments
echo "🧪 Test 3: Testing run.py with partition arguments (dry run)"

# Test each partition with a very small subset
for i in $(seq 0 $((NUM_PARTITIONS-1))); do
    PARTITION_ID=$(printf "p%02d" $i)

    # Calculate start and end indices for this partition
    EXAMPLES_PER_PARTITION=$(( (MAX_EXAMPLES + NUM_PARTITIONS - 1) / NUM_PARTITIONS ))
    START_IDX=$((i * EXAMPLES_PER_PARTITION))
    END_IDX=$(( (i + 1) * EXAMPLES_PER_PARTITION ))

    # Don't exceed max examples
    if [ $END_IDX -gt $MAX_EXAMPLES ]; then
        END_IDX=$MAX_EXAMPLES
    fi

    # Skip if start index is beyond max examples
    if [ $START_IDX -ge $MAX_EXAMPLES ]; then
        continue
    fi

    echo "  Testing partition $PARTITION_ID (rows $START_IDX-$((END_IDX-1)))"

    # Test the command that would be run (but don't actually run it)
    echo "    Command: python ../run.py --model meta-llama/Meta-Llama-3-8B-Instruct --prompt zero_shot --enable_attention_analysis --start_idx $START_IDX --end_idx $END_IDX --partition_id $PARTITION_ID"

done

echo ""
echo "✅ All tests passed!"
echo ""

# Test 4: Show what files would be created
echo "📁 Test 4: Expected output files"
echo "The following files would be created:"
for i in $(seq 0 $((NUM_PARTITIONS-1))); do
    PARTITION_ID=$(printf "p%02d" $i)
    echo "  ../outputs/meta-llama_Meta-Llama-3-8B-Instruct_zero_shot_partition_${PARTITION_ID}.jsonl"
done

echo ""
echo "🔗 After running, merge with:"
echo "  python ../merge_results.py"
echo ""

# Cleanup test files
echo "🧹 Cleaning up test files..."
rm -rf test_partitions/
echo "✅ Cleanup complete"

echo ""
echo "🎉 Parallel processing setup is ready!"
echo "Run ./run_parallel.sh to start the actual parallel processing"