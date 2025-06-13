#!/usr/bin/env python3
"""
Dataset Partitioning Script for MedCalc-Bench Parallel Processing

This script partitions the dataset into chunks for parallel processing across multiple Babel instances.
"""

import pandas as pd
import argparse
import math
import os

def partition_dataset(dataset_path, num_partitions, max_examples=None, output_dir="partitions"):
    """
    Partition the dataset into chunks for parallel processing.
    
    Args:
        dataset_path: Path to the CSV dataset
        num_partitions: Number of partitions to create
        max_examples: Maximum number of examples to process (None for all)
        output_dir: Directory to save partition info
    """
    
    # Load dataset
    df = pd.read_csv(dataset_path)
    
    # Limit to max_examples if specified
    if max_examples:
        df = df.head(max_examples)
        print(f"Limited dataset to first {max_examples} examples")
    
    total_examples = len(df)
    examples_per_partition = math.ceil(total_examples / num_partitions)
    
    print(f"Dataset: {dataset_path}")
    print(f"Total examples: {total_examples}")
    print(f"Number of partitions: {num_partitions}")
    print(f"Examples per partition: {examples_per_partition}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate partition information
    partitions = []
    for i in range(num_partitions):
        start_idx = i * examples_per_partition
        end_idx = min((i + 1) * examples_per_partition, total_examples)
        
        if start_idx >= total_examples:
            break
            
        partition_info = {
            "partition_id": f"p{i:02d}",
            "start_idx": start_idx,
            "end_idx": end_idx,
            "num_examples": end_idx - start_idx
        }
        partitions.append(partition_info)
        
        print(f"Partition {partition_info['partition_id']}: rows {start_idx}-{end_idx-1} ({partition_info['num_examples']} examples)")
    
    # Save partition info to file
    partition_file = os.path.join(output_dir, "partition_info.txt")
    with open(partition_file, "w") as f:
        f.write(f"Dataset: {dataset_path}\n")
        f.write(f"Total examples: {total_examples}\n")
        f.write(f"Number of partitions: {len(partitions)}\n")
        f.write(f"Examples per partition: {examples_per_partition}\n\n")
        
        for partition in partitions:
            f.write(f"{partition['partition_id']}: {partition['start_idx']}-{partition['end_idx']-1} ({partition['num_examples']} examples)\n")
    
    print(f"\nPartition information saved to: {partition_file}")
    
    return partitions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Partition MedCalc-Bench dataset for parallel processing")
    parser.add_argument("--dataset", type=str, default="../dataset/test_data.csv", 
                       help="Path to the dataset CSV file")
    parser.add_argument("--num_partitions", type=int, default=4, 
                       help="Number of partitions to create")
    parser.add_argument("--max_examples", type=int, default=None,
                       help="Maximum number of examples to process (default: all)")
    parser.add_argument("--output_dir", type=str, default="partitions",
                       help="Directory to save partition information")
    
    args = parser.parse_args()
    
    partitions = partition_dataset(
        dataset_path=args.dataset,
        num_partitions=args.num_partitions,
        max_examples=args.max_examples,
        output_dir=args.output_dir
    )
    
    print(f"\n✅ Dataset partitioned into {len(partitions)} chunks")
    print("Use the generated partition information with run_parallel.sh") 