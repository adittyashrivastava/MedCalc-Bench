#!/usr/bin/env python3
"""
Merge parallel processing results into a single file
"""

import pandas as pd
import glob
import os
import argparse

def merge_partition_results(pattern="outputs/*partition*.jsonl", output_file=None):
    """Merge all partition result files into a single file"""
    
    # Find all partition files
    partition_files = glob.glob(pattern)
    
    if not partition_files:
        print(f"❌ No partition files found matching pattern: {pattern}")
        return
    
    print(f"📁 Found {len(partition_files)} partition files:")
    for f in sorted(partition_files):
        print(f"  {f}")
    
    # Read and combine all files
    all_results = []
    total_rows = 0
    
    for file_path in sorted(partition_files):
        try:
            df = pd.read_json(file_path, lines=True)
            all_results.append(df)
            total_rows += len(df)
            print(f"  ✅ {file_path}: {len(df)} rows")
        except Exception as e:
            print(f"  ❌ {file_path}: Error - {e}")
    
    if not all_results:
        print("❌ No valid partition files to merge")
        return
    
    # Combine all dataframes
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Sort by Row Number to maintain order
    combined_df = combined_df.sort_values('Row Number').reset_index(drop=True)
    
    # Generate output filename if not provided
    if output_file is None:
        # Extract model and prompt info from first partition file
        first_file = os.path.basename(partition_files[0])
        # Remove partition suffix: model_prompt_partition_pXX.jsonl -> model_prompt.jsonl
        base_name = first_file.replace('_partition_p00', '').replace('_partition_p01', '').replace('_partition_p02', '').replace('_partition_p03', '')
        if '_partition_' in base_name:
            # More generic removal
            parts = base_name.split('_partition_')
            base_name = parts[0] + '.jsonl'
        output_file = f"outputs/merged_{base_name}"
    
    # Save merged results
    combined_df.to_json(output_file, orient='records', lines=True)
    
    print(f"\n✅ Merged results saved to: {output_file}")
    print(f"📊 Total rows: {total_rows}")
    print(f"🎯 Unique calculators: {combined_df['Calculator ID'].nunique()}")
    print(f"📝 Unique notes: {combined_df['Note ID'].nunique()}")
    
    # Show accuracy summary if available
    if 'Result' in combined_df.columns:
        correct = (combined_df['Result'] == 'Correct').sum()
        total = len(combined_df)
        accuracy = correct / total * 100 if total > 0 else 0
        print(f"🎯 Overall Accuracy: {correct}/{total} ({accuracy:.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge parallel processing results")
    parser.add_argument("--pattern", default="outputs/*partition*.jsonl", 
                       help="Pattern to match partition files")
    parser.add_argument("--output", default=None,
                       help="Output file name (auto-generated if not specified)")
    
    args = parser.parse_args()
    merge_partition_results(args.pattern, args.output)
