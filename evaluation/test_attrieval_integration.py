#!/usr/bin/env python3
"""
Test script for ATTRIEVAL integration in run.py

This script tests the ATTRIEVAL functionality with a sample medical case
to verify that the integration works correctly.
"""

import os
import sys
import subprocess
import json

def create_test_csv():
    """Create a minimal test CSV file with one entry for testing."""
    test_data = """Row Number,Calculator Name,Calculator ID,Category,Note ID,Patient Note,Question,Ground Truth Answer,Ground Truth Explanation,Upper Limit,Lower Limit
1,"Test Calculator",1,"Cardiovascular",001,"Patient is a 65-year-old male with a history of hypertension and diabetes. Current medications include metformin 1000mg twice daily and lisinopril 10mg once daily. Recent laboratory results show creatinine 1.2 mg/dL, glucose 150 mg/dL, and HbA1c 7.2%. Blood pressure today is 140/90 mmHg. Patient reports occasional chest pain with exertion.","Calculate the patient's cardiovascular risk score","15","Based on age, hypertension, diabetes, and symptoms",20,10"""
    
    with open("test_data_attrieval.csv", "w") as f:
        f.write(test_data)
    print("✅ Created test CSV file: test_data_attrieval.csv")

def run_attrieval_test():
    """Run the ATTRIEVAL test using the run.py script."""
    print("🚀 Starting ATTRIEVAL integration test...")
    
    # Command to run with ATTRIEVAL enabled
    cmd = [
        "python", "run.py",
        "--model", "gpt2",  # Use small model for testing
        "--prompt", "zero_shot",
        "--enable_attrieval",
        "--debug_run",
        "--num_examples", "1",
        "--output_dir", "test_outputs_attrieval"
    ]
    
    print(f"🔧 Running command: {' '.join(cmd)}")
    
    try:
        # Update the dataset path temporarily
        if os.path.exists("../dataset/test_data.csv"):
            # Backup original
            subprocess.run(["cp", "../dataset/test_data.csv", "../dataset/test_data_backup.csv"])
            # Copy test data
            subprocess.run(["cp", "test_data_attrieval.csv", "../dataset/test_data.csv"])
        
        # Run the test
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        
        print("📋 STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("⚠️  STDERR:")
            print(result.stderr)
        
        print(f"🏁 Return code: {result.returncode}")
        
        # Check if ATTRIEVAL outputs were created
        if os.path.exists("test_outputs_attrieval/attrieval_results"):
            print("✅ ATTRIEVAL output directory created successfully!")
            
            # List what was created
            for root, dirs, files in os.walk("test_outputs_attrieval/attrieval_results"):
                for file in files:
                    file_path = os.path.join(root, file)
                    print(f"   📄 {file_path}")
                    
                    # If it's a JSON file, show a preview
                    if file.endswith('.json'):
                        try:
                            with open(file_path, 'r') as f:
                                data = json.load(f)
                                print(f"      Preview: {str(data)[:200]}...")
                        except Exception as e:
                            print(f"      Could not preview: {e}")
        else:
            print("❌ ATTRIEVAL output directory not found")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False
    
    finally:
        # Restore original dataset if backed up
        if os.path.exists("../dataset/test_data_backup.csv"):
            subprocess.run(["mv", "../dataset/test_data_backup.csv", "../dataset/test_data.csv"])
        
        # Clean up test files
        if os.path.exists("test_data_attrieval.csv"):
            os.remove("test_data_attrieval.csv")

def main():
    """Main test function."""
    print("🧪 ATTRIEVAL Integration Test")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("run.py"):
        print("❌ Error: This script must be run from the evaluation/ directory")
        print("   Please navigate to the evaluation/ directory and run again.")
        sys.exit(1)
    
    # Check if the dataset directory exists
    if not os.path.exists("../dataset"):
        print("❌ Error: Could not find ../dataset directory")
        print("   Please ensure you're running from the evaluation/ directory")
        sys.exit(1)
    
    # Create test data
    create_test_csv()
    
    # Run the test
    success = run_attrieval_test()
    
    if success:
        print("\n🎉 ATTRIEVAL integration test completed successfully!")
        print("   Check the test_outputs_attrieval/attrieval_results/ directory for outputs")
    else:
        print("\n❌ ATTRIEVAL integration test failed")
        print("   Check the error messages above for details")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 