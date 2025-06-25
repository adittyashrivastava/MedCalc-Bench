import re
import os
import json
import tqdm
import argparse
import pandas as pd
import sys
from llm_inference import LLMInference
from evaluate import check_correctness
import math
import numpy as np
import ast
from table_stats import compute_overall_accuracy
from datetime import datetime

# Import HF token configuration
try:
    from hf_config import HF_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACE_TOKEN"] = HF_TOKEN  # Also set this for llm_inference.py compatibility
except ImportError:
    print("Warning: hf_config.py not found. You may need to set HUGGINGFACE_TOKEN manually.")

# Import attention_viz for attention analysis and ATTRIEVAL
from attention_viz import AttentionVisualizer, AttentionExtractor, AttentionAnalyzer, AttrievelRetriever, AttrievelConfig
from attention_viz.utils.helpers import load_model_and_tokenizer

# Load formula catalogue
def load_formula_catalogue():
    """Load the formula catalogue from formula_catalogue.txt"""
    try:
        with open("formula_catalogue.txt", "r") as f:
            catalogue_content = f.read()
        return f"\n\nMEDICAL FORMULA CATALOGUE:\nFor reference, here are commonly used medical formulas and calculations:\n{catalogue_content}"
    except FileNotFoundError:
        print("Warning: formula_catalogue.txt not found. Continuing without formula catalogue.")
        return ""
    except Exception as e:
        print(f"Warning: Could not load formula catalogue: {e}")
        return ""


def zero_shot(note, question, formula_catalogue=""):
    system_msg = 'You are a helpful assistant for calculating a score for a given patient note. Please think step-by-step to solve the question and then generate the required score. Your output should only contain a JSON dict formatted as {"step_by_step_thinking": str(your_step_by_step_thinking_procress_to_solve_the_question), "answer": str(short_and_direct_answer_of_the_question)}.'
    system_msg += formula_catalogue
    user_temp = f'Here is the patient note:\n{note}\n\nHere is the task:\n{question}\n\nPlease directly output the JSON dict formatted as {{"step_by_step_thinking": str(your_step_by_step_thinking_procress_to_solve_the_question), "answer": str(short_and_direct_answer_of_the_question)}}:'
    return system_msg, user_temp

def direct_answer(note, question, formula_catalogue=""):
    system_msg = 'You are a helpful assistant for calculating a score for a given patient note. Please output answer only without any other text. Your output should only contain a JSON dict formatted as {"answer": str(value which is the answer to the question)}.'
    system_msg += formula_catalogue
    user_temp = f'Here is the patient note:\n{note}\n\nHere is the task:\n{question}\n\nPlease directly output the JSON dict formatted as {{"answer": str(value which is the answer to the question)}}:'
    return system_msg, user_temp

def one_shot(note, question, example_note, example_output, formula_catalogue=""):
    system_msg = 'You are a helpful assistant for calculating a score for a given patient note. Please think step-by-step to solve the question and then generate the required score. Your output should only contain a JSON dict formatted as {{"step_by_step_thinking": str(your_step_by_step_thinking_procress_to_solve_the_question), "answer": str(short_and_direct_answer_of_the_question)}}.'
    system_msg += formula_catalogue
    system_msg += f'Here is an example patient note:\n\n{example_note}'
    system_msg += f'\n\nHere is an example task:\n\n{question}'
    system_msg += f'\n\nPlease directly output the JSON dict formatted as {{"step_by_step_thinking": str(your_step_by_step_thinking_procress_to_solve_the_question), "answer": str(value which is the answer to the question)}}:\n\n{json.dumps(example_output)}'
    user_temp = f'Here is the patient note:\n\n{note}\n\nHere is the task:\n\n{question}\n\nPlease directly output the JSON dict formatted as {{"step_by_step_thinking": str(your_step_by_step_thinking_procress_to_solve_the_question), "answer": str(short_and_direct_answer_of_the_question)}}:'
    return system_msg, user_temp

def zero_shot_meditron(note, question, formula_catalogue=""):
    system_msg = '''You are a helpful assistant for calculating a score for a given patient note. Please think step-by-step to solve the question and then generate the required score. Your output should only contain a JSON dict formatted as {"step_by_step_thinking": str(your_step_by_step_thinking_procress_to_solve_the_question), "answer": str(short_and_direct_answer_of_the_question)}. Here is a demonstration (Replace the rationale and "X.XX" with your actual rationale and calculated value):\n\n### User:\nHere is the patient note:\n...\n\nHere is the task:\n...?\n\nPlease directly output the JSON dict formatted as {"step_by_step_thinking": str(your_step_by_step_thinking_procress_to_solve_the_question), "answer": str(short_and_direct_answer_of_the_question)}.\n\n### Assistant:\n{"step_by_step_thinking": rationale, "answer": X.XX}'''
    system_msg += formula_catalogue
    user_temp = f'###User:\nHere is the patient note:\n\n{note}\n\nHere is the task:\n{question}\n\nPlease directly output the JSON dict formatted as {{"step_by_step_thinking": str(your_step_by_step_thinking_procress_to_solve_the_question), "answer": str(short_and_direct_answer_of_the_question)}}.\n\n### Assistant:\n'
    return system_msg, user_temp

def direct_answer_meditron(note, question, formula_catalogue=""):
    system_msg = 'You are a helpful assistant for calculating a score for a given patient note. Please output answer only without any other text. Your output should only contain a JSON dict formatted as {"answer": str(value which is the answer to the question)}. Here is a demonstration (Replace "X.XX" with your the actual calculated value):\n\n### User:\nHere is the patient note:\n...\n\nHere is the task:\n...?\n\nPlease directly output the JSON dict formatted as {"answer": str(value which is the answer to the question)}.\n\n### Assistant:\n{"answer": X.XX}'
    system_msg += formula_catalogue
    user_temp = f'###User:\nHere is the patient note:\n\n{note}\n\nHere is the task:\n\n{question}\n\nPlease directly output the JSON dict formatted as {{"answer": str(value which is the answer to the question)}}.\n\n### Assistant:\n'
    return system_msg, user_temp

def one_shot_meditron(note, question, example_note, example_output, formula_catalogue=""):
    system_msg = 'You are a helpful assistant for calculating a score for a given patient note. Please think step-by-step to solve the question and then generate the required score. Your output should only contain a JSON dict formatted as {{"step_by_step_thinking": str(your_step_by_step_thinking_procress_to_solve_the_question), "answer": str(short_and_direct_answer_of_the_question)}}.'
    system_msg += formula_catalogue
    system_msg += f'\n\n###User:\nHere is an example patient note:\n\n{example_note}'
    system_msg += f'\n\nHere is an example task:\n\n{question}'
    system_msg += f'\n\nPlease directly output the JSON dict formatted as {{"step_by_step_thinking": str(your_step_by_step_thinking_procress_to_solve_the_question), "answer": str(value which is the answer to the question)}}:\n\n### Assistant:\n{json.dumps(example_output)}'
    user_temp = f'###User:\nHere is the patient note:\n{note}\n\nHere is the task:\n{question}\n\nPlease directly output the JSON dict formatted as {{"step_by_step_thinking": str(your_step_by_step_thinking_procress_to_solve_the_question), "answer": str(short_and_direct_answer_of_the_question)}}:\n\n### Assistant:\n'
    return system_msg, user_temp

def extract_answer(answer, calid):

    calid = int(calid)
    extracted_answer = re.findall(r'[Aa]nswer":\s*(.*?)\}', answer)
    matches = re.findall(r'"step_by_step_thinking":\s*"([^"]+)"\s*,\s*"[Aa]nswer"', answer)


    if matches:
    # Select the last match
        last_match = matches[-1]
        explanation = last_match
    else:
        explanation = "No Explanation"


    if len(extracted_answer) == 0:
        extracted_answer = "Not Found"
    else:
        extracted_answer = extracted_answer[-1].strip().strip('"')
        if extracted_answer == "str(short_and_direct_answer_of_the_question)" or extracted_answer == "str(value which is the answer to the question)" or extracted_answer == "X.XX":
            extracted_answer = "Not Found"

    if calid in [13, 68]:
        # Output Type: date
        match = re.search(r"^(0?[1-9]|1[0-2])\/(0?[1-9]|[12][0-9]|3[01])\/(\d{4})", extracted_answer)
        if match:
            month = int(match.group(1))
            day = int(match.group(2))
            year = match.group(3)
            answer = f"{month:02}/{day:02}/{year}"
        else:
            answer = "N/A"

    elif calid in [69]:
        # Output Type: integer (A, B)
        match = re.search(r"\(?[\"\']?(\d+)\s*(weeks?)?[\"\']?,?\s*[\"\']?(\d+)\s*(days?)?[\"\']?\s*\)?", extracted_answer)
        ground_truth = f"({match.group(1)}, {match.group(3)})"
        extracted_answer = extracted_answer.replace("[", "(").replace("]", ")").replace("'", "").replace('"', "")
        match = re.search(r"\(?[\"\']?(\d+)\s*(weeks?)?[\"\']?,?\s*[\"\']?(\d+)\s*(days?)?[\"\']?\s*\)?", extracted_answer)
        if match:
            weeks = match.group(1)
            days = match.group(3)
            answer = f"({weeks}, {days})"
        else:
            answer = "N/A"
    elif calid in [4, 15, 16, 17, 18, 20, 21, 25, 27, 28, 29, 32, 33, 36, 43, 45, 48, 51, 69]:
        # Output Type: integer A
        match = re.search(r"(\d+) out of", extracted_answer)
        if match: # cases like "3 out of 5"
            answer = match.group(1)
        else:
            match = re.search(r"-?\d+(, ?-?\d+)+", extracted_answer)
            if match: # cases like "3, 4, 5"
                answer = str(len(match.group(0).split(",")))
            else:
                # match = re.findall(r"(?<!-)\d+", extracted_answer)
                match = re.findall(r"(-?\d+(\.\d+)?)", extracted_answer)
                # match = re.findall(r"-?\d+", extracted_answer)
                if len(match) > 0: # find the last integer
                    answer = match[-1][0]
                    # answer = match[-1].lstrip("0")
                else:
                    answer = "N/A"
    elif calid in [2,  3,  5,  6,  7,  8,  9, 10, 11, 19, 22, 23, 24, 26, 30, 31, 38, 39, 40, 44, 46, 49, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]:
        # Output Type: decimal
        match = re.search(r"str\((.*)\)", extracted_answer)
        if match: # cases like "str(round((140 * (3.15 - 136) / 1400) * 72.36)"
            expression = match.group(1).replace("^", "**").replace("is odd", "% 2 == 1").replace("is even", "% 2 == 0").replace("sqrt", "math.sqrt").replace(".math", "").replace("weight", "").replace("height", "").replace("mg/dl", "").replace("g/dl", "").replace("mmol/L", "").replace("kg", "").replace("g", "").replace("mEq/L", "")
            expression = expression.split('#')[0] # cases like round(45.5 * 166 - 45.3 + 0.4 * (75 - (45.5 * 166 - 45.3))))) # Calculation: ...
            if expression.count('(') > expression.count(')'): # add missing ')
                expression += ')' * (expression.count('(') - expression.count(')'))
            elif expression.count(')') > expression.count('('): # add missing (
                expression = '(' * (expression.count(')') - expression.count('(')) + expression
            try:
                answer = eval(expression, {"__builtins__": None}, {"min": min, "pow": pow, "round": round, "abs": abs, "int": int, "float": float, "math": math, "np": np, "numpy": np})
            except:
                print(f"Error in evaluating expression: {expression}")
                answer = "N/A"
        else:
            match = re.search(r"(-?\d+(\.\d+)?)\s*mL/min/1.73", extracted_answer)
            if match: # cases like "8.1 mL/min/1.73 m\u00b2"
                answer = eval(match.group(1))
            else:
                match = re.findall(r"(-?\d+(\.\d+)?)\%", extracted_answer)
                if len(match) > 0: # cases like "53.1%"
                    answer = eval(match[-1][0]) / 100
                else:
                    match = re.findall(r"(-?\d+(\.\d+)?)", extracted_answer)
                    if len(match) > 0: # cases like "8.1 mL/min/1.73 m\u00b2" or "11.1"
                        answer = eval(match[-1][0])
                    else:
                        answer = "N/A"
        if answer != "N/A":
            answer = str(answer)

    return answer, explanation

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Parse arguments')
    parser.add_argument('--model', type=str, help='Specify which model you are using. Options are OpenAI/GPT-4, OpenAI/GPT-3.5-turbo, mistralai/Mistral-7B-Instruct-v0.2, mistralai/Mixtral-8x7B-Instruct-v0.1, meta-llama/Meta-Llama-3-8B-Instruct, meta-llama/Meta-Llama-3-70B-Instruct, epfl-llm/meditron-70b, axiong/PMC_LLaMA_13B')
    parser.add_argument('--prompt', type=str, help='Specify prompt type. Options are direct_answer, zero_shot, one_shot')
    parser.add_argument('--enable_attention_analysis', action='store_true', help='Enable attention visualization and analysis for each entry')
    parser.add_argument('--enable_attrieval', action='store_true', help='Enable ATTRIEVAL fact retrieval analysis for each entry')
    parser.add_argument('--enable_formula_catalogue', action='store_true', help='Enable medical formula catalogue augmentation in system prompts')
    parser.add_argument('--debug_run', action='store_true', help='Enable debug run mode. In debug run mode, only process the specified number of rows. In full run mode, process all rows.')
    parser.add_argument('--num_examples', type=int, default=10, help='Number of examples to process in debug mode (default: 10)')
    parser.add_argument('--start_idx', type=int, default=0, help='Starting index for processing (for parallelization)')
    parser.add_argument('--end_idx', type=int, default=None, help='Ending index for processing (for parallelization)')
    parser.add_argument('--partition_id', type=str, default="", help='Partition identifier for output file naming')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory path. If not provided, defaults to current outputs/ directory')

    args = parser.parse_args()

    model_name = args.model
    prompt_style = args.prompt
    enable_attention = args.enable_attention_analysis
    enable_attrieval = args.enable_attrieval
    enable_formula_catalogue = args.enable_formula_catalogue

    # Load formula catalogue if enabled
    formula_catalogue_content = ""
    if enable_formula_catalogue:
        print("📚 Loading medical formula catalogue...")
        formula_catalogue_content = load_formula_catalogue()
        if formula_catalogue_content:
            print("✅ Formula catalogue loaded successfully")
        else:
            print("⚠️  Formula catalogue could not be loaded")
    else:
        print("📚 Formula catalogue augmentation disabled")

    # Setup output directory structure
    if args.output_dir:
        base_output_dir = args.output_dir
        llm_output_dir = os.path.join(base_output_dir, "llm_results")
        attention_output_dir = os.path.join(base_output_dir, "attention_results")
        attrieval_output_dir = os.path.join(base_output_dir, "attrieval_results")

        # Create directories if they don't exist
        os.makedirs(llm_output_dir, exist_ok=True)
        if enable_attention:
            os.makedirs(attention_output_dir, exist_ok=True)
        if enable_attrieval:
            os.makedirs(attrieval_output_dir, exist_ok=True)

        print(f"📁 Using custom output directory: {base_output_dir}")
        print(f"📊 LLM results will be saved to: {llm_output_dir}")
        if enable_attention:
            print(f"👁️  Attention results will be saved to: {attention_output_dir}")
        if enable_attrieval:
            print(f"🔍 ATTRIEVAL results will be saved to: {attrieval_output_dir}")
    else:
        # Default behavior - use current outputs directory
        llm_output_dir = "outputs"
        attention_output_dir = os.path.join("outputs", "attention_analysis")
        attrieval_output_dir = os.path.join("outputs", "attrieval_analysis")
        if not os.path.exists(llm_output_dir):
            os.makedirs(llm_output_dir)

    # Handle partition naming for parallel processing
    if args.partition_id:
        output_path = f"{model_name.replace('/', '_')}_{prompt_style}_partition_{args.partition_id}.jsonl"
    else:
        output_path = f"{model_name.replace('/', '_')}_{prompt_style}.jsonl"

    # Create attention analysis output directory if needed
    if enable_attention:
        os.makedirs(attention_output_dir, exist_ok=True)
        print(f"🔍 Attention analysis enabled. Outputs will be saved to: {attention_output_dir}")

    # Create ATTRIEVAL analysis output directory if needed
    if enable_attrieval:
        os.makedirs(attrieval_output_dir, exist_ok=True)
        print(f"🎯 ATTRIEVAL analysis enabled. Outputs will be saved to: {attrieval_output_dir}")

    # Check for existing results in the LLM output directory
    full_output_path = os.path.join(llm_output_dir, output_path)
    if not os.path.exists(full_output_path):
        existing = None
    else:
        existing = pd.read_json(full_output_path, lines=True)
        existing["Calculator ID"] = existing["Calculator ID"].astype(str)
        existing["Note ID"] = existing["Note ID"].astype(str)

    if "meditron" in model_name.lower():
        zero_shot = zero_shot_meditron
        direct_answer = direct_answer_meditron
        one_shot = one_shot_meditron

    llm = LLMInference(llm_name=model_name)

    # Initialize attention analysis components if enabled
    attention_visualizer = None
    attention_analyzer = None
    attrieval_retriever = None
    shared_extractor = None

    if enable_attention or enable_attrieval:
        # Check if this is an OpenAI model - they don't support attention extraction
        if "openai" in model_name.lower():
            print("⚠️  Attention analysis and ATTRIEVAL are not supported for OpenAI models. Disabling both.")
            enable_attention = False
            enable_attrieval = False
        else:
            try:
                print("🔧 Initializing attention analysis and ATTRIEVAL components...")
                # Get model and tokenizer from LLMInference object
                model = llm.model
                tokenizer = llm.tokenizer

                # Initialize shared attention extractor
                shared_extractor = AttentionExtractor(model, tokenizer)

                # Initialize attention visualization components if needed
                if enable_attention:
                    attention_visualizer = AttentionVisualizer(model, tokenizer)
                    attention_analyzer = AttentionAnalyzer(shared_extractor)

                # Initialize ATTRIEVAL components if needed
                if enable_attrieval:
                    attrieval_config = AttrievelConfig(
                        layer_fraction=0.25,      # Use last 25% of layers
                        top_k=50,                 # Top 50 tokens per CoT token
                        frequency_threshold=0.99, # Filter attention sinks
                        max_facts=10             # Retrieve top 10 facts
                    )
                    attrieval_retriever = AttrievelRetriever(shared_extractor, attrieval_config)

                # Test basic functionality
                print("🧪 Testing basic attention extraction...")
                try:
                    test_result = shared_extractor.extract_attention_weights("Hello world, this is a test.")
                    print(f"✅ Basic test passed - Model has {test_result['num_layers']} layers, {test_result['num_heads']} heads")
                except Exception as e:
                    print(f"❌ Basic test failed: {e}")
                    raise e  # Re-raise to prevent using broken components

                print("✅ Attention analysis and ATTRIEVAL components initialized successfully")
            except Exception as e:
                print(f"⚠️  Failed to initialize attention analysis/ATTRIEVAL: {e}")
                print("Continuing without attention analysis and ATTRIEVAL...")
                enable_attention = False
                enable_attrieval = False

    with open("one_shot_finalized_explanation.json", "r") as file:
        one_shot_json = json.load(file)

    df = pd.read_csv("../dataset/test_data.csv")

    # Handle different processing modes
    if args.debug_run:
        df = df.head(args.num_examples)
        print(f"Debug mode: Processing first {args.num_examples} rows...")
    elif args.start_idx is not None or args.end_idx is not None:
        # Parallel processing mode
        start_idx = args.start_idx if args.start_idx is not None else 0
        end_idx = args.end_idx if args.end_idx is not None else len(df)
        df = df.iloc[start_idx:end_idx]
        print(f"Parallel processing mode: Processing rows {start_idx} to {end_idx-1} (partition {args.partition_id})")
    else:
        print(f"Full processing mode: Processing all {len(df)} rows...")

    print(f"Processing {len(df)} rows...")

    for index in tqdm.tqdm(range(len(df))):

        row = df.iloc[index]

        patient_note = row["Patient Note"]
        question = row["Question"]
        calculator_id = str(row["Calculator ID"])
        note_id = str(row["Note ID"])

        if existing is not None and not args.debug_run:
            if existing[(existing["Calculator ID"] == calculator_id) & (existing["Note ID"] == str(row["Note ID"]))].shape[0] > 0:
                print(f"Skipping Calculator {calculator_id}, Note {note_id} because it already exists")
                continue

        if "pmc_llama" in model_name.lower():
            patient_note = llm.tokenizer.decode(llm.tokenizer.encode(patient_note, add_special_tokens=False)[:256])
        if prompt_style == "zero_shot":
            system, user = zero_shot(patient_note, question, formula_catalogue_content)
        elif prompt_style == "one_shot":
            example = one_shot_json[calculator_id]
            if "meditron" in model_name.lower():
                example["Patient Note"] = llm.tokenizer.decode(llm.tokenizer.encode(example["Patient Note"], add_special_tokens=False)[:512])
                example["Response"]["step_by_step_thinking"] = llm.tokenizer.decode(llm.tokenizer.encode(example["Response"]["step_by_step_thinking"], add_special_tokens=False)[:512])
            elif "pmc_llama" in model_name.lower():
                example["Patient Note"] = llm.tokenizer.decode(llm.tokenizer.encode(example["Patient Note"], add_special_tokens=False)[:256])
                example["Response"]["step_by_step_thinking"] = llm.tokenizer.decode(llm.tokenizer.encode(example["Response"]["step_by_step_thinking"], add_special_tokens=False)[:256])
            system, user = one_shot(patient_note, question, example["Patient Note"], {"step_by_step_thinking": example["Response"]["step_by_step_thinking"], "answer": example["Response"]["answer"]}, formula_catalogue_content)
        elif prompt_style == "direct_answer":
            system, user = direct_answer(patient_note, question, formula_catalogue_content)

        print("System:\n", system)
        print("User:\n", user)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]

        answer = llm.answer(messages)
        print(answer)

        # Perform attention analysis if enabled
        attention_data = {}
        # Perform ATTRIEVAL analysis if enabled
        attrieval_data = {}

        # print attention visualizer
        print(attention_visualizer)
        if enable_attention and attention_visualizer is not None:
            try:
                print(f"🔍 Performing attention analysis for Calculator {calculator_id}, Note {note_id}...")

                # Create organized output directory for this specific entry
                # Use unique identifier: calc_{calculator_id}_note_{note_id}_row_{row_number}
                row_number = int(row["Row Number"])
                entry_dir = os.path.join(attention_output_dir, f"calc_{calculator_id}_note_{note_id}_row_{row_number}")
                os.makedirs(entry_dir, exist_ok=True)
                print(f"📁 Created output directory: {entry_dir}")

                # Combine the full input text for attention analysis
                full_input_text = f"System: {system}\n\nUser: {user}"
                print(f"📝 Input text length: {len(full_input_text)} characters")

                # Get model info for dynamic layer/head selection
                try:
                    print("🔍 Detecting model architecture...")
                    # Extract a small sample to get model dimensions
                    sample_data = shared_extractor.extract_attention_weights("Sample text for architecture detection.")
                    num_layers = sample_data["num_layers"]
                    num_heads = sample_data["num_heads"]

                    # Select middle layer and a valid head
                    target_layer = min(6, num_layers - 1)  # Use layer 6 or the last layer if fewer than 7 layers
                    target_head = min(4, num_heads - 1)    # Use head 4 or the last head if fewer than 5 heads

                    print(f"📊 Model has {num_layers} layers and {num_heads} heads per layer")


                    # print(f"🎯 Using layer {target_layer} and head {target_head} for analysis")

                    # # Also select layer range for comparison
                    # layer_indices = [0, max(1, num_layers//4), max(2, num_layers//2), max(3, num_layers-1)]
                    # layer_indices = sorted(list(set(layer_indices)))  # Remove duplicates and sort
                    # print(f"📐 Using layers {layer_indices} for comparison")

                except Exception as e:
                    print(f"⚠️ Could not detect model architecture: {e}")
                    # Fallback to default values
                    target_layer = 6
                    target_head = 4
                    layer_indices = [0, 3, 6, 9]
                    print(f"🔄 Falling back to default layer {target_layer}, head {target_head}")

                # 1. Export essential attention data first (memory-efficient format)
                try:
                    print("💾 Exporting essential attention data early (memory-efficient)...")
                    # Extract just the essential attention weights for the target layer/head
                    essential_data = shared_extractor.extract_attention_weights(full_input_text)

                    # Only keep the target layer and a few key layers to save memory
                    essential_layers = [0, target_layer, essential_data["num_layers"]-1]  # First, target, last
                    filtered_attention = []
                    for i, layer_attn in enumerate(essential_data["attention_weights"]):
                        if i in essential_layers:
                            # Only keep a subset of heads to save memory
                            max_heads_to_keep = min(8, layer_attn.shape[0])  # Keep max 8 heads
                            filtered_attention.append(layer_attn[:max_heads_to_keep])

                    # Create memory-efficient export
                    essential_export = {
                        "tokens": essential_data["tokens"],
                        "num_layers": len(essential_layers),
                        "num_heads": max_heads_to_keep,
                        "target_layer": target_layer,
                        "target_head": target_head,
                        "sequence_length": essential_data["sequence_length"],
                        "calculator_id": calculator_id,
                        "note_id": note_id,
                        "row_number": row_number
                    }

                    # Save as compressed numpy format (much more memory efficient than JSON)
                    np.savez_compressed(
                        os.path.join(entry_dir, "essential_attention_data.npz"),
                        attention_weights=np.array(filtered_attention, dtype=object),
                        **essential_export
                    )
                    attention_data["essential_data"] = "essential_attention_data.npz"
                    print("✅ Essential attention data export completed")

                    # Clear memory
                    del essential_data, filtered_attention, essential_export
                    import gc
                    gc.collect()

                except Exception as e:
                    print(f"❌ Essential attention data export failed: {e}")
                    import traceback
                    print(f"Traceback:\n{traceback.format_exc()}")

                # # 2. Basic attention visualization
                # try:
                #     print("📊 Generating basic attention visualization...")
                #     attention_visualizer.visualize_attention(
                #         text=full_input_text,
                #         layer=target_layer,
                #         head=target_head,
                #         save_path=os.path.join(entry_dir, "basic_attention.html"),
                #         interactive=True
                #     )
                #     attention_data["basic_visualization"] = "basic_attention.html"
                #     print("✅ Basic attention visualization completed")
                # except Exception as e:
                #     print(f"❌ Basic attention visualization failed: {e}")
                #     import traceback
                #     print(f"Traceback:\n{traceback.format_exc()}")

                # # 3. Attention heatmap
                # try:
                #     print("🔥 Generating attention heatmap...")
                #     attention_visualizer.plot_attention_heatmap(
                #         text=full_input_text,
                #         layer=target_layer,
                #         head=target_head,
                #         title=f"Medical Calculation - Calculator {calculator_id}, Note {note_id}, Row {row_number}",
                #         save_path=os.path.join(entry_dir, "attention_heatmap.png")
                #     )
                #     attention_data["heatmap"] = "attention_heatmap.png"
                #     print("✅ Attention heatmap completed")

                #     # Clear any cached attention data to free memory
                #     import gc
                #     gc.collect()

                # except Exception as e:
                #     print(f"❌ Attention heatmap failed: {e}")
                #     import traceback
                #     print(f"Traceback:\n{traceback.format_exc()}")

                # # 4. Layer comparison (use fewer layers to save memory)
                # try:
                #     print("📐 Generating layer comparison...")
                #     # Limit to 4 layers max to reduce memory usage
                #     limited_layers = layer_indices[:4]
                #     attention_visualizer.compare_layers(
                #         text=full_input_text,
                #         layers=limited_layers,
                #         save_path=os.path.join(entry_dir, "layer_comparison.png")
                #     )
                #     attention_data["layer_comparison"] = "layer_comparison.png"
                #     print("✅ Layer comparison completed")

                #     # Clear memory
                #     import gc
                #     gc.collect()

                # except Exception as e:
                #     print(f"❌ Layer comparison failed: {e}")
                #     import traceback
                #     print(f"Traceback:\n{traceback.format_exc()}")

                # # 7. Generate comprehensive report
                # try:
                #     print("📋 Generating comprehensive analysis report...")
                #     attention_analyzer.export_analysis_report(
                #         text=full_input_text,
                #         save_path=os.path.join(entry_dir, "attention_report.md")
                #     )
                #     attention_data["analysis_report"] = "attention_report.md"
                #     print("✅ Analysis report completed")
                # except Exception as e:
                #     print(f"❌ Analysis report failed: {e}")
                #     import traceback
                #     print(f"Traceback:\n{traceback.format_exc()}")

                # # 8. Head specialization analysis (with multiple related texts)
                # try:
                #     print("🎯 Performing head specialization analysis...")
                #     related_texts = [
                #         full_input_text,
                #         f"Patient Note: {patient_note}",
                #         f"Question: {question}",
                #         f"Medical calculation for {row.get('Calculator Name', 'Unknown Calculator')}"
                #     ]

                #     head_analysis = attention_visualizer.analyze_head_specialization(
                #         texts=related_texts,
                #         layer=target_layer
                #     )

                #     attention_visualizer.plot_head_specialization(
                #         head_analysis,
                #         save_path=os.path.join(entry_dir, "head_specialization.png")
                #     )
                #     attention_data["head_specialization"] = "head_specialization.png"
                #     print("✅ Head specialization analysis completed")
                # except Exception as e:
                #     print(f"❌ Head specialization analysis failed: {e}")
                #     import traceback
                #     print(f"Traceback:\n{traceback.format_exc()}")

                # 9. Save attention analysis summary as JSON
                try:
                    attention_summary = {
                        "calculator_id": calculator_id,
                        "note_id": note_id,
                        "row_number": row_number,
                        "calculator_name": row.get("Calculator Name", "Unknown"),
                        "category": row.get("Category", "Unknown"),
                        "model_name": model_name,
                        "prompt_style": prompt_style,
                        "input_length": len(full_input_text),
                        "patient_note_length": len(patient_note),
                        "question_length": len(question),
                        "attention_files": attention_data,
                        "timestamp": datetime.now().isoformat(),
                        "unique_identifier": f"calc_{calculator_id}_note_{note_id}_row_{row_number}"
                    }

                    with open(os.path.join(entry_dir, "attention_summary.json"), "w") as f:
                        json.dump(attention_summary, f, indent=2)

                    print(f"Attention analysis completed for Calculator {calculator_id}, Note {note_id}, Row {row_number}")
                    print(f"Files saved to: {entry_dir}")

                except Exception as e:
                    print(f"Attention summary save failed: {e}")

            except Exception as e:
                print(f"Attention analysis failed for Calculator {calculator_id}, Note {note_id}: {e}")

        # Perform ATTRIEVAL analysis if enabled
        if enable_attrieval and attrieval_retriever is not None:
            try:
                print(f"🎯 Performing ATTRIEVAL analysis for Calculator {calculator_id}, Note {note_id}...")

                # Create organized output directory for this specific entry
                # Use unique identifier: calc_{calculator_id}_note_{note_id}_row_{row_number}
                row_number = int(row["Row Number"])
                entry_dir = os.path.join(attrieval_output_dir, f"calc_{calculator_id}_note_{note_id}_row_{row_number}")
                os.makedirs(entry_dir, exist_ok=True)
                print(f"📁 Created ATTRIEVAL output directory: {entry_dir}")

                # Extract the raw LLM response for CoT analysis
                # Use the full answer for ATTRIEVAL
                raw_answer, explanation = extract_answer(answer, int(calculator_id))

                # For ATTRIEVAL, we need:
                # 1. Context (patient note)
                # 2. Question
                # 3. CoT response (the step-by-step thinking part)
                context = patient_note
                question_text = question

                # Try to extract step-by-step thinking from the answer
                cot_response = explanation if explanation != "No Explanation" else answer

                print(f"📝 Context length: {len(context)} characters")
                print(f"❓ Question length: {len(question_text)} characters")
                print(f"🧠 CoT response length: {len(cot_response)} characters")

                # Run ATTRIEVAL fact retrieval
                print("🔍 Running ATTRIEVAL fact retrieval...")
                retrieval_result = attrieval_retriever.retrieve_facts(
                    context=context,
                    question=question_text,
                    cot_response=cot_response,
                    use_cross_evaluation=True
                )

                # Save detailed ATTRIEVAL results
                try:
                    print("💾 Saving ATTRIEVAL results...")

                    # 1. Export comprehensive results as JSON
                    attrieval_retriever.export_retrieval_result(
                        retrieval_result,
                        os.path.join(entry_dir, "attrieval_results.json")
                    )
                    attrieval_data["results_json"] = "attrieval_results.json"
                    print("✅ ATTRIEVAL results JSON saved")

                    # 2. Generate and save human-readable report
                    readable_report = attrieval_retriever.visualize_retrieved_facts(retrieval_result)
                    with open(os.path.join(entry_dir, "attrieval_analysis_report.md"), "w") as f:
                        f.write(readable_report)
                    attrieval_data["analysis_report"] = "attrieval_analysis_report.md"
                    print("✅ ATTRIEVAL analysis report saved")

                    # 3. Save top retrieved facts as separate JSON for easy access
                    top_facts_summary = {
                        "calculator_id": calculator_id,
                        "note_id": note_id,
                        "row_number": row_number,
                        "question": question_text,
                        "top_retrieved_facts": retrieval_result['retrieved_facts'],
                        "num_facts_retrieved": len(retrieval_result['retrieved_facts']),
                        "attrieval_config": retrieval_result['config'],
                        "context_length": len(context),
                        "cot_length": len(cot_response),
                        "timestamp": datetime.now().isoformat()
                    }

                    with open(os.path.join(entry_dir, "top_facts_summary.json"), "w") as f:
                        json.dump(top_facts_summary, f, indent=2)
                    attrieval_data["top_facts"] = "top_facts_summary.json"
                    print("✅ Top facts summary saved")

                    # 4. Save aggregated attention data (compressed)
                    np.savez_compressed(
                        os.path.join(entry_dir, "attrieval_attention_data.npz"),
                        aggregated_attention=retrieval_result['aggregated_attention'],
                        retriever_tokens=retrieval_result['retriever_tokens'],
                        fact_scores=retrieval_result['fact_scores']
                    )
                    attrieval_data["attention_data"] = "attrieval_attention_data.npz"
                    print("✅ ATTRIEVAL attention data saved")

                    print(f"🎯 ATTRIEVAL analysis completed for Calculator {calculator_id}, Note {note_id}, Row {row_number}")
                    print(f"📊 Retrieved {len(retrieval_result['retrieved_facts'])} top facts")
                    print(f"📁 Files saved to: {entry_dir}")

                except Exception as e:
                    print(f"❌ ATTRIEVAL results save failed: {e}")
                    import traceback
                    print(f"Traceback:\n{traceback.format_exc()}")

            except Exception as e:
                print(f"❌ ATTRIEVAL analysis failed for Calculator {calculator_id}, Note {note_id}: {e}")
                import traceback
                print(f"Traceback:\n{traceback.format_exc()}")

        try:
            answer_value, explanation = extract_answer(answer, int(calculator_id))

            print(answer_value)
            print(explanation)

            correctness = check_correctness(answer_value, row["Ground Truth Answer"], calculator_id, row["Upper Limit"], row["Lower Limit"])

            status = "Correct" if correctness else "Incorrect"

            outputs = {
                "Row Number": int(row["Row Number"]),
                "Calculator Name": row["Calculator Name"],
                "Calculator ID": calculator_id,
                "Category": row["Category"],
                "Note ID": note_id,
                "Patient Note": patient_note,
                "Question": question,
                "LLM Answer": answer_value,
                "LLM Explanation": explanation,
                "Ground Truth Answer": row["Ground Truth Answer"],
                "Ground Truth Explanation": row["Ground Truth Explanation"],
                "Result": status,
                "Timestamp": datetime.now().isoformat(),
                "Unique_Identifier": f"calc_{calculator_id}_note_{note_id}_row_{int(row['Row Number'])}"
            }

            # Add attention analysis info to outputs if available
            if enable_attention and attention_data:
                outputs["Attention_Analysis_Directory"] = f"attention_results/calc_{calculator_id}_note_{note_id}_row_{int(row['Row Number'])}"
                outputs["Attention_Files_Generated"] = list(attention_data.keys())

            # Add ATTRIEVAL analysis info to outputs if available
            if enable_attrieval and attrieval_data:
                outputs["ATTRIEVAL_Analysis_Directory"] = f"attrieval_results/calc_{calculator_id}_note_{note_id}_row_{int(row['Row Number'])}"
                outputs["ATTRIEVAL_Files_Generated"] = list(attrieval_data.keys())

            if prompt_style == "direct_answer":
                outputs["LLM Explanation"] = "N/A"


        except Exception as e:
            outputs = {
                "Row Number": int(row["Row Number"]),
                "Calculator Name": row["Calculator Name"],
                "Calculator ID": calculator_id,
                "Category": row["Category"],
                "Note ID": note_id,
                "Patient Note": patient_note,
                "Question": question,
                "LLM Answer": str(e),
                "LLM Explanation": str(e),
                "Ground Truth Answer": row["Ground Truth Answer"],
                "Ground Truth Explanation": row["Ground Truth Explanation"],
                "Result": "Incorrect",
                "Timestamp": datetime.now().isoformat(),
                "Unique_Identifier": f"calc_{calculator_id}_note_{note_id}_row_{int(row['Row Number'])}"
            }
            print(f"error in {calculator_id} {note_id}: "  + str(e))

            if prompt_style == "direct_answer":
                outputs["LLM Explanation"] = "N/A"

        print(outputs)

        # Save to the appropriate LLM results directory
        with open(full_output_path, "a") as f:
            f.write(json.dumps(outputs) + "\n")

    # Use the full output path for accuracy computation
    compute_overall_accuracy(os.path.basename(full_output_path), model_name, prompt_style)

    # Print analysis summary if enabled
    print(f"\nProcessing completed!")
    print(f"📊 LLM results saved to: {llm_output_dir}")

    if enable_attention:
        print(f"👁️  Attention visualizations saved to: {attention_output_dir}")
        print("   Generated attention files for each entry:")
        # print("   - basic_attention.html (interactive visualization)")
        # print("   - attention_heatmap.png (static heatmap)")
        # print("   - layer_comparison.png (multi-layer comparison)")
        print("   - essential_attention_data.npz (compressed attention weights)")
        # print("   - attention_report.md (comprehensive analysis)")
        # print("   - head_specialization.png (head analysis)")
        print("   - attention_summary.json (metadata and file list)")

    if enable_attrieval:
        print(f"🎯 ATTRIEVAL analysis results saved to: {attrieval_output_dir}")
        print("   Generated ATTRIEVAL files for each entry:")
        print("   - attrieval_results.json (comprehensive retrieval results)")
        print("   - attrieval_analysis_report.md (human-readable analysis)")
        print("   - top_facts_summary.json (top retrieved facts)")
        print("   - attrieval_attention_data.npz (attention weights and scores)")

    if enable_attention and enable_attrieval:
        print("\n🚀 Both attention analysis and ATTRIEVAL fact retrieval completed!")
    elif enable_attention:
        print("\n👁️  Attention analysis completed!")
    elif enable_attrieval:
        print("\n🎯 ATTRIEVAL fact retrieval analysis completed!")








