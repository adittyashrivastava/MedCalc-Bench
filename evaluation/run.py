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

# Import HF token configuration
try:
    from hf_config import HF_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACE_TOKEN"] = HF_TOKEN  # Also set this for llm_inference.py compatibility
except ImportError:
    print("Warning: hf_config.py not found. You may need to set HUGGINGFACE_TOKEN manually.")

# Import attention_viz for attention analysis
from attention_viz import AttentionVisualizer, AttentionExtractor, AttentionAnalyzer
from attention_viz.utils.helpers import load_model_and_tokenizer


def zero_shot(note, question):
    system_msg = 'You are a helpful assistant for calculating a score for a given patient note. Please think step-by-step to solve the question and then generate the required score. Your output should only contain a JSON dict formatted as {"step_by_step_thinking": str(your_step_by_step_thinking_procress_to_solve_the_question), "answer": str(short_and_direct_answer_of_the_question)}.'
    user_temp = f'Here is the patient note:\n{note}\n\nHere is the task:\n{question}\n\nPlease directly output the JSON dict formatted as {{"step_by_step_thinking": str(your_step_by_step_thinking_procress_to_solve_the_question), "answer": str(short_and_direct_answer_of_the_question)}}:'
    return system_msg, user_temp

def direct_answer(note, question):
    system_msg = 'You are a helpful assistant for calculating a score for a given patient note. Please output answer only without any other text. Your output should only contain a JSON dict formatted as {"answer": str(value which is the answer to the question)}.'
    user_temp = f'Here is the patient note:\n{note}\n\nHere is the task:\n{question}\n\nPlease directly output the JSON dict formatted as {{"answer": str(value which is the answer to the question)}}:'
    return system_msg, user_temp

def one_shot(note, question, example_note, example_output):
    system_msg = 'You are a helpful assistant for calculating a score for a given patient note. Please think step-by-step to solve the question and then generate the required score. Your output should only contain a JSON dict formatted as {{"step_by_step_thinking": str(your_step_by_step_thinking_procress_to_solve_the_question), "answer": str(short_and_direct_answer_of_the_question)}}.'
    system_msg += f'Here is an example patient note:\n\n{example_note}'
    system_msg += f'\n\nHere is an example task:\n\n{question}'
    system_msg += f'\n\nPlease directly output the JSON dict formatted as {{"step_by_step_thinking": str(your_step_by_step_thinking_procress_to_solve_the_question), "answer": str(value which is the answer to the question)}}:\n\n{json.dumps(example_output)}'
    user_temp = f'Here is the patient note:\n\n{note}\n\nHere is the task:\n\n{question}\n\nPlease directly output the JSON dict formatted as {{"step_by_step_thinking": str(your_step_by_step_thinking_procress_to_solve_the_question), "answer": str(short_and_direct_answer_of_the_question)}}:'
    return system_msg, user_temp

def zero_shot_meditron(note, question):
    system_msg = '''You are a helpful assistant for calculating a score for a given patient note. Please think step-by-step to solve the question and then generate the required score. Your output should only contain a JSON dict formatted as {"step_by_step_thinking": str(your_step_by_step_thinking_procress_to_solve_the_question), "answer": str(short_and_direct_answer_of_the_question)}. Here is a demonstration (Replace the rationale and "X.XX" with your actual rationale and calculated value):\n\n### User:\nHere is the patient note:\n...\n\nHere is the task:\n...?\n\nPlease directly output the JSON dict formatted as {"step_by_step_thinking": str(your_step_by_step_thinking_procress_to_solve_the_question), "answer": str(short_and_direct_answer_of_the_question)}.\n\n### Assistant:\n{"step_by_step_thinking": rationale, "answer": X.XX}'''
    user_temp = f'###User:\nHere is the patient note:\n\n{note}\n\nHere is the task:\n{question}\n\nPlease directly output the JSON dict formatted as {{"step_by_step_thinking": str(your_step_by_step_thinking_procress_to_solve_the_question), "answer": str(short_and_direct_answer_of_the_question)}}.\n\n### Assistant:\n'
    return system_msg, user_temp

def direct_answer_meditron(note, question):
    system_msg = 'You are a helpful assistant for calculating a score for a given patient note. Please output answer only without any other text. Your output should only contain a JSON dict formatted as {"answer": str(value which is the answer to the question)}. Here is a demonstration (Replace "X.XX" with your the actual calculated value):\n\n### User:\nHere is the patient note:\n...\n\nHere is the task:\n...?\n\nPlease directly output the JSON dict formatted as {"answer": str(value which is the answer to the question)}.\n\n### Assistant:\n{"answer": X.XX}'
    user_temp = f'###User:\nHere is the patient note:\n\n{note}\n\nHere is the task:\n\n{question}\n\nPlease directly output the JSON dict formatted as {{"answer": str(value which is the answer to the question)}}.\n\n### Assistant:\n'
    return system_msg, user_temp

def one_shot_meditron(note, question, example_note, example_output):
    system_msg = 'You are a helpful assistant for calculating a score for a given patient note. Please think step-by-step to solve the question and then generate the required score. Your output should only contain a JSON dict formatted as {{"step_by_step_thinking": str(your_step_by_step_thinking_procress_to_solve_the_question), "answer": str(short_and_direct_answer_of_the_question)}}.'
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
    parser.add_argument('--debug_run', action='store_true', help='Enable debug run mode. In debug run mode, only process the first 10 rows. In full run mode, process all rows.')
    parser.add_argument('--start_idx', type=int, default=0, help='Starting index for processing (for parallelization)')
    parser.add_argument('--end_idx', type=int, default=None, help='Ending index for processing (for parallelization)')
    parser.add_argument('--partition_id', type=str, default="", help='Partition identifier for output file naming')

    args = parser.parse_args()

    model_name = args.model
    prompt_style = args.prompt
    enable_attention = args.enable_attention_analysis

    # Handle partition naming for parallel processing
    if args.partition_id:
        output_path = f"{model_name.replace('/', '_')}_{prompt_style}_partition_{args.partition_id}.jsonl"
    else:
        output_path = f"{model_name.replace('/', '_')}_{prompt_style}.jsonl"

    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    # Create attention analysis output directory if needed
    if enable_attention:
        attention_output_dir = os.path.join("outputs", "attention_analysis")
        os.makedirs(attention_output_dir, exist_ok=True)
        print(f"🔍 Attention analysis enabled. Outputs will be saved to: {attention_output_dir}")

    if not os.path.exists(os.path.join("outputs", output_path)):
        existing = None
    else:
        existing = pd.read_json(os.path.join("outputs", output_path), lines=True)
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
    if enable_attention:
        # Check if this is an OpenAI model - they don't support attention extraction
        if "openai" in model_name.lower():
            print("⚠️  Attention analysis is not supported for OpenAI models. Disabling attention analysis.")
            enable_attention = False
        else:
            try:
                print("🔧 Initializing attention analysis components...")
                # Get model and tokenizer from LLMInference object
                model = llm.model
                tokenizer = llm.tokenizer

                # Initialize attention visualization components
                attention_visualizer = AttentionVisualizer(model, tokenizer)
                attention_extractor = AttentionExtractor(model, tokenizer)
                attention_analyzer = AttentionAnalyzer(attention_extractor)

                # Test basic functionality
                print("🧪 Testing basic attention extraction...")
                try:
                    test_result = attention_extractor.extract_attention_weights("Hello world, this is a test.")
                    print(f"✅ Basic test passed - Model has {test_result['num_layers']} layers, {test_result['num_heads']} heads")
                except Exception as e:
                    print(f"❌ Basic test failed: {e}")
                    raise e  # Re-raise to prevent using broken components

                print("✅ Attention analysis components initialized successfully")
            except Exception as e:
                print(f"⚠️  Failed to initialize attention analysis: {e}")
                print("Continuing without attention analysis...")
                enable_attention = False

    with open("one_shot_finalized_explanation.json", "r") as file:
        one_shot_json = json.load(file)

    df = pd.read_csv("../dataset/test_data.csv")

    # Handle different processing modes
    if args.debug_run:
        df = df.head(10)
        print(f"Debug mode: Processing first 10 rows...")
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

        if existing is not None:
            if existing[(existing["Calculator ID"] == calculator_id) & (existing["Note ID"] == str(row["Note ID"]))].shape[0] > 0:
                continue

        if "pmc_llama" in model_name.lower():
            patient_note = llm.tokenizer.decode(llm.tokenizer.encode(patient_note, add_special_tokens=False)[:256])
        if prompt_style == "zero_shot":
            system, user = zero_shot(patient_note, question)
        elif prompt_style == "one_shot":
            example = one_shot_json[calculator_id]
            if "meditron" in model_name.lower():
                example["Patient Note"] = llm.tokenizer.decode(llm.tokenizer.encode(example["Patient Note"], add_special_tokens=False)[:512])
                example["Response"]["step_by_step_thinking"] = llm.tokenizer.decode(llm.tokenizer.encode(example["Response"]["step_by_step_thinking"], add_special_tokens=False)[:512])
            elif "pmc_llama" in model_name.lower():
                example["Patient Note"] = llm.tokenizer.decode(llm.tokenizer.encode(example["Patient Note"], add_special_tokens=False)[:256])
                example["Response"]["step_by_step_thinking"] = llm.tokenizer.decode(llm.tokenizer.encode(example["Response"]["step_by_step_thinking"], add_special_tokens=False)[:256])
            system, user = one_shot(patient_note, question, example["Patient Note"], {"step_by_step_thinking": example["Response"]["step_by_step_thinking"], "answer": example["Response"]["answer"]})
        elif prompt_style == "direct_answer":
            system, user = direct_answer(patient_note, question)

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
        if enable_attention and attention_visualizer is not None:
            try:
                print(f"🔍 Performing attention analysis for Calculator {calculator_id}, Note {note_id}...")

                # Create organized output directory for this specific entry
                entry_dir = os.path.join(attention_output_dir, f"calc_{calculator_id}_note_{note_id}")
                os.makedirs(entry_dir, exist_ok=True)
                print(f"📁 Created output directory: {entry_dir}")

                # Combine the full input text for attention analysis
                full_input_text = f"System: {system}\n\nUser: {user}"
                print(f"📝 Input text length: {len(full_input_text)} characters")

                # Get model info for dynamic layer/head selection
                try:
                    print("🔍 Detecting model architecture...")
                    # Extract a small sample to get model dimensions
                    sample_data = attention_extractor.extract_attention_weights("Sample text for architecture detection.")
                    num_layers = sample_data["num_layers"]
                    num_heads = sample_data["num_heads"]
                    
                    # Select middle layer and a valid head
                    target_layer = min(6, num_layers - 1)  # Use layer 6 or the last layer if fewer than 7 layers
                    target_head = min(4, num_heads - 1)    # Use head 4 or the last head if fewer than 5 heads
                    
                    print(f"📊 Model has {num_layers} layers and {num_heads} heads per layer")
                    print(f"🎯 Using layer {target_layer} and head {target_head} for analysis")
                    
                    # Also select layer range for comparison
                    layer_indices = [0, max(1, num_layers//4), max(2, num_layers//2), max(3, num_layers-1)]
                    layer_indices = sorted(list(set(layer_indices)))  # Remove duplicates and sort
                    print(f"📐 Using layers {layer_indices} for comparison")
                    
                except Exception as e:
                    print(f"⚠️ Could not detect model architecture: {e}")
                    # Fallback to default values
                    target_layer = 6
                    target_head = 4
                    layer_indices = [0, 3, 6, 9]
                    print(f"🔄 Falling back to default layer {target_layer}, head {target_head}")

                # 1. Basic attention visualization
                try:
                    print("📊 Generating basic attention visualization...")
                    attention_visualizer.visualize_attention(
                        text=full_input_text,
                        layer=target_layer,
                        head=target_head,
                        save_path=os.path.join(entry_dir, "basic_attention.html"),
                        interactive=True
                    )
                    attention_data["basic_visualization"] = "basic_attention.html"
                    print("✅ Basic attention visualization completed")
                except Exception as e:
                    print(f"❌ Basic attention visualization failed: {e}")
                    import traceback
                    print(f"Traceback:\n{traceback.format_exc()}")

                # 2. Attention heatmap
                try:
                    print("🔥 Generating attention heatmap...")
                    attention_visualizer.plot_attention_heatmap(
                        text=full_input_text,
                        layer=target_layer,
                        head=target_head,
                        title=f"Medical Calculation - Calculator {calculator_id}",
                        save_path=os.path.join(entry_dir, "attention_heatmap.png")
                    )
                    attention_data["heatmap"] = "attention_heatmap.png"
                    print("✅ Attention heatmap completed")
                except Exception as e:
                    print(f"❌ Attention heatmap failed: {e}")
                    import traceback
                    print(f"Traceback:\n{traceback.format_exc()}")

                # 3. Layer comparison
                try:
                    print("📐 Generating layer comparison...")
                    attention_visualizer.compare_layers(
                        text=full_input_text,
                        layers=layer_indices,
                        save_path=os.path.join(entry_dir, "layer_comparison.png")
                    )
                    attention_data["layer_comparison"] = "layer_comparison.png"
                    print("✅ Layer comparison completed")
                except Exception as e:
                    print(f"❌ Layer comparison failed: {e}")
                    import traceback
                    print(f"Traceback:\n{traceback.format_exc()}")

                # 4. Get attention statistics
                try:
                    print("📈 Computing attention statistics...")
                    stats = attention_visualizer.get_attention_stats(full_input_text)
                    attention_data["statistics"] = {
                        "overall_entropy": stats.get('overall_stats', {}).get('entropy', 0),
                        "overall_sparsity": stats.get('overall_stats', {}).get('sparsity', 0),
                        "max_attention": stats.get('overall_stats', {}).get('max_attention', 0)
                    }
                    print("✅ Attention statistics completed")
                except Exception as e:
                    print(f"❌ Attention statistics failed: {e}")
                    import traceback
                    print(f"Traceback:\n{traceback.format_exc()}")

                # 5. Advanced analysis with AttentionAnalyzer
                try:
                    print("🧠 Performing advanced positional analysis...")
                    pos_analysis = attention_analyzer.analyze_positional_attention(full_input_text, layer=target_layer)
                    attention_data["positional_analysis"] = {
                        "attention_pattern_type": pos_analysis.get("aggregate_analysis", {}).get("attention_pattern_type", "unknown"),
                        "local_ratio": pos_analysis.get("aggregate_analysis", {}).get("average_local_ratio", 0),
                        "global_ratio": pos_analysis.get("aggregate_analysis", {}).get("average_global_ratio", 0)
                    }
                    print("✅ Positional analysis completed")
                except Exception as e:
                    print(f"❌ Positional analysis failed: {e}")
                    import traceback
                    print(f"Traceback:\n{traceback.format_exc()}")

                # 6. Export attention data
                try:
                    print("💾 Exporting raw attention data...")
                    attention_visualizer.export_attention_data(
                        text=full_input_text,
                        format="json",
                        save_path=os.path.join(entry_dir, "attention_data.json")
                    )
                    attention_data["raw_data"] = "attention_data.json"
                    print("✅ Attention data export completed")
                except Exception as e:
                    print(f"❌ Attention data export failed: {e}")
                    import traceback
                    print(f"Traceback:\n{traceback.format_exc()}")

                # 7. Generate comprehensive report
                try:
                    print("📋 Generating comprehensive analysis report...")
                    attention_analyzer.export_analysis_report(
                        text=full_input_text,
                        save_path=os.path.join(entry_dir, "attention_report.md")
                    )
                    attention_data["analysis_report"] = "attention_report.md"
                    print("✅ Analysis report completed")
                except Exception as e:
                    print(f"❌ Analysis report failed: {e}")
                    import traceback
                    print(f"Traceback:\n{traceback.format_exc()}")

                # 8. Head specialization analysis (with multiple related texts)
                try:
                    print("🎯 Performing head specialization analysis...")
                    related_texts = [
                        full_input_text,
                        f"Patient Note: {patient_note}",
                        f"Question: {question}",
                        f"Medical calculation for {row.get('Calculator Name', 'Unknown Calculator')}"
                    ]

                    head_analysis = attention_visualizer.analyze_head_specialization(
                        texts=related_texts,
                        layer=target_layer
                    )

                    attention_visualizer.plot_head_specialization(
                        head_analysis,
                        save_path=os.path.join(entry_dir, "head_specialization.png")
                    )
                    attention_data["head_specialization"] = "head_specialization.png"
                    print("✅ Head specialization analysis completed")
                except Exception as e:
                    print(f"❌ Head specialization analysis failed: {e}")
                    import traceback
                    print(f"Traceback:\n{traceback.format_exc()}")

                # 9. Save attention analysis summary as JSON
                try:
                    attention_summary = {
                        "calculator_id": calculator_id,
                        "note_id": note_id,
                        "calculator_name": row.get("Calculator Name", "Unknown"),
                        "category": row.get("Category", "Unknown"),
                        "model_name": model_name,
                        "prompt_style": prompt_style,
                        "input_length": len(full_input_text),
                        "patient_note_length": len(patient_note),
                        "question_length": len(question),
                        "attention_files": attention_data,
                        "timestamp": pd.Timestamp.now().isoformat()
                    }

                    with open(os.path.join(entry_dir, "attention_summary.json"), "w") as f:
                        json.dump(attention_summary, f, indent=2)

                    print(f"Attention analysis completed for Calculator {calculator_id}, Note {note_id}")
                    print(f"Files saved to: {entry_dir}")

                except Exception as e:
                    print(f"Attention summary save failed: {e}")

            except Exception as e:
                print(f"Attention analysis failed for Calculator {calculator_id}, Note {note_id}: {e}")

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
                "Result": status
            }

            # Add attention analysis info to outputs if available
            if enable_attention and attention_data:
                outputs["Attention_Analysis_Directory"] = f"attention_analysis/calc_{calculator_id}_note_{note_id}"
                outputs["Attention_Files_Generated"] = list(attention_data.keys())

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
                "Result": "Incorrect"
            }
            print(f"error in {calculator_id} {note_id}: "  + str(e))

            if prompt_style == "direct_answer":
                outputs["LLM Explanation"] = "N/A"

        print(outputs)

        with open(f"outputs/{output_path}", "a") as f:
            f.write(json.dumps(outputs) + "\n")

    compute_overall_accuracy(output_path, model_name, prompt_style)

    # Print attention analysis summary if enabled
    if enable_attention:
        print(f"\nProcessing completed with attention analysis!")
        print(f"Attention visualizations saved to: {attention_output_dir}")
        print("Generated files for each entry:")
        print("   - basic_attention.html (interactive visualization)")
        print("   - attention_heatmap.png (static heatmap)")
        print("   - layer_comparison.png (multi-layer comparison)")
        print("   - attention_data.json (raw attention weights)")
        print("   - attention_report.md (comprehensive analysis)")
        print("   - head_specialization.png (head analysis)")
        print("   - attention_summary.json (metadata and file list)")








