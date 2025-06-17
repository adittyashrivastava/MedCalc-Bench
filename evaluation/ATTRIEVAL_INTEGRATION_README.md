# ATTRIEVAL Integration in run.py

This document explains the integration of ATTRIEVAL (Attention-guided Retrieval for Long-Context Reasoning) into the medical calculation evaluation pipeline.

## 🔬 What is ATTRIEVAL?

ATTRIEVAL is a training-free algorithm that leverages attention weights from Chain-of-Thought (CoT) tokens to retrieve relevant facts from long contexts and improve reasoning performance. It addresses the key limitation that models often struggle to retrieve implicit facts during multi-hop reasoning tasks.

For more details, see: `attention_viz/ATTRIEVAL_IMPLEMENTATION.md`

## 🚀 Usage

### Basic Usage

Run evaluation with ATTRIEVAL analysis enabled:

```bash
python run.py \
    --model "mistralai/Mistral-7B-Instruct-v0.2" \
    --prompt "zero_shot" \
    --enable_attrieval \
    --debug_run \
    --num_examples 5
```

### Combined with Attention Analysis

Run both attention analysis and ATTRIEVAL:

```bash
python run.py \
    --model "meta-llama/Meta-Llama-3-8B-Instruct" \
    --prompt "zero_shot" \
    --enable_attention_analysis \
    --enable_attrieval \
    --debug_run \
    --num_examples 10
```

### Custom Output Directory

Specify a custom output directory:

```bash
python run.py \
    --model "mistralai/Mixtral-8x7B-Instruct-v0.1" \
    --prompt "one_shot" \
    --enable_attrieval \
    --output_dir "/path/to/custom/outputs" \
    --debug_run
```

### Production Run

Run on the full dataset (remove `--debug_run`):

```bash
python run.py \
    --model "meta-llama/Meta-Llama-3-70B-Instruct" \
    --prompt "zero_shot" \
    --enable_attrieval
```

## 📁 Output Structure

When ATTRIEVAL is enabled, results are saved in the following structure:

```
outputs/                                    # Default output directory
├── llm_results/                           # Standard LLM evaluation results
│   └── [model_name]_[prompt_style].jsonl
├── attention_results/                     # Attention analysis (if enabled)
│   └── calc_[ID]_note_[ID]_row_[N]/
└── attrieval_results/                     # ATTRIEVAL analysis results
    └── calc_[ID]_note_[ID]_row_[N]/       # Per-entry directory
        ├── attrieval_results.json         # Comprehensive results
        ├── attrieval_analysis_report.md   # Human-readable analysis
        ├── top_facts_summary.json         # Top retrieved facts
        └── attrieval_attention_data.npz   # Attention weights (compressed)
```

### Custom Output Directory

With `--output_dir /custom/path`, the structure becomes:

```
/custom/path/
├── llm_results/
├── attention_results/
└── attrieval_results/
```

## 📊 Generated Files

For each processed entry, ATTRIEVAL generates:

### 1. `attrieval_results.json`
Comprehensive results including:
- Retrieved facts with scores
- Configuration parameters
- Attention data metadata
- Algorithm execution details

### 2. `attrieval_analysis_report.md`
Human-readable analysis report with:
- Input summary (context, question, CoT response)
- Top retrieved facts with explanations
- Attention analysis insights
- Configuration details

### 3. `top_facts_summary.json`
Quick access to key results:
```json
{
  "calculator_id": "1",
  "note_id": "001",
  "row_number": 1,
  "question": "Calculate the patient's cardiovascular risk...",
  "top_retrieved_facts": [
    {
      "id": 0,
      "text": "Patient is a 65-year-old male with hypertension...",
      "score": 0.0234,
      "length": 15
    }
  ],
  "num_facts_retrieved": 5,
  "attrieval_config": {...},
  "timestamp": "2024-01-01T12:00:00"
}
```

### 4. `attrieval_attention_data.npz`
Compressed numpy arrays containing:
- Aggregated attention weights
- Retriever token indices
- Fact scores
- Other attention-related data

## 🔧 Configuration

ATTRIEVAL uses the following default configuration:

```python
AttrievelConfig(
    layer_fraction=0.25,      # Use last 25% of layers
    top_k=50,                 # Top 50 tokens per CoT token
    frequency_threshold=0.99, # Filter attention sinks
    max_facts=10             # Retrieve top 10 facts
)
```

## 🎯 Integration Details

### New Command Line Arguments

- `--enable_attrieval`: Enable ATTRIEVAL fact retrieval analysis
- Works alongside existing `--enable_attention_analysis`

### Output Integration

ATTRIEVAL results are added to the main LLM results JSON:

```json
{
  "Row Number": 1,
  "Calculator ID": "1",
  "Note ID": "001",
  "LLM Answer": "15",
  "LLM Explanation": "Based on age, hypertension...",
  "ATTRIEVAL_Analysis_Directory": "attrieval_results/calc_1_note_001_row_1",
  "ATTRIEVAL_Files_Generated": ["results_json", "analysis_report", "top_facts", "attention_data"],
  ...
}
```

## 🧪 Testing

Test the ATTRIEVAL integration:

```bash
cd evaluation/
python test_attrieval_integration.py
```

This script:
1. Creates a minimal test dataset
2. Runs ATTRIEVAL analysis on a sample medical case
3. Verifies that all expected output files are generated
4. Shows a preview of the results

## 🚫 Limitations

### Model Compatibility

ATTRIEVAL requires models that expose attention weights:
- ✅ **Supported**: Mistral, Llama, local transformer models
- ❌ **Not Supported**: OpenAI models (GPT-3.5, GPT-4)

When using OpenAI models, ATTRIEVAL is automatically disabled:
```
⚠️  Attention analysis and ATTRIEVAL are not supported for OpenAI models. Disabling both.
```

### Memory Requirements

ATTRIEVAL processes attention weights, which can be memory-intensive for:
- Large models (70B+ parameters)
- Long input sequences (>4000 tokens)
- Batch processing

For large-scale runs, consider:
- Using smaller models for testing
- Processing in smaller batches
- Monitoring memory usage

## 🔍 How It Works

1. **Input Processing**: ATTRIEVAL takes three inputs:
   - Context (patient note)
   - Question (medical calculation task)
   - CoT response (step-by-step reasoning)

2. **Attention Extraction**: Extracts attention weights from the model when processing the CoT response

3. **Fact Segmentation**: Breaks the patient note into discrete facts based on sentence boundaries

4. **Attention Aggregation**: Aggregates attention across the last 25% of model layers

5. **Fact Scoring**: Scores each fact based on how much attention the CoT tokens pay to it

6. **Retrieval**: Returns the top-scored facts that are most relevant to the reasoning process

## 📈 Expected Benefits

Based on the ATTRIEVAL paper:
- **Better fact retrieval**: Surfaces implicit relationships not explicitly mentioned in CoT
- **Improved reasoning**: Identifies which facts the model actually uses for reasoning
- **Training-free**: No model fine-tuning required
- **Context-length robust**: Performance maintained on long medical notes

## 🤔 Use Cases

### Research Applications
- **Attention analysis**: Understand which facts models focus on during medical reasoning
- **Error analysis**: Identify when models miss important clinical information
- **Model comparison**: Compare fact retrieval patterns across different models

### Clinical Applications
- **Decision support**: Highlight key facts the model uses for calculations
- **Verification**: Ensure important clinical data is being considered
- **Transparency**: Make model reasoning more interpretable for clinicians

## 🛠️ Troubleshooting

### Common Issues

**Issue**: "ATTRIEVAL analysis failed"
**Solution**: Check that you're using a compatible model (not OpenAI)

**Issue**: Out of memory errors
**Solution**: 
- Use `--debug_run --num_examples 1` for testing
- Try smaller models like `mistralai/Mistral-7B-Instruct-v0.2`

**Issue**: No ATTRIEVAL output directory created
**Solution**: Ensure `--enable_attrieval` flag is set and model initialization succeeded

### Debug Mode

Always test with debug mode first:
```bash
python run.py --model "gpt2" --prompt "zero_shot" --enable_attrieval --debug_run --num_examples 1
```

This runs on just one example to verify the integration works before processing the full dataset.

---

For more detailed information about the ATTRIEVAL algorithm itself, see the implementation guide at `attention_viz/ATTRIEVAL_IMPLEMENTATION.md`. 