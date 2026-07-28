# LLM Fine-Tuning

A local fine-tuning project for adapting **DeepSeek-R1-Distill-Qwen** to produce more structured and reliable competitive-programming solutions.

The project uses a conversational prompt/completion dataset, QLoRA-style parameter-efficient fine-tuning, automated evaluation, and optional GGUF export for running the resulting model through Ollama.

> This is an experimental learning project. Fine-tuning a small local model does not guarantee reliable solutions to unseen Codeforces problems rated 2100. The immediate goal is to build a measurable training and evaluation pipeline.

## Project goals

* Fine-tune a local reasoning model on verified competitive-programming examples.
* Improve constraint analysis, algorithm selection, correctness explanations, and C++17 implementation.
* Measure improvement using held-out problems rather than training loss alone.
* Export the resulting model or adapter for local inference with Ollama.

## Current dataset

The repository includes:

```text
deepseek_cf_starter_train.jsonl
```

Dataset summary:

* 24 original training examples
* Approximate difficulty range: 800–2100
* Conversational `prompt` and `completion` format
* Complete problem statements
* Constraint and complexity analysis
* Key observations and algorithms
* Correctness explanations
* GNU C++17 solutions
* Metadata containing title, difficulty, tags, and source

All 24 JSON records parse successfully, and every included C++17 code block passes:

```bash
g++ -std=c++17 -fsyntax-only solution.cpp
```

Syntax compilation does not prove that every algorithm is correct. Add sample tests, randomized differential tests, and brute-force comparisons before treating an example as fully verified training data.

## Dataset format

Each line in the `.jsonl` file is one independent JSON object:

```json
{
  "prompt": [
    {
      "role": "system",
      "content": "You are an expert competitive programmer..."
    },
    {
      "role": "user",
      "content": "Full problem statement"
    }
  ],
  "completion": [
    {
      "role": "assistant",
      "content": "Constraint analysis, algorithm, proof, complexity, and C++17 code"
    }
  ],
  "metadata": {
    "title": "Problem title",
    "difficulty": 1800,
    "tags": ["graphs", "dynamic programming"],
    "source": "synthetic-original"
  }
}
```

The metadata is useful for analysis and curriculum construction, but it is not required by the trainer.

## Recommended model

Default checkpoint:

```text
deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
```

Smaller checkpoint for testing the pipeline on limited hardware:

```text
deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
```

The 1.5B checkpoint is useful for confirming that data loading, training, checkpointing, and evaluation work. The 7B checkpoint is the more relevant experiment for difficult competitive-programming tasks, but it requires substantially more memory.

## Recommended repository structure

```text
.
├── deepseek_cf_starter_train.jsonl
├── data/
│   ├── train.jsonl
│   ├── validation.jsonl
│   └── test.jsonl
├── training/
│   ├── train.py
│   └── config.yaml
├── evaluation/
│   ├── generate.py
│   ├── extract_code.py
│   ├── compile.py
│   ├── run_tests.py
│   └── metrics.py
├── export/
│   ├── merge_adapter.py
│   └── Modelfile
├── results/
├── requirements.txt
└── README.md
```

The starter dataset may remain at the repository root or be moved into `data/train.jsonl`.

## Environment setup

Python 3.10 or 3.11 is recommended.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install a CUDA-enabled PyTorch build that matches the installed NVIDIA driver, then install the training libraries:

```powershell
pip install --upgrade transformers datasets accelerate bitsandbytes peft trl sentencepiece
```

Verify that PyTorch can access the GPU:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Expected output on an NVIDIA system:

```text
True
NVIDIA GeForce RTX 3060 Ti
```

## Load the dataset

```python
from datasets import load_dataset

TRAIN_FILE = "deepseek_cf_starter_train.jsonl"

dataset = load_dataset(
    "json",
    data_files=TRAIN_FILE,
    split="train",
)

print(dataset)
print(dataset[0]["metadata"])
print(dataset[0]["prompt"])
```

## Create train and validation splits

For an initial pipeline test:

```python
splits = dataset.train_test_split(
    test_size=0.15,
    seed=42,
)

train_dataset = splits["train"]
validation_dataset = splits["test"]
```

For meaningful evaluation, create a separate held-out set of problems that never appears in training. A random split of only 24 examples is useful for debugging but is not a reliable benchmark.

## Suggested QLoRA configuration

A conservative starting configuration for the 7B checkpoint is:

```yaml
model: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
quantization: 4-bit NF4
double_quantization: true
sequence_length: 2048
micro_batch_size: 1
gradient_accumulation_steps: 16
epochs: 1
learning_rate: 0.0001
warmup_ratio: 0.05
weight_decay: 0.01
optimizer: paged_adamw_8bit
gradient_checkpointing: true
completion_only_loss: true

lora:
  rank: 16
  alpha: 32
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
```

On an 8 GB GPU, the 7B run may require shorter sequences, CPU offloading, or a smaller checkpoint. Begin with the 1.5B model when validating the code path.

## Training objective

The dataset uses conversational prompt/completion records. Training should apply loss to the assistant completion rather than teaching the model to reproduce the user prompt.

The desired completion structure is:

```text
Required complexity
Key observation
Algorithm
Correctness argument
Time and memory complexity
Complete GNU C++17 solution
```

Do not train directly on unverified model generations. A confidently written wrong solution is harmful supervision.

## Run training

Once `training/train.py` is implemented:

```powershell
python training/train.py
```

Store the output adapter separately from the original model:

```text
outputs/deepseek-cf-lora/
```

Useful files in the saved adapter directory normally include:

```text
adapter_config.json
adapter_model.safetensors
tokenizer_config.json
```

## Evaluation

Evaluate the base model and fine-tuned model on exactly the same held-out problems and generation settings.

Recommended metrics:

* Valid response-format rate
* C++ extraction rate
* Compilation rate
* Public-sample pass rate
* Generated-test pass rate
* `pass@1`
* `pass@8`
* Average response length
* Repetition or hallucination rate

Recommended failure labels:

```text
MISREAD_CONSTRAINT
WRONG_ALGORITHM
INVALID_COMPLEXITY
INCOMPLETE_PROOF
IMPLEMENTATION_BUG
MISSED_EDGE_CASE
OUTPUT_FORMAT_ERROR
HALLUCINATED_PROBLEM
```

A useful experiment report should include both the result and the exact generation configuration:

```json
{
  "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
  "checkpoint": "outputs/deepseek-cf-lora",
  "temperature": 0.2,
  "top_p": 0.9,
  "max_new_tokens": 4096,
  "samples_per_problem": 8,
  "compile_rate": 0.0,
  "sample_pass_rate": 0.0,
  "pass_at_1": 0.0,
  "pass_at_8": 0.0
}
```

## Stronger solution verification

For problems with manageable small constraints, build:

1. A random test generator.
2. A slow brute-force solution.
3. The candidate optimized solution.
4. A differential-testing script.

Example workflow:

```bash
for seed in {1..10000}; do
    ./generator "$seed" > input.txt
    ./brute < input.txt > expected.txt
    ./candidate < input.txt > actual.txt
    diff -w expected.txt actual.txt || break
done
```

Only examples that survive verification should be promoted into the high-confidence training set.

## Export to Ollama

Ollama is intended for local inference, while Hugging Face, TRL, PEFT, and bitsandbytes handle training.

Recommended export flow:

```text
Hugging Face base checkpoint
        ↓
QLoRA training
        ↓
LoRA adapter
        ↓
Merge adapter with base model
        ↓
Convert merged model to GGUF
        ↓
Import GGUF into Ollama
```

Example `Modelfile` after conversion:

```text
FROM ./deepseek-cf-q4_k_m.gguf

PARAMETER temperature 0.2
PARAMETER top_p 0.9
PARAMETER num_ctx 8192
PARAMETER num_predict 4096

SYSTEM """
You are an expert competitive programmer.
Solve only the problem supplied by the user.
Analyze constraints before selecting an algorithm.
Do not invent missing information or switch to another problem.
If a rigorous solution cannot be derived, say UNSOLVED rather than guessing.
"""
```

Create and run the Ollama model:

```powershell
ollama create deepseek-cf -f Modelfile
ollama run deepseek-cf
```

## Dataset growth plan

The current 24-example dataset is intended to validate the pipeline. It is too small to produce broad, reliable competitive-programming capability.

A stronger progression would be:

| Stage                 | Verified examples | Purpose                                             |
| --------------------- | ----------------- | --------------------------------------------------- |
| Pipeline test         | 24–100            | Confirm loading, training, saving, and inference    |
| Initial experiment    | 300–500           | Detect measurable behavioral changes                |
| Useful supervised run | 1,000–2,000       | Cover major algorithm families and difficulties     |
| Larger research run   | 5,000+            | Improve diversity, robustness, and curriculum depth |

Prioritize quality over raw volume. Include counterexamples, failed approaches, proof ideas, overflow checks, and implementation pitfalls.

## Important limitations

* A small supervised dataset can improve output style without improving genuine algorithm discovery.
* Training and testing on related problem variants can create misleading memorization results.
* Public Codeforces problems may already exist in a base model's pretraining data.
* Compilation success does not imply algorithmic correctness.
* A local 1.5B or 7B model is unlikely to solve unseen 2100-rated problems consistently.
* Longer reasoning is not automatically better; repetition and hallucination should be treated as evaluation failures.

## Roadmap

* [x] Create starter prompt/completion dataset
* [x] Validate JSONL structure
* [x] Syntax-check included C++17 solutions
* [ ] Add automated sample testing
* [ ] Add brute-force differential testing
* [ ] Implement QLoRA training script
* [ ] Implement baseline evaluation harness
* [ ] Add held-out Codeforces-style benchmark
* [ ] Compare 1.5B and 7B checkpoints
* [ ] Add rejection sampling from verified generations
* [ ] Merge and export the adapter to GGUF
* [ ] Run the fine-tuned model through Ollama

## Responsible experimentation

Keep raw data, cleaned data, generated data, and verified data in separate directories. Record seeds, model revisions, package versions, prompts, and generation parameters for every experiment so results can be reproduced.

Do not report the model as solving a difficulty tier based on one successful generation. Use a fixed benchmark and multiple samples per problem.

## License

The dataset entries are marked `synthetic-original`. Add a repository license before publishing or redistributing the project. Model weights are not included in this repository and remain subject to the license of the selected base checkpoint.
