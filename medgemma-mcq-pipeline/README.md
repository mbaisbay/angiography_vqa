# MedGemma MCQ Generation Pipeline

Generate board-exam-quality multiple-choice questions (MCQs) for coronary angiography education using MedGemma 27B multimodal.

## Overview

This pipeline takes coronary angiogram images + structured metadata from 4 datasets and produces validated MCQs covering:

| MCQ Type | What It Tests |
|----------|--------------|
| Vessel Identification | Identify coronary artery segments (SYNTAX 1-16c) |
| Stenosis Severity | Assess degree of stenosis (mild/moderate/severe/total) |
| Coronary Dominance | Determine RCA vs LCx dominance pattern |
| SYNTAX Scoring | Risk stratification and lesion modifier knowledge |
| Lesion Morphology | Identify calcification, thrombus, tortuosity, etc. |
| View Identification | Recognize angiographic projections (RAO, LAO, etc.) |
| Clinical Reasoning | Integrate findings for management decisions |

**Key design principle**: MCQs are *metadata-grounded* — the correct answer comes from verified dataset labels, not the model's image interpretation. MedGemma crafts the clinical vignette, distractors, and explanation around the known ground truth.

## Hardware Requirements

| Setup | GPUs | Notes |
|-------|------|-------|
| Recommended | 6x RTX 4090 (144GB total) | Full BF16, YOLO on GPU 0 + MedGemma on GPUs 1-5 |
| Minimum (vLLM) | 2x RTX 4090 (48GB) | BF16 with tensor-parallel=2 |
| Minimum (Ollama) | 1x RTX 4090 (24GB) | GGUF quantized model fits in 24GB |
| Minimum (Transformers) | 2x RTX 4090 | BF16 with device_map="auto" |

## Three Backend Options

| Feature | Ollama | vLLM | Transformers |
|---------|--------|------|--------------|
| Setup complexity | Easiest | Medium | Medium |
| Quantization | GGUF (built-in) | BF16 / AWQ | BF16 / BnB 4-bit / 8-bit |
| Throughput | Low-medium | Highest | Low |
| Batching | Thread-based | Native | Sequential |
| Fine-tuning | No | No | Yes (PEFT/TRL) |
| GPU memory | ~20GB (GGUF Q4) | ~27GB/GPU (BF16, TP=2) | ~54GB total (BF16) |
| Best for | Quick start, single GPU | Production throughput | Fine-tuning, research |

## Quick Start

### Option 1: Ollama (Easiest)

```bash
cd medgemma-mcq-pipeline

# Install Ollama, pull model, configure multi-GPU, create custom model
bash setup/ollama_setup.sh

# Generate MCQs
python -m pipeline.generate_mcqs \
    --backend ollama \
    --dataset chd_syntax \
    --image-dir /data/CHD_Syntax/images/ \
    --metadata-file /data/CHD_Syntax/metadata.json \
    --output output/mcqs.jsonl \
    --mcq-types vessel_identification,stenosis_severity
```

### Option 2: vLLM (Fastest)

```bash
cd medgemma-mcq-pipeline

# Create venv, install deps, start server (requires HF login)
huggingface-cli login
bash setup/vllm_setup.sh

# Generate MCQs (in another terminal)
source venv/bin/activate
python -m pipeline.generate_mcqs \
    --backend vllm \
    --dataset chd_syntax \
    --image-dir /data/CHD_Syntax/images/ \
    --metadata-file /data/CHD_Syntax/metadata.json \
    --output output/mcqs.jsonl \
    --mcq-types vessel_identification,stenosis_severity
```

### Option 3: Transformers (For Fine-tuning)

```bash
cd medgemma-mcq-pipeline

# Create venv with PyTorch + PEFT + TRL
huggingface-cli login
bash setup/transformers_setup.sh

# Generate MCQs
source venv/bin/activate
python -m pipeline.generate_mcqs \
    --backend transformers \
    --dataset chd_syntax \
    --image-dir /data/CHD_Syntax/images/ \
    --metadata-file /data/CHD_Syntax/metadata.json \
    --output output/mcqs.jsonl \
    --mcq-types vessel_identification,stenosis_severity
```

## Preparing Metadata

Each dataset needs a metadata file (JSON or pickle) where each entry maps to an image and contains the relevant fields. The `AngiogramMetadata` schema in `pipeline/metadata_schema.py` defines all supported fields.

### ARCADE Dataset
```json
[
  {
    "image_path": "image_001.png",
    "dataset": "arcade",
    "artery_segments": [6, 7, 9],
    "stenosis_locations": [{"segment": 7, "bbox": [100, 200, 50, 30]}],
    "stenosis_severity": "severe",
    "view_angle": "RAO_cranial"
  }
]
```

### CHD Syntax Dataset
```json
[
  {
    "image_path": "patient_042.png",
    "dataset": "chd_syntax",
    "syntax_score": 28.5,
    "risk_group": "medium",
    "stenosis_severity": "severe",
    "modifiers": ["calcification", "bifurcation", "diffuse_disease"],
    "dominance": "right",
    "artery_segments": [1, 2, 6, 7, 11],
    "view_angle": "LAO_cranial"
  }
]
```

### CardioSyntax Dataset
```json
[
  {
    "image_path": "case_015.png",
    "dataset": "cardiosyntax",
    "syntax_score": 35.0,
    "risk_group": "high",
    "modifiers": ["total_occlusion", "calcification", "tortuosity"],
    "dominance": "right"
  }
]
```

### CoronaryDominance Dataset
```json
[
  {
    "image_path": "study_089.png",
    "dataset": "coronary_dominance",
    "dominance": "left",
    "view_angle": "LAO_cranial"
  }
]
```

## Running the Pipeline

### Full generation

```bash
python -m pipeline.generate_mcqs \
    --backend vllm \
    --dataset chd_syntax \
    --image-dir /data/CHD_Syntax/images/ \
    --metadata-file /data/CHD_Syntax/metadata.json \
    --output output/chd_syntax_mcqs.jsonl \
    --mcq-types vessel_identification,stenosis_severity,syntax_scoring,lesion_morphology,coronary_dominance,view_identification,clinical_reasoning \
    --num-per-image 1 \
    --temperature 0.7
```

### Validate generated MCQs

```bash
# Basic structural validation
python -m pipeline.validate_mcqs \
    --input output/chd_syntax_mcqs.jsonl \
    --output output/validation_report.json

# With self-review (sends MCQs back through MedGemma)
python -m pipeline.validate_mcqs \
    --input output/chd_syntax_mcqs.jsonl \
    --output output/validation_report.json \
    --self-review \
    --backend vllm
```

## GPU Allocation Strategy

For running YOLO inference and MedGemma simultaneously:

```bash
# Terminal 1: YOLO on GPU 0
CUDA_VISIBLE_DEVICES=0 python train.py --task syntax

# Terminal 2: MedGemma (vLLM) on GPUs 1-2
CUDA_VISIBLE_DEVICES=1,2 vllm serve google/medgemma-27b-it \
    --tensor-parallel-size 2 --max-model-len 16384 \
    --dtype bfloat16 --gpu-memory-utilization 0.90 \
    --host 0.0.0.0 --port 8000

# Terminal 3: MCQ generation (connects to vLLM on port 8000)
python -m pipeline.generate_mcqs --backend vllm ...
```

For maximum MedGemma throughput (all 6 GPUs):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 vllm serve google/medgemma-27b-it \
    --tensor-parallel-size 6 --max-model-len 16384 \
    --dtype bfloat16 --gpu-memory-utilization 0.90 \
    --host 0.0.0.0 --port 8000
```

## Configuration

All tunable parameters are in `configs/generation_config.yaml`:

- Model backend, temperature, top_p, max_tokens
- MCQ types to generate and number per image
- Validation thresholds (distractor count, length ratio)
- Per-dataset image directories and available MCQ types

## Project Structure

```
medgemma-mcq-pipeline/
├── setup/
│   ├── ollama_setup.sh          # Install & configure Ollama + MedGemma
│   ├── vllm_setup.sh            # vLLM server setup
│   └── transformers_setup.sh    # HF Transformers + PEFT setup
├── clients/
│   ├── ollama_client.py         # Ollama API client
│   ├── vllm_client.py           # vLLM OpenAI-compatible client
│   └── transformers_client.py   # Direct HF model loading client
├── prompts/
│   ├── system_prompts.py        # System prompt + 7 MCQ type templates
│   └── few_shot_examples.py     # Hand-crafted examples per MCQ type
├── pipeline/
│   ├── metadata_schema.py       # AngiogramMetadata dataclass
│   ├── generate_mcqs.py         # Main generation pipeline
│   └── validate_mcqs.py         # MCQ quality validation
├── configs/
│   └── generation_config.yaml   # All tunable parameters
├── logs/                        # Runtime failure logs
└── README.md
```
