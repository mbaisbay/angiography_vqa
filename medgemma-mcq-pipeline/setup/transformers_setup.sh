#!/usr/bin/env bash
# =============================================================================
# Transformers + Accelerate Setup for MedGemma 27B
# =============================================================================
# For direct model loading, prompt-tuning, and fine-tuning with PEFT/TRL.
# Hardware target: 6x NVIDIA RTX 4090 (24GB VRAM each, 144GB total)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="${PROJECT_DIR}/venv"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# =============================================================================
# Step 1: Create virtual environment
# =============================================================================
info "Creating Python virtual environment at ${VENV_DIR}..."

if [ -d "$VENV_DIR" ]; then
    warn "Virtual environment already exists. Skipping creation."
else
    python3 -m venv "$VENV_DIR"
    info "Virtual environment created."
fi

source "${VENV_DIR}/bin/activate"
info "Activated virtual environment: $(which python)"

# =============================================================================
# Step 2: Install dependencies
# =============================================================================
info "Upgrading pip..."
pip install --upgrade pip

info "Installing PyTorch with CUDA 12.1 support..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

info "Installing Transformers ecosystem and training libraries..."
pip install \
    "transformers>=4.50.0" \
    accelerate \
    bitsandbytes \
    peft \
    trl \
    datasets \
    Pillow \
    requests \
    tqdm \
    pyyaml \
    scipy

info "All dependencies installed."

# =============================================================================
# Step 3: HuggingFace authentication reminder
# =============================================================================
echo ""
warn "============================================================"
warn "IMPORTANT: MedGemma is a gated model on HuggingFace."
warn "You must accept the license and authenticate before use."
warn ""
warn "  1. Visit https://huggingface.co/google/medgemma-27b-it"
warn "     and accept the model license agreement."
warn ""
warn "  2. Run:  huggingface-cli login"
warn "     and paste your HuggingFace access token."
warn "============================================================"
echo ""

if huggingface-cli whoami &>/dev/null; then
    info "HuggingFace CLI is authenticated as: $(huggingface-cli whoami 2>/dev/null | head -1)"
else
    warn "Not currently logged in to HuggingFace. Run 'huggingface-cli login' before loading the model."
fi

# =============================================================================
# Step 4: GPU verification
# =============================================================================
info "Running GPU verification..."
echo ""

python3 <<'PYEOF'
import torch
import sys

print("=" * 50)
print("GPU VERIFICATION")
print("=" * 50)

if not torch.cuda.is_available():
    print("ERROR: CUDA is not available!")
    print("  torch version:", torch.__version__)
    print("  CUDA built with:", torch.version.cuda)
    sys.exit(1)

num_gpus = torch.cuda.device_count()
print(f"PyTorch version:  {torch.__version__}")
print(f"CUDA version:     {torch.version.cuda}")
print(f"cuDNN version:    {torch.backends.cudnn.version()}")
print(f"GPUs available:   {num_gpus}")
print()

total_vram = 0
for i in range(num_gpus):
    props = torch.cuda.get_device_properties(i)
    vram_gb = props.total_mem / (1024 ** 3)
    total_vram += vram_gb
    print(f"  GPU {i}: {props.name}")
    print(f"         VRAM: {vram_gb:.1f} GB")
    print(f"         Compute capability: {props.major}.{props.minor}")
    print()

print(f"Total VRAM: {total_vram:.1f} GB")
print()

# BF16 support check
if num_gpus > 0:
    if torch.cuda.get_device_capability(0)[0] >= 8:
        print("BF16 support: YES (compute capability >= 8.0)")
    else:
        print("BF16 support: NO (compute capability < 8.0, will need FP16)")

# Memory estimate for MedGemma 27B
print()
print("MedGemma 27B memory estimates:")
print(f"  BF16 (full):     ~54 GB across {num_gpus} GPUs = ~{54/num_gpus:.1f} GB/GPU")
print(f"  INT8 (quantized): ~27 GB across {num_gpus} GPUs = ~{27/num_gpus:.1f} GB/GPU")
print(f"  INT4 (quantized): ~14 GB across {num_gpus} GPUs = ~{14/num_gpus:.1f} GB/GPU")

if total_vram >= 54:
    print("\n  --> BF16 full precision FITS in your GPU memory")
elif total_vram >= 27:
    print("\n  --> INT8 quantization recommended (use load_in_8bit=True)")
else:
    print("\n  --> INT4 quantization required (use load_in_4bit=True)")

print("=" * 50)
PYEOF

echo ""

# Quick test: verify transformers can import and see the model config
info "Verifying transformers installation..."
python3 -c "
from transformers import AutoProcessor
print('Transformers import: OK')
print('AutoProcessor available: OK')
try:
    from transformers import AutoModelForImageTextToText
    print('AutoModelForImageTextToText: OK')
except ImportError:
    print('WARNING: AutoModelForImageTextToText not found, may need transformers>=4.50.0')
"

echo ""
info "Setup complete!"
info "  Virtual env: ${VENV_DIR}"
info "  Activate with: source ${VENV_DIR}/bin/activate"
echo ""
info "To load MedGemma 27B in Python:"
echo "  from transformers import AutoProcessor, AutoModelForImageTextToText"
echo "  import torch"
echo ""
echo "  processor = AutoProcessor.from_pretrained('google/medgemma-27b-it')"
echo "  model = AutoModelForImageTextToText.from_pretrained("
echo "      'google/medgemma-27b-it',"
echo "      torch_dtype=torch.bfloat16,"
echo "      device_map='auto',"
echo "  )"
