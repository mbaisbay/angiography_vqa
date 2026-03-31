#!/usr/bin/env bash
# =============================================================================
# vLLM Setup for MedGemma 27B MCQ Generation
# =============================================================================
# Hardware target: 6x NVIDIA RTX 4090 (24GB VRAM each, 144GB total)
# Model: google/medgemma-27b-it (BF16 via HuggingFace)
#
# NOTE: tensor-parallel-size=2 uses ~27GB per GPU for BF16 weights (2 GPUs).
# This leaves 4 GPUs free for YOLO or other tasks.
# To dedicate all GPUs to MedGemma, increase to --tensor-parallel-size 4 or 6.
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

info "Installing vLLM and dependencies..."
pip install \
    vllm \
    "transformers>=4.50.0" \
    accelerate \
    Pillow \
    requests \
    tqdm \
    pyyaml

info "All dependencies installed."

# =============================================================================
# Step 3: HuggingFace authentication reminder
# =============================================================================
echo ""
warn "============================================================"
warn "IMPORTANT: MedGemma is a gated model on HuggingFace."
warn "You must accept the license and authenticate before serving."
warn ""
warn "  1. Visit https://huggingface.co/google/medgemma-27b-it"
warn "     and accept the model license agreement."
warn ""
warn "  2. Run:  huggingface-cli login"
warn "     and paste your HuggingFace access token."
warn "============================================================"
echo ""

# Check if already logged in
if huggingface-cli whoami &>/dev/null; then
    info "HuggingFace CLI is authenticated as: $(huggingface-cli whoami 2>/dev/null | head -1)"
else
    warn "Not currently logged in to HuggingFace. Run 'huggingface-cli login' before starting the server."
fi

# =============================================================================
# Step 4: Start vLLM server
# =============================================================================
info "Starting vLLM server..."
info "  Model:                google/medgemma-27b-it"
info "  Tensor parallel size: 2 (using 2 GPUs, ~27GB per GPU for BF16)"
info "  Max model length:     16384 tokens"
info "  GPU memory util:      90%"
info "  Host:                 0.0.0.0:8000"
echo ""

# NOTE: tensor-parallel-size=2 uses ~27GB across 2 GPUs for BF16 weights.
# This leaves GPUs 2-5 free for YOLO or other workloads.
# To use more GPUs for MedGemma (faster inference, lower per-GPU memory):
#   --tensor-parallel-size 4  → uses 4 GPUs (~14GB each)
#   --tensor-parallel-size 6  → uses all 6 GPUs (~10GB each)
#
# You can also specify which GPUs to use:
#   CUDA_VISIBLE_DEVICES=1,2 vllm serve ...    (GPUs 1-2, leave GPU 0 for YOLO)
#   CUDA_VISIBLE_DEVICES=2,3,4,5 vllm serve ... (GPUs 2-5, leave 0-1 for YOLO)

vllm serve google/medgemma-27b-it \
    --tensor-parallel-size 2 \
    --max-model-len 16384 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000 \
    &

VLLM_PID=$!
info "vLLM server starting (PID: $VLLM_PID)..."

# =============================================================================
# Step 5: Health check
# =============================================================================
info "Waiting for vLLM server to be ready (this may take several minutes for model loading)..."

for i in $(seq 1 180); do
    if curl -s http://localhost:8000/v1/models &>/dev/null; then
        info "vLLM server is ready!"
        break
    fi
    if [ "$i" -eq 180 ]; then
        error "vLLM server failed to start within 3 minutes."
        error "Check logs above for errors. Common issues:"
        error "  - Not authenticated with HuggingFace (run: huggingface-cli login)"
        error "  - Insufficient GPU memory"
        error "  - Model license not accepted"
        exit 1
    fi
    sleep 1
done

info "Health check — listing available models:"
curl -s http://localhost:8000/v1/models | python3 -m json.tool

echo ""
info "Quick test — sending a test request:"
echo ""
curl -s http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "google/medgemma-27b-it",
        "messages": [{"role": "user", "content": "Respond with: OK"}],
        "max_tokens": 10
    }' | python3 -m json.tool

echo ""
info "Setup complete!"
info "  Server:  http://localhost:8000 (OpenAI-compatible API)"
info "  Model:   google/medgemma-27b-it"
info "  GPUs:    2 (tensor-parallel-size=2)"
echo ""
info "Test endpoint:"
echo "  curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"
echo "    \"model\": \"google/medgemma-27b-it\","
echo "    \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}],"
echo "    \"max_tokens\": 50"
echo "  }'"
