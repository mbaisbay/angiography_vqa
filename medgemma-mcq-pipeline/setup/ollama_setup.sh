#!/usr/bin/env bash
# =============================================================================
# Ollama Setup for MedGemma 27B MCQ Generation
# =============================================================================
# Hardware target: 6x NVIDIA RTX 4090 (24GB VRAM each, 144GB total)
# Model: alibayram/medgemma:27b (GGUF multimodal)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# =============================================================================
# Step 1: Install Ollama if not present
# =============================================================================
info "Checking for Ollama installation..."

if command -v ollama &>/dev/null; then
    info "Ollama is already installed: $(ollama --version)"
else
    info "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    info "Ollama installed successfully."
fi

# =============================================================================
# Step 2: Configure environment for multi-GPU
# =============================================================================
info "Configuring Ollama environment variables..."

# Set environment variables for the current session
export OLLAMA_NUM_GPU=6
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q8_0

# Persist environment variables in systemd service override (if systemd is used)
if systemctl is-active --quiet ollama 2>/dev/null || systemctl list-unit-files | grep -q ollama; then
    info "Configuring systemd service overrides..."
    sudo mkdir -p /etc/systemd/system/ollama.service.d
    sudo tee /etc/systemd/system/ollama.service.d/override.conf > /dev/null <<EOF
[Service]
Environment="OLLAMA_NUM_GPU=6"
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
EOF
    sudo systemctl daemon-reload
    info "Systemd overrides applied."
else
    warn "Ollama systemd service not found. Environment variables set for this session only."
    warn "Add the following to your ~/.bashrc or /etc/environment for persistence:"
    echo "  export OLLAMA_NUM_GPU=6"
    echo "  export OLLAMA_FLASH_ATTENTION=1"
    echo "  export OLLAMA_KV_CACHE_TYPE=q8_0"
fi

# =============================================================================
# Step 3: Start Ollama server
# =============================================================================
info "Starting Ollama server..."

# Stop existing instance if running
if pgrep -x ollama &>/dev/null; then
    info "Stopping existing Ollama process..."
    ollama stop 2>/dev/null || true
    sleep 2
fi

# Start in background
OLLAMA_NUM_GPU=6 \
OLLAMA_FLASH_ATTENTION=1 \
OLLAMA_KV_CACHE_TYPE=q8_0 \
    ollama serve &>/dev/null &

OLLAMA_PID=$!
info "Ollama server started (PID: $OLLAMA_PID)"

# Wait for server to be ready
info "Waiting for Ollama server to be ready..."
for i in $(seq 1 30); do
    if curl -s http://localhost:11434/api/tags &>/dev/null; then
        info "Ollama server is ready."
        break
    fi
    if [ "$i" -eq 30 ]; then
        error "Ollama server failed to start within 30 seconds."
        exit 1
    fi
    sleep 1
done

# =============================================================================
# Step 4: Pull MedGemma model
# =============================================================================
info "Pulling alibayram/medgemma:27b model (this may take a while)..."
ollama pull alibayram/medgemma:27b

# =============================================================================
# Step 5: Create custom Modelfile with MCQ system prompt
# =============================================================================
info "Creating custom Modelfile for MCQ generation..."

MODELFILE_PATH="${PROJECT_DIR}/setup/Modelfile.medgemma-mcq"

cat > "$MODELFILE_PATH" <<'MODELFILE'
FROM alibayram/medgemma:27b

# Context window: 16K tokens for long prompts with few-shot examples
PARAMETER num_ctx 16384

# Sampling parameters for diverse but coherent MCQ generation
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1

SYSTEM """You are an expert interventional cardiologist and medical educator with 20 years of experience creating board-exam-quality multiple-choice questions (MCQs) for cardiology fellowship training. You specialize in coronary angiography interpretation.

Your task is to generate a single high-quality MCQ based on the provided coronary angiogram image and the accompanying ground-truth metadata. The metadata contains the verified correct findings — you MUST use these as the basis for the correct answer. Do NOT rely on your own image interpretation for the diagnosis; instead, use the image to craft a realistic clinical vignette and ensure visual consistency.

You MUST respond with a single valid JSON object and nothing else. No markdown, no code fences, no explanatory text. The JSON must have exactly these fields:

{
  "stem": "The clinical vignette and question text",
  "correct_answer": "The correct answer option",
  "distractors": ["Distractor 1", "Distractor 2", "Distractor 3"],
  "explanation": "Detailed explanation of why the correct answer is right and each distractor is wrong",
  "difficulty": "easy | medium | hard",
  "topic": "The MCQ type category",
  "bloom_level": "remembering | understanding | applying | analyzing | evaluating"
}

Rules:
- Exactly 3 plausible distractors from the same category as the correct answer
- Distractors represent common clinical misconceptions or look-alike findings
- No duplicate options; no obviously wrong answers
- Explanation must address why each distractor is incorrect
- Use SYNTAX segment numbering (1-16c) for vessel references"""
MODELFILE

info "Creating custom model 'medgemma-mcq' from Modelfile..."
ollama create medgemma-mcq -f "$MODELFILE_PATH"

# =============================================================================
# Step 6: Health check
# =============================================================================
info "Running health check..."

HEALTH_RESPONSE=$(curl -s http://localhost:11434/api/chat -d '{
  "model": "medgemma-mcq",
  "messages": [{"role": "user", "content": "Respond with only: OK"}],
  "stream": false,
  "options": {"num_predict": 10}
}')

if echo "$HEALTH_RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['message']['content'])" 2>/dev/null; then
    info "Health check PASSED — model is loaded and responding."
else
    error "Health check FAILED. Response:"
    echo "$HEALTH_RESPONSE"
    exit 1
fi

# =============================================================================
# Step 7: GPU utilization summary
# =============================================================================
info "GPU utilization summary:"
echo ""
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits | \
    awk -F',' '{printf "  GPU %s (%s): %s/%s MB (%.0f%% util)\n", $1, $2, $3, $4, $5}'
echo ""

# =============================================================================
# Done
# =============================================================================
info "Setup complete!"
info "  Model:   medgemma-mcq (based on alibayram/medgemma:27b)"
info "  Server:  http://localhost:11434"
info "  GPUs:    6 (OLLAMA_NUM_GPU=6)"
info "  KV Cache: q8_0"
info "  Flash Attention: enabled"
info "  Context: 16384 tokens"
echo ""
info "To test manually:"
echo "  curl http://localhost:11434/api/chat -d '{"
echo "    \"model\": \"medgemma-mcq\","
echo "    \"messages\": [{\"role\": \"user\", \"content\": \"Generate a test MCQ about coronary anatomy.\"}],"
echo "    \"stream\": false"
echo "  }'"
