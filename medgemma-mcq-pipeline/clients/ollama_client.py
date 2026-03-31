"""
Ollama client for MedGemma MCQ generation.

Connects to a local Ollama server running the alibayram/medgemma:27b model.
Uses the /api/chat endpoint with image support.
"""

import base64
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from prompts.system_prompts import SYSTEM_PROMPT_MCQ_GENERATOR, get_prompt_builder

logger = logging.getLogger(__name__)


class OllamaMedGemmaClient:
    """Client for generating MCQs via Ollama's local API."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 11434,
        model: str = "medgemma-mcq",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 1024,
        retry_attempts: int = 3,
        log_dir: str = "logs",
    ):
        self.base_url = f"http://{host}:{port}"
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.retry_attempts = retry_attempts
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.failure_log = self.log_dir / "failed_generations.jsonl"

    def generate_mcq(
        self,
        image_path: str,
        metadata: dict,
        mcq_type: Optional[str] = None,
    ) -> dict:
        """Generate a single MCQ from an image and metadata.

        Args:
            image_path: Path to the angiogram image.
            metadata: Dict with ground-truth metadata fields.
            mcq_type: MCQ type to generate. If None, picked from available types.

        Returns:
            Parsed MCQ dict with stem, correct_answer, distractors, etc.

        Raises:
            RuntimeError: If all retry attempts fail.
        """
        if mcq_type is None:
            mcq_type = self._pick_mcq_type(metadata)

        user_prompt = self._build_prompt(metadata, mcq_type)
        b64_image = self._encode_image(image_path)

        last_error = None
        for attempt in range(1, self.retry_attempts + 1):
            try:
                raw_response = self._call_ollama(user_prompt, b64_image)
                mcq = self._parse_model_output(raw_response)
                mcq["topic"] = mcq.get("topic", mcq_type)
                return mcq
            except Exception as e:
                last_error = e
                logger.warning(
                    "Attempt %d/%d failed for %s (%s): %s",
                    attempt, self.retry_attempts, image_path, mcq_type, e,
                )
                if attempt < self.retry_attempts:
                    time.sleep(1)

        # All attempts failed — log and raise
        self._log_failure(image_path, metadata, mcq_type, str(last_error))
        raise RuntimeError(
            f"Failed to generate MCQ after {self.retry_attempts} attempts: {last_error}"
        )

    def generate_mcq_batch(
        self,
        image_paths: list[str],
        metadata_list: list[dict],
        mcq_types: Optional[list[str]] = None,
        max_workers: int = 4,
    ) -> list[dict]:
        """Generate MCQs for a batch of images with progress tracking.

        Args:
            image_paths: List of image file paths.
            metadata_list: Corresponding list of metadata dicts.
            mcq_types: Optional list of MCQ types (one per image). If None, auto-selected.
            max_workers: Number of concurrent threads.

        Returns:
            List of result dicts. Each has 'mcq' (parsed dict or None) and 'error' (str or None).
        """
        if mcq_types is None:
            mcq_types = [None] * len(image_paths)

        results = [None] * len(image_paths)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {}
            for i, (img, meta, mtype) in enumerate(
                zip(image_paths, metadata_list, mcq_types)
            ):
                future = executor.submit(self.generate_mcq, img, meta, mtype)
                future_to_idx[future] = i

            with tqdm(total=len(image_paths), desc="Generating MCQs (Ollama)") as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        mcq = future.result()
                        results[idx] = {"mcq": mcq, "error": None}
                    except Exception as e:
                        results[idx] = {"mcq": None, "error": str(e)}
                    pbar.update(1)

        return results

    def _build_prompt(self, metadata: dict, mcq_type: str) -> str:
        """Build the user prompt for a given MCQ type and metadata."""
        builder = get_prompt_builder(mcq_type)
        return builder(metadata)

    def _pick_mcq_type(self, metadata: dict) -> str:
        """Pick an MCQ type based on available metadata fields."""
        # Import here to avoid circular dependency
        from pipeline.metadata_schema import AngiogramMetadata

        if isinstance(metadata, AngiogramMetadata):
            available = metadata.available_mcq_types()
        elif isinstance(metadata, dict):
            meta_obj = AngiogramMetadata.from_dict(metadata)
            available = meta_obj.available_mcq_types()
        else:
            available = ["vessel_identification"]

        return available[0] if available else "vessel_identification"

    def _encode_image(self, image_path: str) -> str:
        """Read and base64-encode an image file."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _call_ollama(self, user_prompt: str, b64_image: str) -> str:
        """Send a chat request to Ollama and return the raw response text."""
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT_MCQ_GENERATOR,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                    "images": [b64_image],
                },
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_predict": self.max_tokens,
            },
        }

        resp = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()

        data = resp.json()
        return data["message"]["content"]

    def _parse_model_output(self, raw_text: str) -> dict:
        """Parse JSON from model output with fallback regex extraction.

        Tries in order:
        1. Direct json.loads
        2. Extract from ```json ... ``` code fences
        3. Extract outermost { ... } block
        """
        text = raw_text.strip()

        # Attempt 1: direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Attempt 2: extract from code fences
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if fence_match:
            try:
                return json.loads(fence_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Attempt 3: extract outermost braces
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse JSON from model output: {text[:200]}...")

    def _log_failure(
        self, image_path: str, metadata: dict, mcq_type: str, error: str
    ):
        """Append a failure record to the JSONL log."""
        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "backend": "ollama",
            "image_path": image_path,
            "mcq_type": mcq_type,
            "error": error,
        }
        with open(self.failure_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def health_check(self) -> bool:
        """Verify that Ollama is running and the model is available."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=10)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            if self.model in models or any(self.model in m for m in models):
                return True
            logger.warning("Model '%s' not found. Available: %s", self.model, models)
            return False
        except Exception as e:
            logger.error("Ollama health check failed: %s", e)
            return False
