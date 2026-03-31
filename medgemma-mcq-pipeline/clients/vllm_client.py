"""
vLLM client for MedGemma MCQ generation.

Connects to a vLLM server running the OpenAI-compatible API.
Uses the chat completions endpoint with vision (base64 image) support.
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


class VLLMMedGemmaClient:
    """Client for generating MCQs via vLLM's OpenAI-compatible API."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        model: str = "google/medgemma-27b-it",
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
        mime_type = self._get_mime_type(image_path)

        last_error = None
        for attempt in range(1, self.retry_attempts + 1):
            try:
                raw_response = self._call_vllm(user_prompt, b64_image, mime_type)
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
            mcq_types: Optional list of MCQ types (one per image).
            max_workers: Number of concurrent threads.

        Returns:
            List of result dicts with 'mcq' and 'error' keys.
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

            with tqdm(total=len(image_paths), desc="Generating MCQs (vLLM)") as pbar:
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

    def _get_mime_type(self, image_path: str) -> str:
        """Determine MIME type from file extension."""
        ext = Path(image_path).suffix.lower()
        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".webp": "image/webp",
        }
        return mime_map.get(ext, "image/jpeg")

    def _call_vllm(self, user_prompt: str, b64_image: str, mime_type: str) -> str:
        """Send a chat completion request to vLLM and return the response text."""
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT_MCQ_GENERATOR,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{b64_image}"
                            },
                        },
                        {
                            "type": "text",
                            "text": user_prompt,
                        },
                    ],
                },
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }

        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()

        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def _parse_model_output(self, raw_text: str) -> dict:
        """Parse JSON from model output with fallback regex extraction."""
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
            "backend": "vllm",
            "image_path": image_path,
            "mcq_type": mcq_type,
            "error": error,
        }
        with open(self.failure_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def health_check(self) -> bool:
        """Verify that the vLLM server is running and responding."""
        try:
            resp = requests.get(f"{self.base_url}/v1/models", timeout=10)
            resp.raise_for_status()
            models = resp.json().get("data", [])
            model_ids = [m.get("id", "") for m in models]
            if self.model in model_ids or any(self.model in m for m in model_ids):
                return True
            logger.warning("Model '%s' not found. Available: %s", self.model, model_ids)
            return False
        except Exception as e:
            logger.error("vLLM health check failed: %s", e)
            return False
