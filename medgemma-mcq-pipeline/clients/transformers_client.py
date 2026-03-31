"""
Transformers client for MedGemma MCQ generation.

Loads google/medgemma-27b-it directly using HuggingFace Transformers.
Uses AutoProcessor and AutoModelForImageTextToText with device_map="auto"
to spread across all available GPUs.
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

from tqdm import tqdm
from PIL import Image

from prompts.system_prompts import SYSTEM_PROMPT_MCQ_GENERATOR, get_prompt_builder

logger = logging.getLogger(__name__)


class TransformersMedGemmaClient:
    """Client for generating MCQs using HuggingFace Transformers directly."""

    def __init__(
        self,
        model_id: str = "google/medgemma-27b-it",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 1024,
        retry_attempts: int = 3,
        log_dir: str = "logs",
        **kwargs,
    ):
        self.model_id = model_id
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.retry_attempts = retry_attempts
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.failure_log = self.log_dir / "failed_generations.jsonl"

        # Lazy-load model and processor
        self._model = None
        self._processor = None

    def _load_model(self):
        """Load the model and processor on first use."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        logger.info("Loading %s with device_map='auto'...", self.model_id)

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        logger.info("Model loaded. Device map: %s", self._model.hf_device_map)

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
        self._load_model()

        if mcq_type is None:
            mcq_type = self._pick_mcq_type(metadata)

        user_prompt = self._build_prompt(metadata, mcq_type)
        image = Image.open(image_path).convert("RGB")

        last_error = None
        for attempt in range(1, self.retry_attempts + 1):
            try:
                raw_response = self._generate(user_prompt, image)
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
    ) -> list[dict]:
        """Generate MCQs for a batch of images sequentially with progress tracking.

        Note: Transformers client processes sequentially (model is loaded in-process,
        so threading would not help). For parallel processing, use the Ollama or
        vLLM backends.

        Args:
            image_paths: List of image file paths.
            metadata_list: Corresponding list of metadata dicts.
            mcq_types: Optional list of MCQ types (one per image).

        Returns:
            List of result dicts with 'mcq' and 'error' keys.
        """
        if mcq_types is None:
            mcq_types = [None] * len(image_paths)

        results = []
        for img, meta, mtype in tqdm(
            zip(image_paths, metadata_list, mcq_types),
            total=len(image_paths),
            desc="Generating MCQs (Transformers)",
        ):
            try:
                mcq = self.generate_mcq(img, meta, mtype)
                results.append({"mcq": mcq, "error": None})
            except Exception as e:
                results.append({"mcq": None, "error": str(e)})

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

    def _generate(self, user_prompt: str, image: Image.Image) -> str:
        """Run model inference and return generated text."""
        import torch

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT_MCQ_GENERATOR}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                max_new_tokens=self.max_tokens,
            )

        # Decode only the new tokens (skip the input prompt)
        generated_tokens = output[0][input_len:]
        return self._processor.decode(generated_tokens, skip_special_tokens=True)

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
            "backend": "transformers",
            "image_path": image_path,
            "mcq_type": mcq_type,
            "error": error,
        }
        with open(self.failure_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def health_check(self) -> bool:
        """Verify that the model can be loaded."""
        try:
            self._load_model()
            return True
        except Exception as e:
            logger.error("Transformers health check failed: %s", e)
            return False
