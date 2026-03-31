"""
Main MCQ generation pipeline.

Reads angiogram images + metadata, generates MCQs using the selected backend,
validates output structure, and saves results to JSONL.

Usage:
    python -m pipeline.generate_mcqs \
        --backend vllm \
        --dataset chd_syntax \
        --image-dir /data/CHD_Syntax/images/ \
        --metadata-file /data/CHD_Syntax/metadata.json \
        --output output/chd_syntax_mcqs.jsonl \
        --mcq-types vessel_identification,stenosis_severity \
        --num-per-image 1
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from tqdm import tqdm

from pipeline.metadata_schema import AngiogramMetadata, load_metadata_file
from prompts.system_prompts import PROMPT_BUILDERS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Client factory
# =============================================================================

def create_client(backend: str, **kwargs):
    """Create the appropriate MedGemma client based on backend selection.

    Args:
        backend: One of "ollama", "vllm", "transformers".
        **kwargs: Passed to the client constructor.

    Returns:
        Client instance with generate_mcq / generate_mcq_batch methods.
    """
    if backend == "ollama":
        from clients.ollama_client import OllamaMedGemmaClient
        return OllamaMedGemmaClient(**kwargs)
    elif backend == "vllm":
        from clients.vllm_client import VLLMMedGemmaClient
        return VLLMMedGemmaClient(**kwargs)
    elif backend == "transformers":
        from clients.transformers_client import TransformersMedGemmaClient
        return TransformersMedGemmaClient(**kwargs)
    else:
        raise ValueError(f"Unknown backend: '{backend}'. Use ollama, vllm, or transformers.")


# =============================================================================
# Structural validation (quick pre-save check)
# =============================================================================

REQUIRED_MCQ_FIELDS = {"stem", "correct_answer", "distractors", "explanation"}


def validate_mcq_structure(mcq: dict) -> tuple[bool, list[str]]:
    """Run quick structural validation on a generated MCQ.

    Returns:
        (is_valid, list_of_issues)
    """
    issues = []

    # Check required fields
    for field in REQUIRED_MCQ_FIELDS:
        if field not in mcq:
            issues.append(f"Missing required field: {field}")

    if "distractors" in mcq:
        distractors = mcq["distractors"]
        if not isinstance(distractors, list):
            issues.append("'distractors' is not a list")
        elif len(distractors) != 3:
            issues.append(f"Expected 3 distractors, got {len(distractors)}")

        # Check for duplicates
        if isinstance(distractors, list):
            correct = mcq.get("correct_answer", "")
            all_options = [correct] + distractors
            lower_options = [o.strip().lower() for o in all_options if isinstance(o, str)]
            if len(set(lower_options)) != len(lower_options):
                issues.append("Duplicate options detected (correct answer appears in distractors or distractors repeat)")

    return (len(issues) == 0, issues)


# =============================================================================
# Main pipeline
# =============================================================================

def determine_mcq_types_for_image(
    metadata: AngiogramMetadata,
    requested_types: list[str],
) -> list[str]:
    """Determine which MCQ types can be generated for a given image.

    Takes the intersection of what the metadata supports and what the user requested.
    """
    available = metadata.available_mcq_types()
    return [t for t in requested_types if t in available]


def run_pipeline(args):
    """Execute the MCQ generation pipeline."""
    # Load metadata
    logger.info("Loading metadata from %s", args.metadata_file)
    metadata_list = load_metadata_file(args.metadata_file)
    logger.info("Loaded %d metadata entries", len(metadata_list))

    # Match images to metadata
    image_dir = Path(args.image_dir)
    matched = []
    for meta in metadata_list:
        img_path = image_dir / Path(meta.image_path).name
        if img_path.exists():
            meta.image_path = str(img_path)
            matched.append(meta)
        else:
            logger.warning("Image not found: %s", img_path)

    logger.info("Matched %d images with metadata", len(matched))
    if not matched:
        logger.error("No images matched. Check --image-dir and metadata file.")
        sys.exit(1)

    # Parse requested MCQ types
    requested_types = [t.strip() for t in args.mcq_types.split(",")]
    for t in requested_types:
        if t not in PROMPT_BUILDERS:
            logger.error("Unknown MCQ type: '%s'. Available: %s", t, list(PROMPT_BUILDERS.keys()))
            sys.exit(1)

    # Create client
    client_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "retry_attempts": args.retry_attempts,
        "log_dir": args.log_dir,
    }
    if args.backend == "ollama":
        client_kwargs["host"] = args.host
        client_kwargs["port"] = args.port
        client_kwargs["model"] = args.model or "medgemma-mcq"
    elif args.backend == "vllm":
        client_kwargs["host"] = args.host
        client_kwargs["port"] = args.port
        client_kwargs["model"] = args.model or "google/medgemma-27b-it"
    elif args.backend == "transformers":
        client_kwargs["model_id"] = args.model or "google/medgemma-27b-it"

    client = create_client(args.backend, **client_kwargs)

    # Health check
    if hasattr(client, "health_check"):
        logger.info("Running health check...")
        if not client.health_check():
            logger.warning("Health check failed — proceeding anyway, but expect errors.")

    # Prepare output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Statistics
    stats = {
        "total_attempted": 0,
        "total_generated": 0,
        "validation_passed": 0,
        "validation_failed": 0,
        "generation_failed": 0,
        "by_type": {},
    }

    # Generate MCQs
    logger.info("Starting MCQ generation with backend=%s", args.backend)
    start_time = time.time()

    with open(output_path, "w", encoding="utf-8") as out_f:
        for meta in tqdm(matched, desc="Processing images"):
            viable_types = determine_mcq_types_for_image(meta, requested_types)
            if not viable_types:
                logger.debug("No viable MCQ types for %s", meta.image_path)
                continue

            for mcq_type in viable_types:
                for _ in range(args.num_per_image):
                    stats["total_attempted"] += 1

                    try:
                        mcq = client.generate_mcq(
                            image_path=meta.image_path,
                            metadata=meta.to_dict() if hasattr(meta, "to_dict") else meta,
                            mcq_type=mcq_type,
                        )
                    except Exception as e:
                        stats["generation_failed"] += 1
                        stats.setdefault("by_type", {}).setdefault(mcq_type, {})
                        stats["by_type"].setdefault(mcq_type, {})["failed"] = (
                            stats["by_type"].get(mcq_type, {}).get("failed", 0) + 1
                        )
                        logger.debug("Generation failed for %s: %s", meta.image_path, e)
                        continue

                    stats["total_generated"] += 1

                    # Validate structure
                    is_valid, issues = validate_mcq_structure(mcq)
                    if is_valid:
                        stats["validation_passed"] += 1
                    else:
                        stats["validation_failed"] += 1
                        logger.debug("Validation issues for %s: %s", meta.image_path, issues)

                    # Update per-type stats
                    stats.setdefault("by_type", {}).setdefault(mcq_type, {})
                    type_stats = stats["by_type"][mcq_type]
                    type_stats["generated"] = type_stats.get("generated", 0) + 1
                    if is_valid:
                        type_stats["valid"] = type_stats.get("valid", 0) + 1

                    # Write to JSONL (include even if validation failed — can filter later)
                    record = {
                        "mcq": mcq,
                        "source": {
                            "image_path": meta.image_path,
                            "dataset": meta.dataset,
                            "mcq_type": mcq_type,
                        },
                        "generation_info": {
                            "backend": args.backend,
                            "model": args.model or "default",
                            "temperature": args.temperature,
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        },
                        "validation": {
                            "passed": is_valid,
                            "issues": issues,
                        },
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    elapsed = time.time() - start_time

    # Print summary
    print("\n" + "=" * 60)
    print("MCQ GENERATION SUMMARY")
    print("=" * 60)
    print(f"Backend:              {args.backend}")
    print(f"Dataset:              {args.dataset}")
    print(f"Images processed:     {len(matched)}")
    print(f"Total attempted:      {stats['total_attempted']}")
    print(f"Total generated:      {stats['total_generated']}")
    print(f"Validation passed:    {stats['validation_passed']}")
    print(f"Validation failed:    {stats['validation_failed']}")
    print(f"Generation failures:  {stats['generation_failed']}")
    print(f"Time elapsed:         {elapsed:.1f}s")
    if stats["total_attempted"] > 0:
        pass_rate = stats["validation_passed"] / stats["total_attempted"] * 100
        print(f"Overall pass rate:    {pass_rate:.1f}%")
    print(f"\nOutput saved to: {output_path}")

    if stats["by_type"]:
        print("\nPer-type breakdown:")
        for mtype, tstat in sorted(stats["by_type"].items()):
            gen = tstat.get("generated", 0)
            valid = tstat.get("valid", 0)
            failed = tstat.get("failed", 0)
            print(f"  {mtype:25s}  generated={gen}  valid={valid}  failed={failed}")

    print("=" * 60)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate MCQs from coronary angiogram images using MedGemma.",
    )
    parser.add_argument(
        "--backend",
        choices=["ollama", "vllm", "transformers"],
        required=True,
        help="Which MedGemma backend to use.",
    )
    parser.add_argument(
        "--dataset",
        choices=["arcade", "chd_syntax", "cardiosyntax", "coronary_dominance"],
        required=True,
        help="Source dataset name (for metadata in output).",
    )
    parser.add_argument(
        "--image-dir",
        required=True,
        help="Directory containing angiogram images.",
    )
    parser.add_argument(
        "--metadata-file",
        required=True,
        help="Path to metadata JSON or pickle file.",
    )
    parser.add_argument(
        "--output",
        default="output/mcqs.jsonl",
        help="Output JSONL file path (default: output/mcqs.jsonl).",
    )
    parser.add_argument(
        "--mcq-types",
        default="vessel_identification,stenosis_severity",
        help="Comma-separated list of MCQ types to generate.",
    )
    parser.add_argument(
        "--num-per-image",
        type=int,
        default=1,
        help="Number of MCQs to generate per image per type (default: 1).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name/ID override (default depends on backend).",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Server host for ollama/vllm backends (default: localhost).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Server port (default: 11434 for ollama, 8000 for vllm).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling (default: 0.9).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Max tokens to generate (default: 1024).",
    )
    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=3,
        help="Retry attempts per failed generation (default: 3).",
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Directory for failure logs (default: logs/).",
    )

    args = parser.parse_args()

    # Set default ports
    if args.port is None:
        args.port = 11434 if args.backend == "ollama" else 8000

    return args


if __name__ == "__main__":
    run_pipeline(parse_args())
