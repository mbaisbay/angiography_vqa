"""
MCQ validation pipeline.

Reads generated MCQ JSONL files and runs quality checks on each MCQ.
Optionally sends MCQs back through MedGemma for self-review.

Usage:
    python -m pipeline.validate_mcqs \
        --input output/mcqs.jsonl \
        --output output/validation_report.json \
        --self-review \
        --backend vllm
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Validation checks
# =============================================================================

REQUIRED_FIELDS = {"stem", "correct_answer", "distractors", "explanation"}
OPTIONAL_FIELDS = {"difficulty", "topic", "bloom_level"}


def check_structural_validity(mcq: dict) -> list[str]:
    """Check that all required fields are present and correctly typed."""
    issues = []
    for field in REQUIRED_FIELDS:
        if field not in mcq:
            issues.append(f"missing_field:{field}")

    if "stem" in mcq and not isinstance(mcq["stem"], str):
        issues.append("stem_not_string")
    if "correct_answer" in mcq and not isinstance(mcq["correct_answer"], str):
        issues.append("correct_answer_not_string")
    if "explanation" in mcq and not isinstance(mcq["explanation"], str):
        issues.append("explanation_not_string")

    return issues


def check_distractor_count(mcq: dict, min_count: int = 3, max_count: int = 3) -> list[str]:
    """Check that there are exactly the expected number of distractors."""
    issues = []
    distractors = mcq.get("distractors", [])

    if not isinstance(distractors, list):
        issues.append("distractors_not_list")
        return issues

    if len(distractors) < min_count:
        issues.append(f"too_few_distractors:{len(distractors)}")
    if len(distractors) > max_count:
        issues.append(f"too_many_distractors:{len(distractors)}")

    return issues


def check_no_duplicates(mcq: dict) -> list[str]:
    """Check that no options are duplicated."""
    issues = []
    correct = mcq.get("correct_answer", "")
    distractors = mcq.get("distractors", [])

    if not isinstance(distractors, list):
        return issues

    all_options = [correct] + distractors
    normalized = [opt.strip().lower() for opt in all_options if isinstance(opt, str)]

    if len(set(normalized)) != len(normalized):
        issues.append("duplicate_options")

    # Check if correct answer appears in distractors
    correct_norm = correct.strip().lower() if isinstance(correct, str) else ""
    for d in distractors:
        if isinstance(d, str) and d.strip().lower() == correct_norm:
            issues.append("correct_answer_in_distractors")
            break

    return issues


def check_answer_length_consistency(mcq: dict, max_ratio: float = 3.0) -> list[str]:
    """Check that the longest option is not >max_ratio times the shortest.

    This catches the common MCQ flaw where the correct answer is much longer
    than distractors (giving away the answer).
    """
    issues = []
    correct = mcq.get("correct_answer", "")
    distractors = mcq.get("distractors", [])

    if not isinstance(distractors, list):
        return issues

    all_options = [correct] + [d for d in distractors if isinstance(d, str)]
    lengths = [len(opt.strip()) for opt in all_options if opt.strip()]

    if len(lengths) < 2:
        return issues

    shortest = min(lengths)
    longest = max(lengths)

    if shortest > 0 and longest / shortest > max_ratio:
        issues.append(f"length_ratio_exceeded:{longest/shortest:.1f}")

    return issues


def check_distractor_category_consistency(mcq: dict) -> list[str]:
    """Heuristic check: all options should be the same 'type'.

    Checks that options have similar structure (e.g., all contain segment numbers,
    all are severity grades, all are dominance types).
    """
    issues = []
    correct = mcq.get("correct_answer", "")
    distractors = mcq.get("distractors", [])

    if not isinstance(distractors, list):
        return issues

    all_options = [correct] + [d for d in distractors if isinstance(d, str)]

    # Heuristic: check if options contain "Segment" keyword consistently
    has_segment = [("segment" in opt.lower() or "(" in opt) for opt in all_options]
    if any(has_segment) and not all(has_segment):
        issues.append("inconsistent_segment_references")

    # Heuristic: check if options contain percentage patterns consistently
    import re
    has_percent = [bool(re.search(r"\d+%", opt)) for opt in all_options]
    if any(has_percent) and not all(has_percent):
        issues.append("inconsistent_percentage_format")

    return issues


# =============================================================================
# Self-review via MedGemma
# =============================================================================

SELF_REVIEW_PROMPT = """\
You are a medical education quality reviewer. Evaluate the following MCQ for \
coronary angiography education. Check for:
1. Medical accuracy of the correct answer and explanation
2. Plausibility and quality of distractors
3. Clarity and completeness of the clinical vignette
4. Whether the question tests the intended cognitive level

MCQ to review:
{mcq_json}

Respond with a JSON object:
{{
  "quality_score": <1-5 integer>,
  "medical_accuracy": true/false,
  "issues_found": ["list of specific issues, or empty if none"],
  "suggested_improvements": ["list of improvements, or empty if none"]
}}
"""


def self_review_mcq(client, mcq: dict, image_path: str = None) -> dict:
    """Send an MCQ back through MedGemma for self-review.

    Args:
        client: Any MedGemma client instance.
        mcq: The MCQ dict to review.
        image_path: Optional image path for context.

    Returns:
        Review result dict with quality_score, issues, etc.
    """
    mcq_json = json.dumps(mcq, indent=2)
    prompt = SELF_REVIEW_PROMPT.format(mcq_json=mcq_json)

    # We use the client's internal methods for a text-only generation
    # This is a simplified call — no image needed for review
    try:
        if hasattr(client, "_call_ollama"):
            raw = client._call_ollama(prompt, "")
        elif hasattr(client, "_call_vllm"):
            # Text-only request
            import requests as req
            payload = {
                "model": client.model,
                "messages": [
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 512,
            }
            resp = req.post(
                f"{client.base_url}/v1/chat/completions",
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]
        else:
            return {"quality_score": -1, "error": "Self-review not supported for this backend"}

        return client._parse_model_output(raw)
    except Exception as e:
        return {"quality_score": -1, "error": str(e)}


# =============================================================================
# Main validation pipeline
# =============================================================================

def validate_file(
    input_path: str,
    max_length_ratio: float = 3.0,
    min_distractors: int = 3,
    max_distractors: int = 3,
    do_self_review: bool = False,
    backend: str = None,
) -> dict:
    """Run all validation checks on a JSONL file of generated MCQs.

    Returns:
        Validation report dict with per-check and per-type statistics.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    # Load records
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("Invalid JSON at line %d: %s", line_num, e)

    logger.info("Loaded %d records from %s", len(records), input_path)

    # Optional: create client for self-review
    review_client = None
    if do_self_review and backend:
        from pipeline.generate_mcqs import create_client
        review_client = create_client(backend)

    # Run checks
    check_names = [
        "structural_validity",
        "distractor_count",
        "no_duplicates",
        "answer_length_consistency",
        "distractor_category_consistency",
    ]
    if do_self_review:
        check_names.append("self_review")

    report = {
        "total_records": len(records),
        "checks": {name: {"passed": 0, "failed": 0, "issues": []} for name in check_names},
        "by_type": defaultdict(lambda: {"total": 0, "all_passed": 0, "any_failed": 0}),
        "overall": {"fully_valid": 0, "has_issues": 0},
    }

    for record in records:
        mcq = record.get("mcq", record)  # Handle both wrapped and raw MCQ records
        mcq_type = (
            record.get("source", {}).get("mcq_type")
            or mcq.get("topic", "unknown")
        )
        image_path = record.get("source", {}).get("image_path")

        all_issues = []

        # Run each check
        issues = check_structural_validity(mcq)
        if issues:
            report["checks"]["structural_validity"]["failed"] += 1
            report["checks"]["structural_validity"]["issues"].extend(issues)
        else:
            report["checks"]["structural_validity"]["passed"] += 1
        all_issues.extend(issues)

        issues = check_distractor_count(mcq, min_distractors, max_distractors)
        if issues:
            report["checks"]["distractor_count"]["failed"] += 1
            report["checks"]["distractor_count"]["issues"].extend(issues)
        else:
            report["checks"]["distractor_count"]["passed"] += 1
        all_issues.extend(issues)

        issues = check_no_duplicates(mcq)
        if issues:
            report["checks"]["no_duplicates"]["failed"] += 1
            report["checks"]["no_duplicates"]["issues"].extend(issues)
        else:
            report["checks"]["no_duplicates"]["passed"] += 1
        all_issues.extend(issues)

        issues = check_answer_length_consistency(mcq, max_length_ratio)
        if issues:
            report["checks"]["answer_length_consistency"]["failed"] += 1
            report["checks"]["answer_length_consistency"]["issues"].extend(issues)
        else:
            report["checks"]["answer_length_consistency"]["passed"] += 1
        all_issues.extend(issues)

        issues = check_distractor_category_consistency(mcq)
        if issues:
            report["checks"]["distractor_category_consistency"]["failed"] += 1
            report["checks"]["distractor_category_consistency"]["issues"].extend(issues)
        else:
            report["checks"]["distractor_category_consistency"]["passed"] += 1
        all_issues.extend(issues)

        # Self-review
        if do_self_review and review_client:
            review = self_review_mcq(review_client, mcq, image_path)
            score = review.get("quality_score", -1)
            if isinstance(score, int) and score >= 3:
                report["checks"]["self_review"]["passed"] += 1
            else:
                report["checks"]["self_review"]["failed"] += 1
                report["checks"]["self_review"]["issues"].append(
                    review.get("issues_found", [])
                )

        # Aggregate
        type_stats = report["by_type"][mcq_type]
        type_stats["total"] += 1
        if not all_issues:
            report["overall"]["fully_valid"] += 1
            type_stats["all_passed"] += 1
        else:
            report["overall"]["has_issues"] += 1
            type_stats["any_failed"] += 1

    # Convert defaultdict for JSON serialization
    report["by_type"] = dict(report["by_type"])

    # Deduplicate issue lists (keep counts instead)
    for check_name in check_names:
        issue_list = report["checks"][check_name]["issues"]
        if issue_list and isinstance(issue_list[0], str):
            from collections import Counter
            counts = Counter(issue_list)
            report["checks"][check_name]["issue_counts"] = dict(counts)
        report["checks"][check_name].pop("issues", None)

    return report


def print_report(report: dict):
    """Print a human-readable validation report."""
    total = report["total_records"]
    print("\n" + "=" * 60)
    print("MCQ VALIDATION REPORT")
    print("=" * 60)
    print(f"Total MCQs evaluated: {total}")
    print(f"Fully valid:          {report['overall']['fully_valid']}")
    print(f"Has issues:           {report['overall']['has_issues']}")
    if total > 0:
        rate = report["overall"]["fully_valid"] / total * 100
        print(f"Pass rate:            {rate:.1f}%")

    print("\nPer-check results:")
    for check_name, check_data in report["checks"].items():
        passed = check_data["passed"]
        failed = check_data["failed"]
        print(f"  {check_name:35s}  passed={passed}  failed={failed}")
        if "issue_counts" in check_data:
            for issue, count in sorted(check_data["issue_counts"].items()):
                print(f"    - {issue}: {count}")

    if report["by_type"]:
        print("\nPer-type results:")
        for mtype, tdata in sorted(report["by_type"].items()):
            print(
                f"  {mtype:25s}  total={tdata['total']}  "
                f"valid={tdata['all_passed']}  issues={tdata['any_failed']}"
            )

    print("=" * 60)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate generated MCQs for structural and quality issues.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL file of generated MCQs.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for validation report JSON (default: prints to stdout).",
    )
    parser.add_argument(
        "--max-length-ratio",
        type=float,
        default=3.0,
        help="Max ratio of longest to shortest option length (default: 3.0).",
    )
    parser.add_argument(
        "--self-review",
        action="store_true",
        help="Send MCQs back through MedGemma for self-review.",
    )
    parser.add_argument(
        "--backend",
        choices=["ollama", "vllm", "transformers"],
        default=None,
        help="Backend to use for self-review (required if --self-review is set).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.self_review and not args.backend:
        logger.error("--backend is required when --self-review is enabled.")
        sys.exit(1)

    report = validate_file(
        input_path=args.input,
        max_length_ratio=args.max_length_ratio,
        do_self_review=args.self_review,
        backend=args.backend,
    )

    print_report(report)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info("Validation report saved to %s", output_path)


if __name__ == "__main__":
    main()
