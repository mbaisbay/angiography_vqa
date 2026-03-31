"""
Metadata schema for the MedGemma MCQ generation pipeline.

Defines the AngiogramMetadata dataclass that bridges the 4 source datasets
(ARCADE, CHD Syntax, CardioSyntax, CoronaryDominance) into a unified schema
consumed by prompt templates and MCQ generation clients.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional
import json
from pathlib import Path


# SYNTAX segment number -> anatomical name mapping
SYNTAX_SEGMENT_NAMES: dict[str, str] = {
    "1": "Proximal RCA",
    "2": "Mid RCA",
    "3": "Distal RCA",
    "4": "Posterior descending artery (PDA)",
    "4a": "Right posterolateral branch",
    "5": "Left main coronary artery (LMCA)",
    "6": "Proximal LAD",
    "7": "Mid LAD",
    "8": "Distal LAD",
    "9": "First diagonal branch (D1)",
    "9a": "First diagonal branch a",
    "10": "Second diagonal branch (D2)",
    "10a": "Second diagonal branch a",
    "11": "Proximal circumflex (LCx)",
    "12": "Intermediate/obtuse marginal branch (OM1)",
    "12a": "First obtuse marginal branch a",
    "12b": "Second obtuse marginal branch",
    "13": "Distal circumflex (LCx)",
    "14": "Left posterolateral branch",
    "14a": "Left posterolateral branch a",
    "14b": "Left posterolateral branch b",
    "15": "Posterior descending artery (from LCx)",
    "16": "Ramus intermedius",
    "16a": "Ramus intermedius a",
    "16b": "Ramus intermedius b",
    "16c": "Ramus intermedius c",
}


@dataclass
class AngiogramMetadata:
    """Unified metadata schema for angiogram images across all 4 datasets."""

    image_path: str
    dataset: str  # "arcade" | "chd_syntax" | "cardiosyntax" | "coronary_dominance"
    patient_id: Optional[str] = None

    # From ARCADE
    stenosis_locations: Optional[list[dict]] = None  # [{segment: int, bbox: [x,y,w,h]}]
    artery_segments: Optional[list[int]] = None  # SYNTAX segment numbers 1-16c

    # From CHD Syntax
    syntax_score: Optional[float] = None
    risk_group: Optional[str] = None  # "low" | "medium" | "high"
    stenosis_severity: Optional[str] = None  # "mild" | "moderate" | "severe" | "total_occlusion"

    # From CHD Syntax + CardioSyntax
    modifiers: Optional[list[str]] = None  # ["calcification", "thrombus", "tortuosity", ...]
    dominance: Optional[str] = None  # "right" | "left" | "co-dominance"

    # View info
    view_angle: Optional[str] = None  # "RAO_cranial", "LAO_caudal", etc.
    primary_angle: Optional[float] = None  # degrees
    secondary_angle: Optional[float] = None  # degrees

    # YOLO inference results (added later)
    yolo_detections: Optional[list[dict]] = None  # [{class: str, confidence: float, bbox: [...]}]

    def available_mcq_types(self) -> list[str]:
        """Determine which MCQ types can be generated based on available metadata fields."""
        types = []

        if self.artery_segments or self.stenosis_locations:
            types.append("vessel_identification")

        if self.stenosis_severity:
            types.append("stenosis_severity")

        if self.dominance:
            types.append("coronary_dominance")

        if self.syntax_score is not None or self.risk_group:
            types.append("syntax_scoring")

        if self.modifiers:
            types.append("lesion_morphology")

        if self.view_angle:
            types.append("view_identification")

        # Clinical reasoning requires multiple fields
        non_none_count = sum(1 for v in [
            self.stenosis_severity, self.syntax_score, self.dominance,
            self.modifiers, self.artery_segments, self.view_angle,
        ] if v is not None)
        if non_none_count >= 3:
            types.append("clinical_reasoning")

        return types

    def to_prompt_context(self) -> str:
        """Render all available metadata as a text block for prompt grounding."""
        lines = [f"Dataset: {self.dataset}"]

        if self.patient_id:
            lines.append(f"Patient ID: {self.patient_id}")

        if self.artery_segments:
            segment_names = []
            for seg in self.artery_segments:
                seg_str = str(seg)
                name = SYNTAX_SEGMENT_NAMES.get(seg_str, f"Segment {seg_str}")
                segment_names.append(f"Segment {seg_str} ({name})")
            lines.append(f"Artery segments present: {', '.join(segment_names)}")

        if self.stenosis_locations:
            for i, loc in enumerate(self.stenosis_locations):
                seg = loc.get("segment", "unknown")
                seg_name = SYNTAX_SEGMENT_NAMES.get(str(seg), f"Segment {seg}")
                lines.append(f"Stenosis #{i+1}: {seg_name}")

        if self.stenosis_severity:
            lines.append(f"Stenosis severity: {self.stenosis_severity}")

        if self.syntax_score is not None:
            lines.append(f"SYNTAX score: {self.syntax_score}")

        if self.risk_group:
            lines.append(f"SYNTAX risk group: {self.risk_group}")

        if self.modifiers:
            lines.append(f"Lesion modifiers: {', '.join(self.modifiers)}")

        if self.dominance:
            lines.append(f"Coronary dominance: {self.dominance}")

        if self.view_angle:
            lines.append(f"Angiographic view: {self.view_angle}")
            if self.primary_angle is not None:
                lines.append(f"Primary angle: {self.primary_angle}°")
            if self.secondary_angle is not None:
                lines.append(f"Secondary angle: {self.secondary_angle}°")

        if self.yolo_detections:
            lines.append(f"YOLO detections ({len(self.yolo_detections)}):")
            for det in self.yolo_detections[:10]:  # Cap at 10 for prompt length
                lines.append(
                    f"  - {det.get('class', 'unknown')} "
                    f"(conf: {det.get('confidence', 0):.2f})"
                )

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> AngiogramMetadata:
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def load_metadata_file(metadata_path: str) -> list[AngiogramMetadata]:
    """Load metadata from a JSON or pickle file.

    Expects either:
    - A JSON file containing a list of dicts, each matching AngiogramMetadata fields
    - A pickle file containing the same structure

    Returns:
        List of AngiogramMetadata instances.
    """
    path = Path(metadata_path)

    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    elif path.suffix in (".pkl", ".pickle"):
        import pickle
        with open(path, "rb") as f:
            raw = pickle.load(f)
    else:
        raise ValueError(f"Unsupported metadata file format: {path.suffix}. Use .json or .pkl")

    if isinstance(raw, dict):
        # If it's a dict keyed by image name, convert to list
        entries = []
        for key, val in raw.items():
            if isinstance(val, dict):
                if "image_path" not in val:
                    val["image_path"] = key
                entries.append(val)
        raw = entries

    return [AngiogramMetadata.from_dict(entry) for entry in raw]
