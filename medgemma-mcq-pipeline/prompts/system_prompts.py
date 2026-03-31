"""
System prompts and MCQ-type-specific user prompt templates for MedGemma.

This is the most critical file in the pipeline. All prompt engineering lives here.

Design principles:
1. Every prompt includes 2-3 few-shot examples for consistent JSON output.
2. Prompts are METADATA-GROUNDED: the ground-truth answer is provided in the prompt
   so the model creates clinical vignettes around known facts, not its own interpretation.
3. Each template function accepts a metadata dict and returns a formatted prompt string.
4. Missing metadata is handled gracefully — templates check for None fields.
"""

import json
from typing import Optional

from prompts.few_shot_examples import FEW_SHOT_EXAMPLES


# =============================================================================
# MASTER SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT_MCQ_GENERATOR = """\
You are an expert interventional cardiologist and medical educator with 20 years \
of experience creating board-exam-quality multiple-choice questions (MCQs) for \
cardiology fellowship training. You specialize in coronary angiography interpretation.

Your task is to generate a single high-quality MCQ based on the provided coronary \
angiogram image and the accompanying ground-truth metadata. The metadata contains \
the verified correct findings — you MUST use these as the basis for the correct \
answer. Do NOT rely on your own image interpretation for the diagnosis; instead, \
use the image to craft a realistic clinical vignette and ensure visual consistency.

## OUTPUT FORMAT

You MUST respond with a single valid JSON object and nothing else. No markdown, \
no code fences, no explanatory text before or after the JSON. The JSON must have \
exactly these fields:

{
  "stem": "The clinical vignette and question text",
  "correct_answer": "The correct answer option",
  "distractors": ["Distractor 1", "Distractor 2", "Distractor 3"],
  "explanation": "Detailed explanation of why the correct answer is right and why each distractor is wrong",
  "difficulty": "easy | medium | hard",
  "topic": "The MCQ type category",
  "bloom_level": "remembering | understanding | applying | analyzing | evaluating"
}

## RULES FOR HIGH-QUALITY MCQs

1. STEM: Write a clinical vignette (2-4 sentences) including patient demographics, \
   presentation, and relevant history. End with a clear, focused question.

2. CORRECT ANSWER: Must be directly supported by the provided metadata. \
   Use standard medical terminology with the SYNTAX segment number where applicable.

3. DISTRACTORS (exactly 3):
   - Must be plausible and from the SAME category as the correct answer \
     (all vessel names, all severity grades, all dominance types, etc.)
   - Should represent common clinical misconceptions or look-alike findings
   - For vessel identification: use anatomically adjacent or commonly confused segments
   - For severity grading: use neighboring severity grades
   - For dominance: use the other dominance types
   - Must NOT be obviously wrong or absurd
   - Must NOT be duplicates of each other or of the correct answer
   - Should be similar in length and specificity to the correct answer

4. EXPLANATION: Explain why the correct answer is right AND briefly address \
   why each distractor is incorrect. Reference anatomical landmarks and \
   clinical reasoning.

5. DIFFICULTY:
   - easy: Pure recall of anatomy or definitions
   - medium: Requires interpretation or application of knowledge
   - hard: Requires integration of multiple concepts or clinical decision-making

6. BLOOM LEVEL: Assign the appropriate Bloom's taxonomy level:
   - remembering: Recall facts
   - understanding: Explain concepts
   - applying: Use knowledge in new situations
   - analyzing: Break down complex information
   - evaluating: Make clinical judgments
"""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _format_examples(mcq_type: str, num_examples: int = 2) -> str:
    """Format few-shot examples into a prompt-ready string."""
    examples = FEW_SHOT_EXAMPLES.get(mcq_type, [])[:num_examples]
    if not examples:
        return ""

    parts = ["Here are examples of the expected output format:\n"]
    for i, ex in enumerate(examples, 1):
        parts.append(f"Example {i}:")
        parts.append(json.dumps(ex, indent=2))
        parts.append("")  # blank line
    return "\n".join(parts)


def _get_metadata_field(metadata: dict, key: str, default=None):
    """Safely get a metadata field, handling both dict and object access."""
    if isinstance(metadata, dict):
        return metadata.get(key, default)
    return getattr(metadata, key, default)


# =============================================================================
# MCQ TYPE-SPECIFIC USER PROMPT TEMPLATES
# =============================================================================

def build_vessel_identification_prompt(metadata: dict) -> str:
    """Build prompt for vessel identification MCQs.

    Uses: artery_type/artery_segments, view_angle
    Distractors: anatomically adjacent or commonly confused segments.
    """
    artery_type = _get_metadata_field(metadata, "artery_type")
    artery_segments = _get_metadata_field(metadata, "artery_segments")
    view_angle = _get_metadata_field(metadata, "view_angle")
    stenosis_locations = _get_metadata_field(metadata, "stenosis_locations")

    context_parts = [
        "## GROUND-TRUTH METADATA (use this for the correct answer)",
    ]

    if artery_type:
        context_parts.append(f"- Target artery/segment: {artery_type}")
    if artery_segments:
        context_parts.append(f"- Artery segments visible: {artery_segments}")
    if stenosis_locations:
        context_parts.append(f"- Stenosis locations: {stenosis_locations}")
    if view_angle:
        context_parts.append(f"- Angiographic view: {view_angle}")

    context = "\n".join(context_parts)
    examples = _format_examples("vessel_identification", 2)

    return f"""\
Generate a VESSEL IDENTIFICATION MCQ based on the provided coronary angiogram image.

{context}

## TASK
Create a question asking the trainee to identify a specific coronary artery segment \
visible in this angiogram. The correct answer MUST match the ground-truth metadata above.

## DISTRACTOR GUIDELINES
- Use anatomically adjacent segments as distractors (e.g., if the answer is mid-LAD \
  Segment 7, use proximal LAD Segment 6, first diagonal Segment 9, or distal LAD Segment 8)
- Include the SYNTAX segment number in parentheses for each option
- All options must be vessel names — do not mix in non-vessel answers

## SEGMENT REFERENCE (SYNTAX numbering)
- Segments 1-3: RCA (proximal, mid, distal)
- Segment 4/4a: PDA / right posterolateral (from RCA)
- Segment 5: Left main (LMCA)
- Segments 6-8: LAD (proximal, mid, distal)
- Segments 9-10a: Diagonal branches (D1, D1a, D2, D2a)
- Segments 11, 13: LCx (proximal, distal)
- Segments 12-12b: Obtuse marginal branches
- Segments 14-14b: Left posterolateral branches
- Segment 15: PDA (from LCx)
- Segments 16-16c: Ramus intermedius and branches

{examples}

Respond with ONLY the JSON object."""


def build_stenosis_severity_prompt(metadata: dict) -> str:
    """Build prompt for stenosis severity assessment MCQs.

    Uses: stenosis_severity, segment, view_angle
    Distractors: adjacent severity grades + "no significant stenosis".
    """
    stenosis_severity = _get_metadata_field(metadata, "stenosis_severity")
    segment = (_get_metadata_field(metadata, "segment")
               or _get_metadata_field(metadata, "artery_type"))
    view_angle = _get_metadata_field(metadata, "view_angle")

    context_parts = [
        "## GROUND-TRUTH METADATA (use this for the correct answer)",
    ]

    if stenosis_severity:
        context_parts.append(f"- Stenosis severity: {stenosis_severity}")
    if segment:
        context_parts.append(f"- Affected segment: {segment}")
    if view_angle:
        context_parts.append(f"- Angiographic view: {view_angle}")

    context = "\n".join(context_parts)
    examples = _format_examples("stenosis_severity", 2)

    return f"""\
Generate a STENOSIS SEVERITY MCQ based on the provided coronary angiogram image.

{context}

## TASK
Create a question asking the trainee to assess the degree of stenosis severity \
visible in this angiogram. The correct answer MUST match the ground-truth severity above.

## SEVERITY CATEGORIES (use these exact categories for ALL options)
- Mild stenosis (<50% diameter reduction)
- Moderate stenosis (50-69% diameter reduction)
- Severe stenosis (70-99% diameter reduction)
- Total occlusion (100% diameter reduction)
- No significant stenosis (optional distractor)

## DISTRACTOR GUIDELINES
- Use adjacent severity grades as primary distractors
- For example, if the correct answer is "severe," use "moderate" and "total occlusion"
- Include "no significant stenosis" as one distractor if appropriate
- All options must be severity grades — do not mix in other answer types

{examples}

Respond with ONLY the JSON object."""


def build_coronary_dominance_prompt(metadata: dict) -> str:
    """Build prompt for coronary dominance MCQs.

    Uses: dominance_type, artery_type
    Distractors: other dominance types + common misconceptions.
    """
    dominance_type = (_get_metadata_field(metadata, "dominance_type")
                      or _get_metadata_field(metadata, "dominance"))
    artery_type = _get_metadata_field(metadata, "artery_type")

    context_parts = [
        "## GROUND-TRUTH METADATA (use this for the correct answer)",
    ]

    if dominance_type:
        context_parts.append(f"- Coronary dominance: {dominance_type}")
    if artery_type:
        context_parts.append(f"- Artery providing PDA: {artery_type}")

    context = "\n".join(context_parts)
    examples = _format_examples("coronary_dominance", 2)

    return f"""\
Generate a CORONARY DOMINANCE MCQ based on the provided coronary angiogram image.

{context}

## TASK
Create a question about the coronary dominance pattern shown in this angiogram. \
This could ask about:
- Which artery supplies the posterior descending artery (PDA)
- The dominance classification (right/left/co-dominant)
- The clinical significance of the dominance pattern

The correct answer MUST match the ground-truth dominance type above.

## DOMINANCE DEFINITIONS
- Right dominant (~85%): PDA arises from the distal RCA
- Left dominant (~8%): PDA arises from the distal LCx
- Co-dominant (~7%): Both RCA and LCx contribute to posterior septal supply

## DISTRACTOR GUIDELINES
- Always include the other two dominance types as distractors
- The third distractor should be a common misconception (e.g., confusing circumflex \
  territory with RCA territory, or misidentifying the PDA origin)
- All options must relate to coronary dominance or PDA supply

{examples}

Respond with ONLY the JSON object."""


def build_syntax_scoring_prompt(metadata: dict) -> str:
    """Build prompt for SYNTAX score / risk stratification MCQs.

    Uses: syntax_score, risk_group, modifiers_list, num_diseased_segments
    """
    syntax_score = _get_metadata_field(metadata, "syntax_score")
    risk_group = _get_metadata_field(metadata, "risk_group")
    modifiers_list = (_get_metadata_field(metadata, "modifiers_list")
                      or _get_metadata_field(metadata, "modifiers"))
    num_diseased = _get_metadata_field(metadata, "num_diseased_segments")

    context_parts = [
        "## GROUND-TRUTH METADATA (use this for the correct answer)",
    ]

    if syntax_score is not None:
        context_parts.append(f"- SYNTAX score: {syntax_score}")
    if risk_group:
        context_parts.append(f"- Risk group: {risk_group}")
    if modifiers_list:
        context_parts.append(f"- Lesion modifiers present: {modifiers_list}")
    if num_diseased is not None:
        context_parts.append(f"- Number of diseased segments: {num_diseased}")

    context = "\n".join(context_parts)
    examples = _format_examples("syntax_scoring", 2)

    return f"""\
Generate a SYNTAX SCORING MCQ based on the provided coronary angiogram image.

{context}

## TASK
Create a question about SYNTAX score interpretation or lesion scoring. Topics include:
- Risk stratification: Low (≤22), Intermediate (23-32), High (>32)
- Lesion modifiers that add SYNTAX points:
  * Bifurcation (classified by Medina system: disease at proximal, distal, side branch)
  * Trifurcation involvement
  * Heavy calcification
  * Intracoronary thrombus
  * Vessel tortuosity (≥3 bends >90° proximal to lesion)
  * Total occlusion modifiers (duration, blunt stump, bridging collaterals, side branch, length >20mm)
  * Diffuse disease (lesion length >20mm)
  * Aorto-ostial location
- Treatment implications based on score category

The correct answer MUST be consistent with the ground-truth SYNTAX data above.

## DISTRACTOR GUIDELINES
- For risk group questions: use adjacent risk categories
- For modifier questions: use modifiers NOT present in this case
- For treatment questions: use inappropriate strategies for the given score
- All options must be relevant to SYNTAX scoring — no unrelated answer types

{examples}

Respond with ONLY the JSON object."""


def build_lesion_morphology_prompt(metadata: dict) -> str:
    """Build prompt for lesion morphology / characteristic MCQs.

    Uses: modifiers_list, segment, view_angle
    """
    modifiers_list = (_get_metadata_field(metadata, "modifiers_list")
                      or _get_metadata_field(metadata, "modifiers"))
    segment = (_get_metadata_field(metadata, "segment")
               or _get_metadata_field(metadata, "artery_type"))
    view_angle = _get_metadata_field(metadata, "view_angle")

    context_parts = [
        "## GROUND-TRUTH METADATA (use this for the correct answer)",
    ]

    if modifiers_list:
        context_parts.append(f"- Lesion characteristics present: {modifiers_list}")
    if segment:
        context_parts.append(f"- Affected segment: {segment}")
    if view_angle:
        context_parts.append(f"- Angiographic view: {view_angle}")

    context = "\n".join(context_parts)
    examples = _format_examples("lesion_morphology", 2)

    return f"""\
Generate a LESION MORPHOLOGY MCQ based on the provided coronary angiogram image.

{context}

## TASK
Create a question about specific lesion characteristics visible in this angiogram. \
Focus on one of the following morphological features:
- Calcification: Radiopaque densities along vessel wall
- Thrombus: Hazy, irregular intraluminal filling defect
- Tortuosity: ≥3 consecutive bends >90° proximal to the lesion
- Bifurcation involvement: Disease at a vessel branching point
- Diffuse disease: Lesion length >20mm with gradual tapering
- Aorto-ostial location: Lesion at the very origin from the aorta

The correct answer MUST identify one of the characteristics confirmed in the metadata.

## DISTRACTOR GUIDELINES
- Use other lesion characteristics NOT present in this case
- All options must be lesion morphology descriptors — same category
- Distractors should be characteristics that could plausibly be confused \
  with the actual finding on angiography
- Match the specificity level of the correct answer

{examples}

Respond with ONLY the JSON object."""


def build_view_identification_prompt(metadata: dict) -> str:
    """Build prompt for angiographic view/projection identification MCQs.

    Uses: view_angle, primary_angle, secondary_angle
    """
    view_angle = _get_metadata_field(metadata, "view_angle")
    primary_angle = _get_metadata_field(metadata, "primary_angle")
    secondary_angle = _get_metadata_field(metadata, "secondary_angle")

    context_parts = [
        "## GROUND-TRUTH METADATA (use this for the correct answer)",
    ]

    if view_angle:
        context_parts.append(f"- Angiographic projection: {view_angle}")
    if primary_angle is not None:
        context_parts.append(f"- Primary angle (degrees): {primary_angle}")
    if secondary_angle is not None:
        context_parts.append(f"- Secondary angle (degrees): {secondary_angle}")

    context = "\n".join(context_parts)
    examples = _format_examples("view_identification", 2)

    return f"""\
Generate a VIEW IDENTIFICATION MCQ based on the provided coronary angiogram image.

{context}

## TASK
Create a question asking the trainee to identify the angiographic projection used, \
or asking which vessels are best visualized in this particular view. The correct \
answer MUST match the ground-truth projection above.

## STANDARD PROJECTIONS
- RAO cranial: Best for LAD/diagonal separation
- RAO caudal: Best for LCx and obtuse marginal branches
- LAO cranial: Best for distal RCA, PDA, and posterolateral branches
- LAO caudal ("spider view"): Best for left main bifurcation
- AP cranial: Alternative for mid/distal LAD
- AP caudal: Alternative for proximal vessels
- Lateral: Profile view, useful for ostial lesions
- Straight RAO/LAO: Various applications

## DISTRACTOR GUIDELINES
- Use other standard projections as distractors
- Prefer projections that are commonly confused with the correct one
- All options must be angiographic projection names — same category
- If the question asks about best-visualized vessels, distractors should be \
  vessels typically seen in other projections

{examples}

Respond with ONLY the JSON object."""


def build_clinical_reasoning_prompt(metadata: dict) -> str:
    """Build prompt for higher-order clinical reasoning MCQs.

    Uses: ALL available metadata fields.
    Tests Bloom's taxonomy "applying" and "analyzing" levels.
    """
    # Gather all available metadata for the richest possible context
    context_parts = [
        "## GROUND-TRUTH METADATA (use ALL relevant fields for the clinical vignette)",
    ]

    stenosis_severity = _get_metadata_field(metadata, "stenosis_severity")
    if stenosis_severity:
        context_parts.append(f"- Stenosis severity: {stenosis_severity}")

    segment = (_get_metadata_field(metadata, "segment")
               or _get_metadata_field(metadata, "artery_type"))
    if segment:
        context_parts.append(f"- Affected segment: {segment}")

    artery_segments = _get_metadata_field(metadata, "artery_segments")
    if artery_segments:
        context_parts.append(f"- Artery segments involved: {artery_segments}")

    syntax_score = _get_metadata_field(metadata, "syntax_score")
    if syntax_score is not None:
        context_parts.append(f"- SYNTAX score: {syntax_score}")

    risk_group = _get_metadata_field(metadata, "risk_group")
    if risk_group:
        context_parts.append(f"- SYNTAX risk group: {risk_group}")

    modifiers_list = (_get_metadata_field(metadata, "modifiers_list")
                      or _get_metadata_field(metadata, "modifiers"))
    if modifiers_list:
        context_parts.append(f"- Lesion modifiers: {modifiers_list}")

    dominance_type = (_get_metadata_field(metadata, "dominance_type")
                      or _get_metadata_field(metadata, "dominance"))
    if dominance_type:
        context_parts.append(f"- Coronary dominance: {dominance_type}")

    view_angle = _get_metadata_field(metadata, "view_angle")
    if view_angle:
        context_parts.append(f"- Angiographic view: {view_angle}")

    num_diseased = _get_metadata_field(metadata, "num_diseased_segments")
    if num_diseased is not None:
        context_parts.append(f"- Number of diseased segments: {num_diseased}")

    context = "\n".join(context_parts)
    examples = _format_examples("clinical_reasoning", 2)

    return f"""\
Generate a CLINICAL REASONING MCQ based on the provided coronary angiogram image.

{context}

## TASK
Create a higher-order clinical reasoning question that integrates multiple findings \
from the angiogram and metadata. The question should test the trainee's ability to \
synthesize information and make clinical decisions. Question types include:

- "Based on this angiogram showing [findings], what is the most appropriate next step?"
- "Given the SYNTAX score and lesion complexity, which revascularization strategy is recommended?"
- "What is the clinical significance of [specific finding] in the context of [patient factors]?"
- "Which additional finding would change the management recommendation?"

## REQUIREMENTS
- Create a realistic clinical vignette with patient demographics, comorbidities, \
  and presentation (e.g., ACS vs stable angina)
- The question must require integration of AT LEAST two metadata elements
- The correct answer should reflect current guideline-based recommendations \
  (ESC/EACTS, ACC/AHA)
- Target Bloom's level: applying, analyzing, or evaluating

## DISTRACTOR GUIDELINES
- Distractors should be management options that would be appropriate in a \
  DIFFERENT clinical scenario
- Include a "partially correct" distractor that addresses only part of the clinical picture
- Include a common misconception or outdated practice as a distractor
- All options must be clinical management strategies or clinical judgments

{examples}

Respond with ONLY the JSON object."""


# =============================================================================
# PROMPT DISPATCHER
# =============================================================================

# Maps MCQ type string -> prompt builder function
PROMPT_BUILDERS: dict[str, callable] = {
    "vessel_identification": build_vessel_identification_prompt,
    "stenosis_severity": build_stenosis_severity_prompt,
    "coronary_dominance": build_coronary_dominance_prompt,
    "syntax_scoring": build_syntax_scoring_prompt,
    "lesion_morphology": build_lesion_morphology_prompt,
    "view_identification": build_view_identification_prompt,
    "clinical_reasoning": build_clinical_reasoning_prompt,
}


def get_prompt_builder(mcq_type: str):
    """Get the prompt builder function for a given MCQ type.

    Args:
        mcq_type: One of the supported MCQ type strings.

    Returns:
        A function that takes a metadata dict and returns a formatted prompt string.

    Raises:
        ValueError: If mcq_type is not recognized.
    """
    if mcq_type not in PROMPT_BUILDERS:
        raise ValueError(
            f"Unknown MCQ type: '{mcq_type}'. "
            f"Supported types: {list(PROMPT_BUILDERS.keys())}"
        )
    return PROMPT_BUILDERS[mcq_type]
