"""
Hand-crafted few-shot examples for each MCQ type.

These serve as in-context examples within prompts to guide MedGemma toward
producing consistent, well-structured JSON output. All examples use realistic
coronary angiography scenarios with correct SYNTAX segment numbering.

SYNTAX segment reference:
  1-3: RCA (proximal, mid, distal)
  4-4a: PDA and right posterolateral (from RCA)
  5: LMCA
  6-8: LAD (proximal, mid, distal)
  9-10a: Diagonal branches (D1, D1a, D2, D2a)
  11-13: LCx (proximal, intermediate, distal)
  12-12b: Obtuse marginal branches
  14-14b: Left posterolateral branches
  15: PDA (from LCx)
  16-16c: Ramus intermedius and branches
"""

VESSEL_IDENTIFICATION_EXAMPLES = [
    {
        "stem": "A 62-year-old male presents with unstable angina. Coronary angiography is performed in the RAO cranial view. Which coronary artery segment is indicated by the arrow showing significant stenosis?",
        "correct_answer": "Left anterior descending artery, mid segment (Segment 7)",
        "distractors": [
            "First diagonal branch (Segment 9)",
            "Left circumflex artery, proximal segment (Segment 11)",
            "Ramus intermedius (Segment 16)"
        ],
        "explanation": "The stenosed vessel courses in the anterior interventricular groove, characteristic of the mid-LAD (Segment 7). The first diagonal (Segment 9) branches off the LAD but courses laterally over the anterolateral wall. The proximal LCx (Segment 11) runs in the atrioventricular groove posteriorly. The ramus intermedius (Segment 16) arises between the LAD and LCx at the trifurcation point and is not present in all patients.",
        "difficulty": "medium",
        "topic": "vessel_identification",
        "bloom_level": "understanding"
    },
    {
        "stem": "During diagnostic catheterization for a 55-year-old woman with exertional dyspnea, the LAO caudal projection reveals the vessel highlighted below. Which coronary artery segment is this?",
        "correct_answer": "Proximal left circumflex artery (Segment 11)",
        "distractors": [
            "Left main coronary artery (Segment 5)",
            "First obtuse marginal branch (Segment 12)",
            "Proximal left anterior descending artery (Segment 6)"
        ],
        "explanation": "In the LAO caudal view, the LCx is well visualized as it courses in the left atrioventricular groove. The proximal LCx (Segment 11) extends from the LMCA bifurcation to the origin of the first obtuse marginal branch. The LMCA (Segment 5) is shorter and proximal to the bifurcation. The OM1 (Segment 12) branches off laterally. The proximal LAD (Segment 6) courses anteriorly and is foreshortened in this view.",
        "difficulty": "medium",
        "topic": "vessel_identification",
        "bloom_level": "understanding"
    },
    {
        "stem": "A 70-year-old male with a history of prior CABG undergoes angiography. The RAO projection shows a dominant vessel coursing along the inferior cardiac surface. Which segment is most likely visualized?",
        "correct_answer": "Proximal right coronary artery (Segment 1)",
        "distractors": [
            "Distal right coronary artery (Segment 3)",
            "Posterior descending artery (Segment 4)",
            "Mid right coronary artery (Segment 2)"
        ],
        "explanation": "The proximal RCA (Segment 1) originates from the right coronary cusp and courses in the right atrioventricular groove. In the RAO view, this initial segment is seen clearly before the vessel curves at the acute margin of the heart. The mid RCA (Segment 2) follows along the right AV groove toward the crux. The distal RCA (Segment 3) is at the crux, and the PDA (Segment 4) courses in the posterior interventricular groove.",
        "difficulty": "easy",
        "topic": "vessel_identification",
        "bloom_level": "remembering"
    },
]

STENOSIS_SEVERITY_EXAMPLES = [
    {
        "stem": "A 58-year-old male with NSTEMI undergoes urgent coronary angiography. The mid-LAD (Segment 7) shows a focal narrowing as seen in this angiographic image. What is the estimated severity of this stenosis?",
        "correct_answer": "Severe stenosis (70-99% diameter reduction)",
        "distractors": [
            "Moderate stenosis (50-69% diameter reduction)",
            "Mild stenosis (<50% diameter reduction)",
            "Total occlusion (100% diameter reduction)"
        ],
        "explanation": "The angiographic image demonstrates a high-grade focal narrowing of the mid-LAD with preserved distal flow, consistent with severe stenosis (70-99%). Moderate stenosis (50-69%) would show a less pronounced but still hemodynamically significant narrowing. Mild stenosis (<50%) would show a subtle irregularity. Total occlusion would show complete absence of distal contrast filling, though collaterals may reconstitute the distal vessel.",
        "difficulty": "medium",
        "topic": "stenosis_severity",
        "bloom_level": "analyzing"
    },
    {
        "stem": "Coronary angiography in a 65-year-old woman with stable angina shows a narrowing in the proximal RCA (Segment 1). TIMI 3 flow is maintained distally. Based on the image, what is the degree of stenosis?",
        "correct_answer": "Moderate stenosis (50-69% diameter reduction)",
        "distractors": [
            "Severe stenosis (70-99% diameter reduction)",
            "Mild stenosis (<50% diameter reduction)",
            "No significant stenosis"
        ],
        "explanation": "The lesion shows an intermediate narrowing with well-maintained antegrade flow (TIMI 3), consistent with moderate stenosis of 50-69%. The vessel lumen is visibly reduced but more than half of the reference diameter is preserved. Severe stenosis would show a more critical narrowing with potential flow limitation. Mild stenosis would show only a subtle luminal irregularity with minimal hemodynamic significance.",
        "difficulty": "medium",
        "topic": "stenosis_severity",
        "bloom_level": "analyzing"
    },
    {
        "stem": "A 72-year-old diabetic male presents with acute STEMI. Angiography shows absence of antegrade contrast filling beyond the mid-segment of the RCA with a meniscus sign at the point of occlusion. What is the stenosis classification?",
        "correct_answer": "Total occlusion (100% diameter reduction)",
        "distractors": [
            "Severe stenosis (70-99% diameter reduction)",
            "Moderate stenosis (50-69% diameter reduction)",
            "Subtotal occlusion (95-99% diameter reduction)"
        ],
        "explanation": "The absence of antegrade contrast filling beyond the occlusion site with a meniscus sign (concave filling defect at the proximal cap) is the hallmark of a total occlusion (100%). In severe stenosis, there would still be a thread of antegrade contrast, even if TIMI flow is reduced. Subtotal occlusion shows a trickle of flow (TIMI 1). The clinical presentation of acute STEMI supports an acute thrombotic total occlusion.",
        "difficulty": "easy",
        "topic": "stenosis_severity",
        "bloom_level": "understanding"
    },
]

CORONARY_DOMINANCE_EXAMPLES = [
    {
        "stem": "During angiography of a 60-year-old patient, the posterior descending artery (PDA) is seen arising from the distal right coronary artery. What is the coronary dominance pattern in this patient?",
        "correct_answer": "Right dominant circulation",
        "distractors": [
            "Left dominant circulation",
            "Co-dominant (balanced) circulation",
            "Cannot be determined from a single projection"
        ],
        "explanation": "Coronary dominance is defined by which coronary artery gives rise to the posterior descending artery (PDA) that supplies the inferior interventricular septum. In right dominance (~85% of patients), the PDA arises from the distal RCA. In left dominance (~8%), the PDA arises from the distal LCx. In co-dominance (~7%), both the RCA and LCx contribute branches to the posterior septum. The PDA arising from the RCA confirms right dominant circulation.",
        "difficulty": "easy",
        "topic": "coronary_dominance",
        "bloom_level": "remembering"
    },
    {
        "stem": "Coronary angiography reveals that the left circumflex artery continues as a large vessel past the crux of the heart, giving off the posterior descending artery and posterolateral branches. What is the dominance type?",
        "correct_answer": "Left dominant circulation",
        "distractors": [
            "Right dominant circulation",
            "Co-dominant (balanced) circulation",
            "Hyperdominant right coronary artery"
        ],
        "explanation": "When the LCx extends beyond the crux of the heart and gives rise to the PDA and posterolateral branches, this defines a left dominant circulation pattern (~8% of the population). In these patients, the RCA is typically small and terminates before reaching the crux. Left dominance has clinical significance: the LAD and LCx territories supplied by the left coronary system are larger, making left main disease particularly high-risk in these patients.",
        "difficulty": "medium",
        "topic": "coronary_dominance",
        "bloom_level": "understanding"
    },
    {
        "stem": "Selective RCA injection shows the RCA giving off a small PDA that supplies only the proximal portion of the posterior interventricular groove. LCx injection shows a posterolateral branch that supplies the remaining posterior septum. What dominance pattern does this represent?",
        "correct_answer": "Co-dominant (balanced) circulation",
        "distractors": [
            "Right dominant circulation",
            "Left dominant circulation",
            "Variant anatomy with dual PDA"
        ],
        "explanation": "Co-dominance occurs when both the RCA and LCx share the blood supply to the inferior septum and posterior left ventricle. In this pattern, the RCA gives a small PDA while the LCx provides posterolateral branches or a secondary PDA-like vessel. This occurs in approximately 7% of the population. This is distinguished from right dominance (where the RCA PDA supplies the entire posterior septum) and left dominance (where the LCx provides the complete PDA).",
        "difficulty": "hard",
        "topic": "coronary_dominance",
        "bloom_level": "analyzing"
    },
]

SYNTAX_SCORING_EXAMPLES = [
    {
        "stem": "A 68-year-old patient's angiogram reveals three-vessel disease: proximal LAD 80% stenosis with heavy calcification, mid-RCA total occlusion (>3 months), and proximal LCx 70% stenosis at a bifurcation point (Medina 1,1,1). The calculated SYNTAX score is 28. What risk category does this patient fall into?",
        "correct_answer": "Intermediate risk (SYNTAX score 23-32)",
        "distractors": [
            "Low risk (SYNTAX score ≤22)",
            "High risk (SYNTAX score >32)",
            "Very high risk (SYNTAX score >40)"
        ],
        "explanation": "The SYNTAX score stratifies patients into three risk categories for decision-making between PCI and CABG: low (≤22), intermediate (23-32), and high (>32). A score of 28 places this patient in the intermediate risk category. Current guidelines suggest that intermediate SYNTAX scores represent an area where both PCI and CABG may be reasonable options, and the Heart Team discussion should consider additional patient factors. Low scores generally favor PCI, while high scores favor CABG.",
        "difficulty": "medium",
        "topic": "syntax_scoring",
        "bloom_level": "applying"
    },
    {
        "stem": "When calculating the SYNTAX score for a lesion involving the LAD-diagonal bifurcation with disease in both the parent vessel and the side branch (Medina 1,1,1), which of the following correctly describes the additional SYNTAX score points for this lesion characteristic?",
        "correct_answer": "Bifurcation lesion modifier: additional points based on Medina classification, with angulation <70° adding further points",
        "distractors": [
            "Bifurcation adds a flat 5 points regardless of Medina classification",
            "Only the parent vessel stenosis is scored; the side branch is not considered",
            "Bifurcation involvement is only scored if the side branch is >1.5mm diameter"
        ],
        "explanation": "The SYNTAX scoring system includes specific modifiers for bifurcation lesions classified by the Medina system (describing disease at the proximal main vessel, distal main vessel, and side branch). The presence of a bifurcation adds points, with additional weight for a Medina classification showing disease at all three sites (1,1,1) and for angulation of <70° between the branches, which increases procedural complexity. The scoring is nuanced and depends on multiple geometric factors, not a simple flat addition.",
        "difficulty": "hard",
        "topic": "syntax_scoring",
        "bloom_level": "understanding"
    },
    {
        "stem": "A patient with left main disease and a SYNTAX score of 35 is being evaluated by the Heart Team. Which of the following lesion modifiers most likely contributed to the elevated score?",
        "correct_answer": "Chronic total occlusion of the RCA with duration >3 months, heavy calcification, and diffuse disease (lesion length >20mm)",
        "distractors": [
            "Single focal stenosis in the mid-LAD without calcification",
            "Mild ectasia of the proximal RCA without stenosis",
            "Small-vessel disease limited to distal branches <1.5mm"
        ],
        "explanation": "Chronic total occlusions (CTOs) are among the heaviest contributors to SYNTAX score, especially when present for >3 months, with additional modifiers for blunt stump, bridging collaterals, side branch at the occlusion site, and occlusion length >20mm. Heavy calcification and diffuse disease (lesion length >20mm) each add further points. A score of 35 (high risk) typically requires multiple such modifiers. Single focal lesions without adverse characteristics, ectasia without stenosis, and small vessels below the scoring threshold would not significantly elevate the score.",
        "difficulty": "hard",
        "topic": "syntax_scoring",
        "bloom_level": "analyzing"
    },
]

LESION_MORPHOLOGY_EXAMPLES = [
    {
        "stem": "The angiographic image of the proximal LAD shows a hazy, irregular filling defect within the lumen at the site of a severe stenosis, with reduced TIMI flow. Which lesion characteristic is most likely present?",
        "correct_answer": "Intracoronary thrombus",
        "distractors": [
            "Heavy calcification",
            "Vessel tortuosity",
            "Intimal dissection"
        ],
        "explanation": "A hazy, irregular filling defect within the coronary lumen at the site of stenosis is the classic angiographic appearance of intracoronary thrombus. This is especially common in acute coronary syndromes where plaque rupture or erosion triggers thrombus formation. Heavy calcification appears as radiopaque densities along the vessel wall. Tortuosity refers to vessel course, not luminal appearance. Intimal dissection shows as a linear lucency or contrast staining within the vessel wall.",
        "difficulty": "medium",
        "topic": "lesion_morphology",
        "bloom_level": "analyzing"
    },
    {
        "stem": "Coronary angiography demonstrates a long segment of narrowing in the mid-LAD extending over 25mm with gradual tapering. Which lesion morphology descriptor best characterizes this finding?",
        "correct_answer": "Diffuse disease (lesion length >20mm)",
        "distractors": [
            "Focal stenosis",
            "Tandem lesions",
            "Aorto-ostial lesion"
        ],
        "explanation": "Diffuse disease is defined in the SYNTAX scoring system as a lesion length exceeding 20mm. The gradual tapering over 25mm is characteristic of diffuse atherosclerotic disease rather than a focal plaque. Focal stenosis would be a discrete, short narrowing. Tandem lesions are two distinct stenoses within the same segment separated by a normal or near-normal segment. An aorto-ostial lesion occurs at the very origin of a coronary artery from the aorta.",
        "difficulty": "medium",
        "topic": "lesion_morphology",
        "bloom_level": "understanding"
    },
    {
        "stem": "The mid-RCA demonstrates a segment with multiple severe bends (≥3 consecutive curves of >90°) proximal to a significant stenosis. How does this morphological feature impact interventional planning?",
        "correct_answer": "Severe tortuosity increases procedural complexity and the risk of guide catheter disengagement, wire perforation, and device delivery failure",
        "distractors": [
            "Tortuosity has no impact on PCI outcomes and is not scored in SYNTAX",
            "Tortuosity only affects surgical graft anastomosis, not percutaneous intervention",
            "Tortuosity is favorable because it indicates vessel compliance and low calcification"
        ],
        "explanation": "Severe vessel tortuosity (defined as ≥3 consecutive bends of >90° proximal to the lesion) is a SYNTAX score modifier that adds points reflecting increased procedural complexity. It impairs guide catheter stability, complicates wire advancement (with risk of perforation at bend points), and may prevent delivery of stents and balloons. This is in contrast to calcification, which affects lesion preparation. Tortuosity does not imply low calcification; the two frequently coexist.",
        "difficulty": "hard",
        "topic": "lesion_morphology",
        "bloom_level": "applying"
    },
]

VIEW_IDENTIFICATION_EXAMPLES = [
    {
        "stem": "A coronary angiogram is obtained with the image intensifier positioned to the patient's right and angled toward the head. The LAD and its diagonal branches are well separated. Which angiographic projection was most likely used?",
        "correct_answer": "RAO cranial projection",
        "distractors": [
            "LAO cranial projection",
            "RAO caudal projection",
            "AP cranial projection"
        ],
        "explanation": "The RAO cranial projection positions the image intensifier to the right and tilted toward the head. This view is optimal for separating the LAD from its diagonal branches and for evaluating the mid and distal LAD. The LAO cranial view better visualizes the left main bifurcation and proximal LAD/LCx. The RAO caudal view elongates the LCx and its marginal branches. The AP cranial view shows the LAD in a more foreshortened position with less diagonal separation.",
        "difficulty": "medium",
        "topic": "view_identification",
        "bloom_level": "understanding"
    },
    {
        "stem": "To optimally visualize the left main coronary artery bifurcation into the LAD and circumflex, minimizing vessel overlap, which angiographic projection is most appropriate?",
        "correct_answer": "LAO caudal (spider view) projection",
        "distractors": [
            "RAO cranial projection",
            "Straight AP projection",
            "LAO cranial projection"
        ],
        "explanation": "The LAO caudal projection, also known as the 'spider view,' is the standard view for evaluating the left main bifurcation. It opens up the angle between the LAD and LCx, allowing assessment of the distal left main, ostial LAD, and ostial LCx with minimal overlap. The RAO cranial view is better for LAD-diagonal separation. The straight AP view foreshortens the left main. The LAO cranial view overlaps the proximal LAD with the left main.",
        "difficulty": "easy",
        "topic": "view_identification",
        "bloom_level": "remembering"
    },
    {
        "stem": "An angiographic view shows the RCA coursing from upper right to lower left of the screen, with the PDA and posterolateral branches clearly visualized at the crux. Which projection best matches this appearance?",
        "correct_answer": "LAO cranial projection",
        "distractors": [
            "RAO projection",
            "LAO caudal projection",
            "Lateral projection"
        ],
        "explanation": "The LAO cranial projection provides an excellent view of the distal RCA and its bifurcation into the PDA and posterolateral branches at the crux of the heart. The RCA appears to course from upper right to lower left in this view. The RAO view shows the proximal and mid RCA well but foreshortens the distal RCA at the crux. The LAO caudal view is primarily used for left coronary system. The lateral view provides a profile view of the heart but with significant overlap of structures.",
        "difficulty": "medium",
        "topic": "view_identification",
        "bloom_level": "understanding"
    },
]

CLINICAL_REASONING_EXAMPLES = [
    {
        "stem": "A 63-year-old male with diabetes and CKD stage 3 presents with NSTEMI. Angiography reveals: left main 40% stenosis, proximal LAD 90% stenosis with heavy calcification, mid-LCx 80% stenosis at a bifurcation (Medina 1,1,0), and mid-RCA chronic total occlusion with right-to-left collaterals. The SYNTAX score is 34. The patient has right dominant circulation. Based on these findings, what is the most appropriate revascularization strategy?",
        "correct_answer": "Referral for coronary artery bypass grafting (CABG) given the high SYNTAX score and complex three-vessel disease with CTO",
        "distractors": [
            "Multivessel PCI with drug-eluting stents in a single procedure",
            "Medical therapy alone with optimal guideline-directed medications",
            "Staged PCI: treat the LAD culprit first, then CTO recanalization at a later date"
        ],
        "explanation": "With a SYNTAX score of 34 (high risk, >32), complex three-vessel disease including a CTO, bifurcation disease, and heavy calcification, current guidelines (ESC/EACTS 2018) recommend CABG as the preferred revascularization strategy. The presence of diabetes further strengthens the CABG recommendation (FREEDOM trial). While multivessel PCI is technically feasible, the high SYNTAX score predicts worse outcomes with PCI compared to CABG. Medical therapy alone would not address the high ischemic burden in this acute presentation.",
        "difficulty": "hard",
        "topic": "clinical_reasoning",
        "bloom_level": "evaluating"
    },
    {
        "stem": "A 52-year-old woman presents with stable angina. Angiography shows an isolated 75% stenosis in the mid-LAD (Segment 7) with no calcification, no tortuosity, and favorable lesion morphology. SYNTAX score is 8. Left dominant circulation is noted. FFR of the lesion is 0.72. What is the most appropriate management?",
        "correct_answer": "PCI with drug-eluting stent to the mid-LAD, given low SYNTAX score, favorable anatomy, and FFR-confirmed hemodynamic significance",
        "distractors": [
            "CABG with LIMA-to-LAD graft for prognostic benefit in proximal LAD disease",
            "Medical therapy alone since the stenosis is only 75%",
            "Repeat angiography in 6 months to reassess progression"
        ],
        "explanation": "With a low SYNTAX score (8, ≤22), single-vessel disease, favorable lesion characteristics, and FFR-confirmed hemodynamic significance (0.72, below the 0.80 threshold), PCI with a drug-eluting stent is the recommended approach. The lesion is in the mid-LAD (not proximal), reducing the prognostic argument for CABG. CABG is generally reserved for left main or complex multivessel disease. An FFR of 0.72 confirms that this lesion causes significant ischemia, making medical therapy alone inadequate. Surveillance without intervention ignores the proven ischemia.",
        "difficulty": "medium",
        "topic": "clinical_reasoning",
        "bloom_level": "applying"
    },
    {
        "stem": "During angiography for a 67-year-old male with acute STEMI, the culprit lesion is identified as a thrombotic total occlusion of the proximal RCA (Segment 1) with TIMI 0 flow. A significant non-culprit stenosis of 70% is also noted in the proximal LCx (Segment 11). The patient is hemodynamically stable. What is the recommended approach for the non-culprit LCx lesion?",
        "correct_answer": "Complete revascularization of the non-culprit LCx lesion either during the index procedure or as a planned staged procedure within 45 days",
        "distractors": [
            "Treat only the culprit RCA and defer the LCx lesion unless symptoms recur",
            "Treat both lesions simultaneously in all cases to minimize the number of procedures",
            "Refer for CABG to address both the RCA and LCx disease"
        ],
        "explanation": "The COMPLETE trial demonstrated that in STEMI patients with multivessel disease, complete revascularization (treating non-culprit lesions) reduces the composite of cardiovascular death and MI compared to a culprit-only strategy. The timing—whether during the index procedure or as a staged procedure within 45 days—did not significantly differ in outcomes. The stable hemodynamics in this case allow for either approach. Culprit-only strategy is no longer the preferred approach per current guidelines. CABG is not indicated for two-vessel disease with favorable anatomy.",
        "difficulty": "hard",
        "topic": "clinical_reasoning",
        "bloom_level": "evaluating"
    },
]

# Master dictionary mapping MCQ type to its few-shot examples
FEW_SHOT_EXAMPLES: dict[str, list[dict]] = {
    "vessel_identification": VESSEL_IDENTIFICATION_EXAMPLES,
    "stenosis_severity": STENOSIS_SEVERITY_EXAMPLES,
    "coronary_dominance": CORONARY_DOMINANCE_EXAMPLES,
    "syntax_scoring": SYNTAX_SCORING_EXAMPLES,
    "lesion_morphology": LESION_MORPHOLOGY_EXAMPLES,
    "view_identification": VIEW_IDENTIFICATION_EXAMPLES,
    "clinical_reasoning": CLINICAL_REASONING_EXAMPLES,
}
