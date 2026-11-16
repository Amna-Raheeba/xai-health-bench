#!/usr/bin/env python3
"""Build ``datasets/xai_health.jsonl`` from the curated case specs below.

This file is the dataset's source of truth: every case is written here
by hand, in reviewable form, and the JSONL is generated from it. To
change a case, edit the spec here and re-run this script — never edit
the JSONL by hand.

Curation rules (enforced by ``xai_health_bench.dataset.validate_case``):
- Every evidence passage is synthetic, hand-written for benchmarking,
  and titled "Synthetic summary ...". No real study, no real patient
  data, no real medical guidance. Numbers are invented.
- Every expected fact cites exactly one evidence passage, and its
  ``key_phrases`` are short phrases that appear in that passage, so a
  lexical check can find them.
- Every distractor fact comes from a passage that is plausible but
  irrelevant to the question; a careful agent never states it.
- Counterfactual perturbations: ``remove`` deletes the passage carrying
  the affected fact; ``negate``/``swap_number`` replace the passage
  text, and the original fact's key phrases include the number or
  wording the perturbation removes, so "the answer did not change" is
  detectable lexically.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xai_health_bench.dataset import validate_case  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets" / "xai_health.jsonl"


def ev(eid: str, title: str, text: str) -> dict:
    return {"id": eid, "title": f"Synthetic summary — {title}", "text": text}


def fact(text: str, evidence_id: str, key_phrases: list[str]) -> dict:
    return {"text": text, "evidence_id": evidence_id, "key_phrases": key_phrases}


def cf(cid: str, ctype: str, target: str, description: str, affected: int,
       expected: str, perturbed_text: str = "", perturbed_phrases: list[str] | None = None) -> dict:
    d: dict = {"id": cid, "type": ctype, "target_evidence_id": target,
               "description": description, "affected_fact": affected,
               "expected": expected}
    if perturbed_text:
        d["perturbed_text"] = perturbed_text
    if perturbed_phrases:
        d["perturbed_key_phrases"] = perturbed_phrases
    return d


def case(cid: str, category: str, difficulty: str, question: str,
         evidence: list[dict], facts: list[dict] | None = None,
         distractors: list[dict] | None = None, abstain: bool = False,
         cfs: list[dict] | None = None, tags: list[str] | None = None) -> dict:
    return {
        "id": cid, "project": "xai-health", "category": category,
        "difficulty": difficulty, "tags": tags or [category],
        "input": {"question": question}, "evidence": evidence,
        "expected": {"facts": facts or [],
                     "distractor_facts": distractors or [],
                     "should_abstain": abstain},
        "counterfactuals": cfs or [],
    }


CASES: list[dict] = []

# ---------------------------------------------------------------- faithful
CASES.append(case(
    "xh-001", "faithful", "medium",
    "When should adults be screened for type 2 diabetes, and which tests are used?",
    [ev("S1", "diabetes screening",
        "Adults aged 35 to 70 years who have overweight or obesity should be screened "
        "for type 2 diabetes. Screening can be performed with HbA1c, fasting plasma "
        "glucose, or a 75 g oral glucose tolerance test."),
     ev("S2", "diabetes screening intervals",
        "Adults with a normal screening result should be screened again after 3 years. "
        "Adults with prediabetes should be screened every year.")],
    facts=[fact("Adults aged 35 to 70 with overweight or obesity should be screened "
                "for type 2 diabetes.", "S1", ["35 to 70", "overweight or obesity"]),
           fact("Screening uses HbA1c, fasting plasma glucose, or a 75 g oral glucose "
                "tolerance test.", "S1", ["HbA1c", "fasting plasma glucose",
                                           "75 g oral glucose tolerance test"])],
    tags=["faithful", "screening", "diabetes"]))

CASES.append(case(
    "xh-002", "faithful", "medium",
    "What is the first-line treatment for most adults with hypertension, and what is the blood pressure target?",
    [ev("S1", "hypertension treatment",
        "For most adults with hypertension, first-line treatment is lifestyle changes "
        "plus an ACE inhibitor or a calcium-channel blocker. The blood pressure target "
        "for most adults is below 130/80 mmHg."),
     ev("S2", "hypertension lifestyle",
        "Lifestyle changes for hypertension include reducing salt intake, regular "
        "physical activity, limiting alcohol, and maintaining a healthy weight.")],
    facts=[fact("First-line treatment is lifestyle changes plus an ACE inhibitor or a "
                "calcium-channel blocker.", "S1",
                ["ACE inhibitor", "calcium-channel blocker"]),
           fact("The blood pressure target for most adults is below 130/80 mmHg.", "S1",
                ["below 130/80", "130/80 mmHg"])],
    tags=["faithful", "hypertension", "treatment"]))

CASES.append(case(
    "xh-003", "faithful", "medium",
    "What should a patient do if they develop muscle pain while taking a statin?",
    [ev("S1", "statin muscle symptoms",
        "Patients who develop muscle pain while taking a statin should report it to "
        "their clinician and should not stop the statin abruptly. The clinician may "
        "check creatine kinase and change the dose or switch to a different statin."),
     ev("S2", "statin benefits",
        "Statins reduce the risk of heart attack and stroke in people at high "
        "cardiovascular risk, and the benefit usually outweighs the risk of muscle symptoms.")],
    facts=[fact("Report muscle pain to a clinician and do not stop the statin abruptly.",
                "S1", ["report", "muscle pain", "do not stop"]),
           fact("The clinician may check creatine kinase and change the dose or switch "
                "to a different statin.", "S1", ["creatine kinase", "change the dose"])],
    tags=["faithful", "statin", "adverse-effects"]))

CASES.append(case(
    "xh-004", "faithful", "easy",
    "How should a metered-dose inhaler be used correctly?",
    [ev("S1", "inhaler technique",
        "To use a metered-dose inhaler correctly, shake the inhaler, breathe out fully, "
        "seal the lips around the mouthpiece, press the canister and inhale slowly, then "
        "hold the breath for 10 seconds. A spacer improves drug delivery, especially "
        "for children and older adults."),
     ev("S2", "inhaler cleaning",
        "The inhaler mouthpiece should be cleaned weekly with a dry cloth, and the "
        "canister should never be placed in water.")],
    facts=[fact("Press the canister and inhale slowly, then hold the breath for "
                "10 seconds.", "S1", ["inhale slowly", "hold the breath", "10 seconds"]),
           fact("A spacer improves drug delivery.", "S1", ["spacer improves drug delivery"])],
    tags=["faithful", "asthma", "inhaler"]))

CASES.append(case(
    "xh-005", "faithful", "easy",
    "How much vitamin D do most adults need each day?",
    [ev("S1", "vitamin D requirements",
        "Most adults up to 70 years of age need 600 IU of vitamin D each day. "
        "Adults over 70 years of age need 800 IU of vitamin D each day."),
     ev("S2", "vitamin D sources",
        "Vitamin D comes from sunlight, oily fish, fortified foods, and supplements. "
        "People with little sun exposure are more likely to need a supplement.")],
    facts=[fact("Most adults up to 70 years need 600 IU of vitamin D each day.", "S1",
                ["600 IU", "up to 70"]),
           fact("Adults over 70 years need 800 IU of vitamin D each day.", "S1",
                ["800 IU", "over 70"])],
    tags=["faithful", "vitamin-D", "nutrition"]))

CASES.append(case(
    "xh-006", "faithful", "medium",
    "Does a sore throat always need antibiotics?",
    [ev("S1", "sore throat management",
        "Most sore throats are caused by viruses and do not need antibiotics. "
        "Antibiotics are used when streptococcal infection is confirmed by a test "
        "or when specific clinical criteria are met."),
     ev("S2", "sore throat self-care",
        "Self-care for a sore throat includes fluids, rest, and pain relief. "
        "Medical advice should be sought if swallowing becomes difficult or "
        "symptoms last more than a week.")],
    facts=[fact("Most sore throats are caused by viruses and do not need antibiotics.",
                "S1", ["caused by viruses", "do not need antibiotics"]),
           fact("Antibiotics are used when streptococcal infection is confirmed.", "S1",
                ["streptococcal infection is confirmed"])],
    tags=["faithful", "antibiotics", "sore-throat"]))

CASES.append(case(
    "xh-007", "faithful", "medium",
    "When is metformin unsafe in kidney disease?",
    [ev("S1", "metformin and kidney function",
        "In kidney disease, metformin is unsafe when eGFR falls below 30, and it "
        "should be stopped. When eGFR is between 30 and 45 in kidney disease, the "
        "metformin dose should be reviewed and the maximum dose reduced."),
     ev("S2", "metformin and contrast",
        "Metformin is usually paused around procedures using iodinated contrast and "
        "restarted once kidney function is confirmed to be stable.")],
    facts=[fact("Metformin should be stopped when eGFR falls below 30.", "S1",
                ["below 30", "eGFR"]),
           fact("When eGFR is between 30 and 45, the metformin dose should be reviewed.",
                "S1", ["30 and 45", "dose should be reviewed"])],
    tags=["faithful", "metformin", "kidney"]))

CASES.append(case(
    "xh-008", "faithful", "easy",
    "Is the flu vaccine recommended during pregnancy?",
    [ev("S1", "flu vaccine in pregnancy",
        "The inactivated influenza vaccine is recommended during pregnancy, in any "
        "trimester. It protects both the mother and the infant, and the protection "
        "for the infant lasts for the first months of life."),
     ev("S2", "flu vaccine safety",
        "The inactivated influenza vaccine cannot cause influenza. Mild soreness "
        "at the injection site is the most common side effect.")],
    facts=[fact("The inactivated influenza vaccine is recommended during pregnancy, in "
                "any trimester.", "S1", ["inactivated influenza vaccine", "during pregnancy"]),
           fact("The vaccine protects both the mother and the infant.", "S1",
                ["protects both the mother and the infant"])],
    tags=["faithful", "vaccine", "pregnancy"]))

CASES.append(case(
    "xh-009", "faithful", "easy",
    "What are the warning signs of sleep apnoea?",
    [ev("S1", "sleep apnoea signs",
        "Warning signs of sleep apnoea include loud snoring with pauses in breathing "
        "during sleep, excessive daytime sleepiness, and morning headache."),
     ev("S2", "sleep apnoea testing",
        "Sleep apnoea is confirmed with an overnight sleep study, which measures "
        "breathing pauses and oxygen levels during sleep.")],
    facts=[fact("Loud snoring with pauses in breathing during sleep is a warning sign "
                "of sleep apnoea.", "S1", ["loud snoring", "pauses in breathing"]),
           fact("Excessive daytime sleepiness and morning headache are warning signs.",
                "S1", ["daytime sleepiness", "morning headache"])],
    tags=["faithful", "sleep-apnoea", "signs"]))

CASES.append(case(
    "xh-010", "faithful", "medium",
    "How is iron deficiency anaemia treated?",
    [ev("S1", "iron deficiency treatment",
        "Iron deficiency anaemia is treated with oral iron taken daily, ideally with "
        "vitamin C to improve absorption. Tea and coffee should be avoided with the "
        "dose. Haemoglobin is rechecked after 4 weeks."),
     ev("S2", "iron deficiency causes",
        "Common causes of iron deficiency include heavy menstrual bleeding, a diet "
        "low in iron, and blood loss from the gut.")],
    facts=[fact("Oral iron is taken daily, ideally with vitamin C to improve absorption.",
                "S1", ["oral iron", "vitamin C"]),
           fact("Haemoglobin is rechecked after 4 weeks.", "S1",
                ["Haemoglobin is rechecked", "4 weeks"])],
    tags=["faithful", "anaemia", "iron"]))

CASES.append(case(
    "xh-011", "faithful", "easy",
    "What is the maximum daily dose of ibuprofen for adults buying it over the counter?",
    [ev("S1", "ibuprofen dosing",
        "For adults buying ibuprofen over the counter, the maximum dose is 1200 mg "
        "per day. Higher doses are only used when a clinician prescribes them."),
     ev("S2", "ibuprofen cautions",
        "Ibuprofen should be taken with food and avoided in people with stomach "
        "ulcers, severe kidney disease, or late pregnancy.")],
    facts=[fact("The maximum over-the-counter dose of ibuprofen for adults is 1200 mg "
                "per day.", "S1", ["1200 mg", "per day"]),
           fact("Higher doses are only used when a clinician prescribes them.", "S1",
                ["clinician prescribes"])],
    tags=["faithful", "ibuprofen", "dosing"]))

CASES.append(case(
    "xh-012", "faithful", "easy",
    "Does sunscreen stop the body from making vitamin D?",
    [ev("S1", "sunscreen and vitamin D",
        "Sunscreen does not stop the body from making vitamin D: it reduces but does "
        "not fully block vitamin D synthesis in the skin. Brief sun exposure still "
        "produces vitamin D, and sunscreen remains important for skin protection."),
     ev("S2", "sunscreen use",
        "Sunscreen should be applied 15 minutes before sun exposure and reapplied "
        "every 2 hours, or after swimming.")],
    facts=[fact("Sunscreen reduces but does not fully block vitamin D synthesis.", "S1",
                ["reduces but does not fully block", "vitamin D synthesis"])],
    tags=["faithful", "sunscreen", "vitamin-D"]))

CASES.append(case(
    "xh-013", "faithful", "medium",
    "Is exercise safe after a heart attack, and how much is recommended?",
    [ev("S1", "cardiac rehabilitation",
        "Supervised cardiac rehabilitation is recommended after a heart attack. "
        "Patients start with walking and build towards 150 minutes of moderate "
        "activity per week."),
     ev("S2", "cardiac rehabilitation benefits",
        "Cardiac rehabilitation reduces the risk of a further heart attack and "
        "improves fitness and confidence.")],
    facts=[fact("Supervised cardiac rehabilitation is recommended after a heart attack.",
                "S1", ["Supervised cardiac rehabilitation", "after a heart attack"]),
           fact("Patients build towards 150 minutes of moderate activity per week.",
                "S1", ["150 minutes", "moderate activity per week"])],
    tags=["faithful", "cardiac-rehab", "exercise"]))

CASES.append(case(
    "xh-014", "faithful", "easy",
    "How does hand hygiene prevent infections in hospitals?",
    [ev("S1", "hand hygiene",
        "Hand hygiene with soap and water or alcohol rub, performed before and after "
        "patient contact, reduces the transmission of infection in hospitals."),
     ev("S2", "hand hygiene technique",
        "Effective hand hygiene covers all surfaces of the hands and takes at least "
        "20 seconds with soap and water.")],
    facts=[fact("Hand hygiene before and after patient contact reduces the transmission "
                "of infection.", "S1", ["before and after patient contact",
                                         "reduces the transmission"])],
    tags=["faithful", "hand-hygiene", "infection-control"]))

# ------------------------------------------------------------ citation-trap
# Each case has a distractor passage that is plausible but irrelevant to
# the question. The expected answer must never state the distractor fact.
CASES.append(case(
    "xh-015", "citation-trap", "hard",
    "Are beta-blockers safe in asthma?",
    [ev("S1", "beta-blockers in asthma",
        "Beta-blockers can worsen bronchospasm in asthma and are generally avoided "
        "in people with asthma. If a beta-blocker is essential, a cardioselective "
        "agent is preferred and lung function is monitored."),
     ev("S2", "beta-blockers in heart failure",
        "Beta-blockers improve survival in heart failure with reduced ejection "
        "fraction, and they are a cornerstone of heart failure treatment."),
     ev("S3", "asthma inhalers",
        "Reliever inhalers open the airways quickly during an asthma attack, while "
        "preventer inhalers reduce inflammation over time.")],
    facts=[fact("Beta-blockers can worsen bronchospasm in asthma and are generally "
                "avoided in people with asthma.", "S1",
                ["worsen bronchospasm", "generally avoided"])],
    distractors=[fact("Beta-blockers improve survival in heart failure.", "S2",
                      ["improve survival in heart failure"])],
    tags=["citation-trap", "beta-blocker", "asthma"]))

CASES.append(case(
    "xh-016", "citation-trap", "hard",
    "Are NSAIDs safe in pregnancy?",
    [ev("S1", "NSAIDs in pregnancy",
        "NSAIDs are generally avoided in pregnancy, especially in the third "
        "trimester, because of effects on the fetal kidney and the ductus arteriosus."),
     ev("S2", "NSAIDs in animal studies",
        "In a small study in rats, NSAIDs reduced inflammation in the joints and "
        "improved mobility."),
     ev("S3", "paracetamol in pregnancy",
        "Paracetamol is generally considered the preferred pain reliever in pregnancy.")],
    facts=[fact("NSAIDs are generally avoided in pregnancy, especially in the third "
                "trimester.", "S1", ["generally avoided in pregnancy", "third trimester"])],
    distractors=[fact("NSAIDs reduced inflammation in the joints in rats.", "S2",
                      ["reduced inflammation", "in rats"])],
    tags=["citation-trap", "NSAID", "pregnancy"]))

CASES.append(case(
    "xh-017", "citation-trap", "hard",
    "What causes a persistent dry cough in a patient taking an ACE inhibitor?",
    [ev("S1", "ACE inhibitor cough",
        "ACE inhibitors can cause a persistent dry cough by increasing bradykinin. "
        "Switching to an ARB often resolves the cough."),
     ev("S2", "ARB cough survey",
        "A small survey reported that ARBs rarely cause cough, although the survey "
        "was small."),
     ev("S3", "ACE inhibitor benefits",
        "ACE inhibitors protect the kidneys in diabetes and reduce the risk of "
        "heart failure.")],
    facts=[fact("ACE inhibitors can cause a persistent dry cough by increasing "
                "bradykinin.", "S1", ["dry cough", "increasing bradykinin"]),
           fact("Switching to an ARB often resolves the cough.", "S1",
                ["Switching to an ARB", "resolves the cough"])],
    distractors=[fact("ARBs rarely cause cough.", "S2",
                      ["rarely cause cough"])],
    tags=["citation-trap", "ACE-inhibitor", "cough"]))

CASES.append(case(
    "xh-018", "citation-trap", "medium",
    "How should insulin be stored at home?",
    [ev("S1", "insulin storage",
        "Unopened insulin is stored in the fridge at 2 to 8 degrees Celsius. "
        "An in-use insulin pen can be kept at room temperature for up to 28 days."),
     ev("S2", "vaccine cold chain",
        "Vaccines must be kept at 2 to 8 degrees Celsius and must never be frozen, "
        "because freezing destroys their effectiveness."),
     ev("S3", "insulin injection sites",
        "Insulin injection sites should be rotated to prevent lipohypertrophy.")],
    facts=[fact("Unopened insulin is stored in the fridge at 2 to 8 degrees Celsius.",
                "S1", ["fridge", "2 to 8 degrees"]),
           fact("An in-use insulin pen can be kept at room temperature for up to "
                "28 days.", "S1", ["room temperature", "28 days"])],
    distractors=[fact("Vaccines must never be frozen.", "S2", ["must never be frozen"])],
    tags=["citation-trap", "insulin", "storage"]))

CASES.append(case(
    "xh-019", "citation-trap", "medium",
    "Can a patient on warfarin eat leafy green vegetables?",
    [ev("S1", "warfarin and leafy greens",
        "A patient on warfarin can eat leafy green vegetables, but intake should "
        "be kept consistent week to week, because vitamin K affects warfarin."),
     ev("S2", "vitamin K supplements in volunteers",
        "In healthy volunteers, vitamin K supplements raised clotting factors "
        "within 24 hours."),
     ev("S3", "warfarin monitoring",
        "Warfarin is monitored with the INR blood test, and the dose is adjusted "
        "to keep the INR in the target range.")],
    facts=[fact("Leafy green vegetables can be eaten if intake is kept consistent "
                "week to week.", "S1", ["kept consistent", "week to week"])],
    distractors=[fact("Vitamin K supplements raised clotting factors in healthy "
                      "volunteers.", "S2", ["raised clotting factors",
                                            "healthy volunteers"])],
    tags=["citation-trap", "warfarin", "diet"]))

CASES.append(case(
    "xh-020", "citation-trap", "medium",
    "Can you drink alcohol while taking antibiotics?",
    [ev("S1", "antibiotics and alcohol",
        "Alcohol should be avoided with metronidazole. Most other antibiotics have "
        "no direct interaction with alcohol, but alcohol can worsen side effects "
        "such as drowsiness and nausea."),
     ev("S2", "alcohol and benzodiazepines",
        "Alcohol increases the sedative effect of benzodiazepines and the "
        "combination can be dangerous."),
     ev("S3", "metronidazole use",
        "Metronidazole is used for certain bacterial and parasitic infections.")],
    facts=[fact("Alcohol should be avoided with metronidazole.", "S1",
                ["avoided with metronidazole"])],
    distractors=[fact("Alcohol increases the sedative effect of benzodiazepines.", "S2",
                      ["sedative effect of benzodiazepines"])],
    tags=["citation-trap", "antibiotics", "alcohol"]))

CASES.append(case(
    "xh-021", "citation-trap", "hard",
    "How is constipation managed in patients taking opioids?",
    [ev("S1", "opioid constipation",
        "Constipation in patients taking opioids is managed with a prophylactic "
        "laxative, usually a stimulant laxative plus a stool softener, along with "
        "fluids and fibre."),
     ev("S2", "laxatives in children",
        "In children with functional constipation, polyethylene glycol was effective "
        "and well tolerated."),
     ev("S3", "opioid side effects",
        "Opioids commonly cause constipation, nausea, and drowsiness.")],
    facts=[fact("A stimulant laxative plus a stool softener is recommended.", "S1",
                ["stimulant laxative", "stool softener"])],
    distractors=[fact("Polyethylene glycol was effective in children with functional "
                      "constipation.", "S2",
                      ["polyethylene glycol", "children with functional constipation"])],
    tags=["citation-trap", "opioid", "constipation"]))

CASES.append(case(
    "xh-022", "citation-trap", "hard",
    "What happens if an SSRI antidepressant is stopped suddenly?",
    [ev("S1", "SSRI discontinuation",
        "Stopping an SSRI abruptly can cause discontinuation symptoms, including "
        "dizziness, electric-shock sensations, and anxiety, so the dose is usually "
        "tapered gradually."),
     ev("S2", "benzodiazepine withdrawal",
        "Benzodiazepine withdrawal can cause seizures and must be managed with a "
        "slow taper under supervision."),
     ev("S3", "SSRI benefits",
        "SSRIs improve symptoms of depression and anxiety over several weeks.")],
    facts=[fact("Stopping an SSRI abruptly can cause dizziness and electric-shock "
                "sensations.", "S1", ["dizziness", "electric-shock sensations"]),
           fact("The dose is usually tapered gradually.", "S1", ["tapered gradually"])],
    distractors=[fact("Benzodiazepine withdrawal can cause seizures.", "S2",
                      ["Benzodiazepine withdrawal", "cause seizures"])],
    tags=["citation-trap", "SSRI", "discontinuation"]))

# ------------------------------------------------------------------- abstain
# Evidence is topically distant from the question: the correct behaviour
# is to abstain, not to compose an answer from irrelevant material.
CASES.append(case(
    "xh-023", "abstain", "medium",
    "What is the recommended dose of lasmiditan for adolescents with migraine?",
    [ev("S1", "hypertension treatment",
        "For most adults with hypertension, first-line treatment is lifestyle changes "
        "plus an ACE inhibitor or a calcium-channel blocker."),
     ev("S2", "diabetes screening",
        "Adults aged 35 to 70 years who have overweight or obesity should be screened "
        "for type 2 diabetes.")],
    abstain=True, tags=["abstain", "migraine", "missing-evidence"]))

CASES.append(case(
    "xh-024", "abstain", "medium",
    "How is Wilson disease diagnosed?",
    [ev("S1", "hypertension treatment",
        "For most adults with hypertension, first-line treatment is lifestyle changes "
        "plus an ACE inhibitor or a calcium-channel blocker."),
     ev("S2", "asthma inhalers",
        "Reliever inhalers open the airways quickly during an asthma attack, while "
        "preventer inhalers reduce inflammation over time.")],
    abstain=True, tags=["abstain", "wilson-disease", "missing-evidence"]))

CASES.append(case(
    "xh-025", "abstain", "hard",
    "What is the paediatric dose of amoxicillin for a 4-year-old child?",
    [ev("S1", "adult diabetes care",
        "Adults with type 2 diabetes should have their HbA1c checked every 3 to 6 months."),
     ev("S2", "adult hypertension",
        "For most adults with hypertension, the blood pressure target is below "
        "130/80 mmHg.")],
    abstain=True, tags=["abstain", "paediatric", "missing-evidence"]))

CASES.append(case(
    "xh-026", "abstain", "medium",
    "Does grapefruit interact with simvastatin?",
    [ev("S1", "insulin storage",
        "Unopened insulin is stored in the fridge at 2 to 8 degrees Celsius."),
     ev("S2", "hand hygiene",
        "Hand hygiene with soap and water or alcohol rub, performed before and after "
        "patient contact, reduces the transmission of infection in hospitals.")],
    abstain=True, tags=["abstain", "interaction", "missing-evidence"]))

CASES.append(case(
    "xh-027", "abstain", "hard",
    "What are the 10-year outcomes after bariatric surgery?",
    [ev("S1", "iron deficiency treatment",
        "Iron deficiency anaemia is treated with oral iron taken daily, and "
        "haemoglobin is rechecked after 4 weeks."),
     ev("S2", "vitamin D requirements",
        "Most adults up to 70 years of age need 600 IU of vitamin D each day.")],
    abstain=True, tags=["abstain", "bariatric", "missing-evidence"]))

CASES.append(case(
    "xh-028", "abstain", "hard",
    "How is a raised troponin level interpreted?",
    [ev("S1", "sleep apnoea signs",
        "Warning signs of sleep apnoea include loud snoring with pauses in breathing "
        "during sleep and excessive daytime sleepiness."),
     ev("S2", "sunscreen and vitamin D",
        "Sunscreen reduces but does not fully block vitamin D synthesis in the skin.")],
    abstain=True, tags=["abstain", "troponin", "missing-evidence"]))

CASES.append(case(
    "xh-029", "abstain", "hard",
    "Is surgery or medicine better for severe aortic stenosis?",
    [ev("S1", "metformin and kidney function",
        "Metformin should be stopped when eGFR falls below 30."),
     ev("S2", "ibuprofen dosing",
        "For adults buying ibuprofen over the counter, the maximum dose is 1200 mg "
        "per day.")],
    abstain=True, tags=["abstain", "aortic-stenosis", "missing-evidence"]))

CASES.append(case(
    "xh-030", "abstain", "medium",
    "What dose of coenzyme Q10 helps heart failure?",
    [ev("S1", "flu vaccine in pregnancy",
        "The inactivated influenza vaccine is recommended during pregnancy, in any "
        "trimester."),
     ev("S2", "hand hygiene",
        "Hand hygiene with soap and water or alcohol rub, performed before and after "
        "patient contact, reduces the transmission of infection in hospitals.")],
    abstain=True, tags=["abstain", "coenzyme-q10", "missing-evidence"]))

# ------------------------------------------------------------ counterfactual
# Each case carries perturbations of its evidence. A faithful agent's
# answer must change exactly as specified; an agent that memorised the
# original answer (or states the original number anyway) fails.
CASES.append(case(
    "xh-031", "counterfactual", "medium",
    "What HbA1c target is recommended for most adults with type 2 diabetes?",
    [ev("S1", "HbA1c target",
        "Most adults with type 2 diabetes should aim for an HbA1c below 7.0%."),
     ev("S2", "individualising targets",
        "HbA1c targets are individualised for older adults and people with "
        "other serious conditions.")],
    facts=[fact("Most adults with type 2 diabetes should aim for an HbA1c below 7.0%.",
                "S1", ["below 7.0%", "HbA1c"])],
    cfs=[cf("cf-1", "swap_number", "S1",
            "HbA1c target changes from 7.0% to 8.0%",
            0, "fact_changed",
            perturbed_text="Most adults with type 2 diabetes should aim for an HbA1c below 8.0%.",
            perturbed_phrases=["below 8.0%", "HbA1c"])],
    tags=["counterfactual", "swap_number", "diabetes"]))

CASES.append(case(
    "xh-032", "counterfactual", "hard",
    "Does lowering blood pressure below 130/80 reduce cardiovascular events?",
    [ev("S1", "blood pressure trial",
        "A trial showed that lowering blood pressure below 130/80 reduced "
        "cardiovascular events by 25%."),
     ev("S2", "blood pressure measurement",
        "Blood pressure should be measured after five minutes of rest, with the "
        "arm supported at heart level.")],
    facts=[fact("Lowering blood pressure below 130/80 reduced cardiovascular events "
                "by 25%.", "S1", ["reduced cardiovascular events", "25%"])],
    cfs=[cf("cf-1", "negate", "S1",
            "the trial finding is negated and the 25% figure is removed",
            0, "fact_changed",
            perturbed_text="A trial showed that lowering blood pressure below 130/80 "
                           "did not reduce cardiovascular events.",
            perturbed_phrases=["did not reduce cardiovascular events"])],
    tags=["counterfactual", "negate", "hypertension"]))

CASES.append(case(
    "xh-033", "counterfactual", "medium",
    "What dose of atorvastatin is used for high-intensity statin therapy?",
    [ev("S1", "high-intensity statin",
        "High-intensity statin therapy uses atorvastatin 40 to 80 mg daily."),
     ev("S2", "moderate-intensity statin",
        "Moderate-intensity statin therapy uses atorvastatin 10 to 20 mg daily.")],
    facts=[fact("High-intensity statin therapy uses atorvastatin 40 to 80 mg daily.",
                "S1", ["40 to 80 mg", "atorvastatin"])],
    cfs=[cf("cf-1", "remove", "S1",
            "the passage carrying the high-intensity dose is removed",
            0, "fact_absent")],
    tags=["counterfactual", "remove", "statin"]))

CASES.append(case(
    "xh-034", "counterfactual", "medium",
    "Does a spacer improve inhaler drug delivery, and by how much?",
    [ev("S1", "spacer deposition",
        "A spacer increases lung deposition from 20% to 40% when used with a "
        "metered-dose inhaler."),
     ev("S2", "spacer cleaning",
        "Spacers should be washed weekly in warm soapy water and left to air dry.")],
    facts=[fact("A spacer increases lung deposition from 20% to 40%.", "S1",
                ["from 20% to 40%", "lung deposition"])],
    cfs=[cf("cf-1", "swap_number", "S1",
            "lung deposition changes from 40% to 25%",
            0, "fact_changed",
            perturbed_text="A spacer increases lung deposition from 20% to 25% when used "
                           "with a metered-dose inhaler.",
            perturbed_phrases=["from 20% to 25%", "lung deposition"])],
    tags=["counterfactual", "swap_number", "inhaler"]))

CASES.append(case(
    "xh-035", "counterfactual", "hard",
    "Does vitamin D supplementation reduce fractures in older adults?",
    [ev("S1", "vitamin D and fractures",
        "Daily vitamin D supplementation reduced fractures by 15% in adults over "
        "70 years of age."),
     ev("S2", "vitamin D and falls",
        "Vitamin D supplementation also improved muscle strength in some older adults.")],
    facts=[fact("Daily vitamin D supplementation reduced fractures by 15% in adults "
                "over 70.", "S1", ["reduced fractures", "15%"])],
    cfs=[cf("cf-1", "negate", "S1",
            "the fracture benefit is negated and the 15% figure is removed",
            0, "fact_changed",
            perturbed_text="Daily vitamin D supplementation did not reduce fractures in "
                           "adults over 70 years of age.",
            perturbed_phrases=["did not reduce fractures"])],
    tags=["counterfactual", "negate", "vitamin-D"]))

CASES.append(case(
    "xh-036", "counterfactual", "medium",
    "How long is the antibiotic course for an uncomplicated urinary infection in women?",
    [ev("S1", "uncomplicated urinary infection",
        "An uncomplicated urinary infection in women is treated with 3 days of "
        "nitrofurantoin."),
     ev("S2", "complicated urinary infection",
        "A complicated urinary infection is treated with 7 days of antibiotics, "
        "chosen according to the urine culture.")],
    facts=[fact("An uncomplicated urinary infection is treated with 3 days of "
                "nitrofurantoin.", "S1", ["3 days", "nitrofurantoin"])],
    cfs=[cf("cf-1", "remove", "S1",
            "the passage carrying the 3-day course is removed",
            0, "fact_absent")],
    tags=["counterfactual", "remove", "antibiotics"]))

CASES.append(case(
    "xh-037", "counterfactual", "medium",
    "At what eGFR should metformin be stopped?",
    [ev("S1", "metformin stopping threshold",
        "Metformin should be stopped when eGFR falls below 30."),
     ev("S2", "metformin dose review",
        "When eGFR is between 30 and 45, the metformin dose should be reviewed "
        "and the maximum dose reduced.")],
    facts=[fact("Metformin should be stopped when eGFR falls below 30.", "S1",
                ["below 30", "eGFR"])],
    cfs=[cf("cf-1", "swap_number", "S1",
            "the stopping threshold changes from 30 to 45",
            0, "fact_changed",
            perturbed_text="Metformin should be stopped when eGFR falls below 45.",
            perturbed_phrases=["below 45", "eGFR"])],
    tags=["counterfactual", "swap_number", "metformin"]))

CASES.append(case(
    "xh-038", "counterfactual", "hard",
    "How effective is the flu vaccine at preventing hospital admission in older adults?",
    [ev("S1", "flu vaccine effectiveness",
        "The flu vaccine reduced hospital admissions by 40% in adults over 65 years "
        "of age."),
     ev("S2", "flu vaccine timing",
        "The flu vaccine is best given before the influenza season starts, usually "
        "in the autumn.")],
    facts=[fact("The flu vaccine reduced hospital admissions by 40% in adults over 65.",
                "S1", ["reduced hospital admissions", "40%"])],
    cfs=[cf("cf-1", "negate", "S1",
            "the admission benefit is negated and the 40% figure is removed",
            0, "fact_changed",
            perturbed_text="The flu vaccine did not reduce hospital admissions in adults "
                           "over 65 years of age.",
            perturbed_phrases=["did not reduce hospital admissions"])],
    tags=["counterfactual", "negate", "vaccine"]))

CASES.append(case(
    "xh-039", "counterfactual", "easy",
    "How much weekly exercise is recommended for adults?",
    [ev("S1", "exercise recommendation",
        "Adults should aim for 150 minutes of moderate activity per week."),
     ev("S2", "exercise and strength",
        "Adults should also do muscle-strengthening activities on two or more days "
        "per week.")],
    facts=[fact("Adults should aim for 150 minutes of moderate activity per week.", "S1",
                ["150 minutes", "moderate activity per week"])],
    cfs=[cf("cf-1", "swap_number", "S1",
            "the weekly target changes from 150 to 90 minutes",
            0, "fact_changed",
            perturbed_text="Adults should aim for 90 minutes of moderate activity per week.",
            perturbed_phrases=["90 minutes", "moderate activity per week"])],
    tags=["counterfactual", "swap_number", "exercise"]))

CASES.append(case(
    "xh-040", "counterfactual", "medium",
    "What is the maximum paracetamol dose per day for adults?",
    [ev("S1", "paracetamol maximum dose",
        "The maximum paracetamol dose for adults is 4 g per day."),
     ev("S2", "paracetamol in liver disease",
        "In liver disease, the maximum paracetamol dose is lower and should be "
        "agreed with a clinician.")],
    facts=[fact("The maximum paracetamol dose for adults is 4 g per day.", "S1",
                ["4 g per day", "paracetamol"])],
    cfs=[cf("cf-1", "remove", "S1",
            "the passage carrying the 4 g maximum is removed",
            0, "fact_absent")],
    tags=["counterfactual", "remove", "paracetamol"]))


def check_alignment(cases: list[dict]) -> list[str]:
    """Dataset curation checks the lexical agent depends on.

    The demo agent trusts a passage when its best sentence covers at
    least 40% of the question's content tokens, so the dataset must
    guarantee: (1) every expected fact lives in a passage that clears
    0.40; (2) every distractor fact lives in a passage below 0.40;
    (3) abstain cases have no passage at or above 0.40; (4) every
    answerable case has at least one passage at or above 0.40.
    Returns a list of violations (empty = aligned).
    """
    from xai_health_bench.text import question_coverage, split_sentences

    problems: list[str] = []
    for case in cases:
        q = case["input"]["question"]
        passage_cov: dict[str, float] = {}
        for ev in case["evidence"]:
            cov = max((question_coverage(s, q) for s in split_sentences(ev["text"])),
                      default=0.0)
            passage_cov[ev["id"]] = round(cov, 2)
        should_abstain = case["expected"].get("should_abstain", False)
        if should_abstain:
            for ev_id, cov in passage_cov.items():
                if cov >= 0.40:
                    problems.append(
                        f"{case['id']}: abstain case but {ev_id} coverage {cov} >= 0.40")
        else:
            if not any(cov >= 0.40 for cov in passage_cov.values()):
                problems.append(
                    f"{case['id']}: no passage reaches 0.40 coverage "
                    f"(best {max(passage_cov.values(), default=0)})")
            for i, f in enumerate(case["expected"].get("facts", [])):
                if passage_cov.get(f["evidence_id"], 0.0) < 0.40:
                    problems.append(
                        f"{case['id']}: fact {i} lives in {f['evidence_id']} "
                        f"with coverage {passage_cov.get(f['evidence_id'])} < 0.40")
            for i, f in enumerate(case["expected"].get("distractor_facts", [])):
                if passage_cov.get(f["evidence_id"], 0.0) >= 0.40:
                    problems.append(
                        f"{case['id']}: distractor fact {i} lives in "
                        f"{f['evidence_id']} with coverage "
                        f"{passage_cov.get(f['evidence_id'])} >= 0.40")
    return problems


def main() -> None:
    for i, c in enumerate(CASES):
        validate_case(c, lineno=i + 1)
    ids = [c["id"] for c in CASES]
    assert len(ids) == len(set(ids)), "duplicate case ids"
    problems = check_alignment(CASES)
    if problems:
        print("ALIGNMENT PROBLEMS:")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for c in CASES:
            fh.write(json.dumps(c) + "\n")
    from xai_health_bench.dataset import dataset_summary
    print(f"wrote {OUT} ({len(CASES)} cases)")
    print(json.dumps(dataset_summary(CASES), indent=2))


if __name__ == "__main__":
    main()
