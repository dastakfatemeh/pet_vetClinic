"""Shared definitions for the decision-model comparison (no heavy dependencies, safe to import anywhere)."""

NONE_LABEL = "none of these"

# Identical to experiments/jev/compare_classifiers.py and Agents/TriageAgent.py, so every decider sees the same options
CONDITION_CRITERIA = {
    "digestive issues": "Stomach or intestinal problems: vomiting, diarrhea, constipation, bloating, gas, "
    "appetite loss tied to the gut, swallowed objects.",
    "ear infections": "Problems of the ear canal or flap: head shaking, ear scratching, discharge, odor, "
    "redness or swelling of the ear, head tilt.",
    "mobility problems": "Problems moving: limping, stiffness, weakness or paralysis of limbs, reluctance to "
    "jump or climb stairs, joint or back pain.",
    "parasites": "Fleas, ticks, mites, lice, or intestinal worms and protozoa, including signs such as visible "
    "worms, flea dirt, or scooting.",
    "skin irritations": "Skin and coat problems not caused by the ear: itching, rashes, redness, hot spots, "
    "hair loss, flaking, lumps, hives or allergic swelling.",
    NONE_LABEL: "The message does not describe any of the conditions above, for example a different body "
    "system (eyes, teeth, urinary, respiratory), a poisoning, or a non-medical question.",
}
CONDITIONS = [c for c in CONDITION_CRITERIA if c != NONE_LABEL]

# Jev referred to the message as `owner_message` (a field of its state dict). pplx-decider takes the message
# as a plain string under "State:", so the wording points there instead. Everything else is unchanged.
EMERGENCY_INSTRUCTIONS = (
    "A pet owner wrote the message in State. Does it describe a situation where the pet should be seen by a "
    "veterinarian within the next few hours (for example trouble breathing, suspected poisoning, unproductive "
    "retching with a swollen belly, inability to urinate, seizures, sudden paralysis, major trauma, or collapse)?"
)
CONDITION_INSTRUCTIONS = (
    "A pet owner wrote the message in State. Which condition category best matches the pet's problem? "
    "Choose the single best fit."
)


def build_systemone_questions() -> dict:
    """The Jev experiment's questions verbatim, for Jev-compatible models (Clef) that take a dict state.

    Use with state = {"owner_message": text}, exactly as experiments/jev/compare_classifiers.py does.
    """
    questions = {
        "condition": {
            "type": "choice",
            "instructions": "A pet owner wrote `owner_message`. Which condition category best matches the "
            "pet's problem? Choose the single best fit.",
            "criteria": CONDITION_CRITERIA,
        },
        "emergency": {
            "type": "noul",
            "instructions": "A pet owner wrote `owner_message`. Does it describe a situation where the pet "
            "should be seen by a veterinarian within the next few hours (for example trouble breathing, "
            "suspected poisoning, unproductive retching with a swollen belly, inability to urinate, "
            "seizures, sudden paralysis, major trauma, or collapse)?",
        },
    }
    for condition in CONDITIONS:
        questions[f"has::{condition}"] = {
            "type": "noul",
            "instructions": f"A pet owner wrote `owner_message`. Does the pet show signs of {condition}? "
            f"Definition: {CONDITION_CRITERIA[condition]}",
        }
    return questions


def build_questions() -> dict:
    """The same 7 questions the Jev experiment asked, in pplx-decider's question format."""
    questions = {
        "condition": {"type": "choice", "instructions": CONDITION_INSTRUCTIONS, "criteria": CONDITION_CRITERIA},
        "emergency": {"type": "noul", "instructions": EMERGENCY_INSTRUCTIONS},
    }
    for condition in CONDITIONS:
        questions[f"has::{condition}"] = {
            "type": "noul",
            "instructions": f"A pet owner wrote the message in State. Does the pet show signs of {condition}? "
            f"Definition: {CONDITION_CRITERIA[condition]}",
        }
    return questions
