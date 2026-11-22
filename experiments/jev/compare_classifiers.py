"""
Compare the production VetBERTDx classifier against zero-shot Jev (TypeSafe System One).

Evaluation sets:
    val         The notebook's held-out split (owner observations, stratified 20%, seed 42).
    clinical    Clinical notes the classifier never trained on (different writing register).
    hard        Hand-written hard cases: emergencies, out-of-scope, multi-condition, slang, ...

Both models go through the same production gate (CONFIDENCE_THRESHOLD = 0.8):
a prediction below the gate becomes "uncertain" (the pipeline offers an appointment).

Usage (from the project root):
    python experiments/jev/compare_classifiers.py
    python experiments/jev/compare_classifiers.py --sets val hard --clinical-sample 200
"""

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul, RetryPolicy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from Agents.ClassificationAgent import ClassificationAgent  # noqa: E402

DATA_PATH = PROJECT_ROOT / "Raw_Data" / "pet-health-symptoms-dataset.csv"
HARD_CASES_PATH = Path(__file__).parent / "hard_cases.csv"
RESULTS_DIR = Path(__file__).parent / "results"

CLASSIFIER_REPO = "fdastak/model_classification"
CONFIDENCE_THRESHOLD = 0.8  # Same gate as AI_Agents.py
EMERGENCY_THRESHOLD = 0.5
NONE_LABEL = "none of these"
UNCERTAIN = "uncertain"

# Generic veterinary descriptions, written independently of the dataset to avoid leaking test text
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


def build_questions() -> dict:
    """All questions run in one request per message; they are independent and answered in parallel."""
    questions = {
        "condition": Choice(
            instructions="A pet owner wrote `owner_message`. Which condition category best matches the "
            "pet's problem? Choose the single best fit.",
            criteria=CONDITION_CRITERIA,
        ),
        "emergency": Noul(
            instructions="A pet owner wrote `owner_message`. Does it describe a situation where the pet "
            "should be seen by a veterinarian within the next few hours (for example trouble breathing, "
            "suspected poisoning, unproductive retching with a swollen belly, inability to urinate, "
            "seizures, sudden paralysis, major trauma, or collapse)?"
        ),
    }
    for condition in CONDITIONS:
        questions[f"has::{condition}"] = Noul(
            instructions=f"A pet owner wrote `owner_message`. Does the pet show signs of {condition}? "
            f"Definition: {CONDITION_CRITERIA[condition]}"
        )
    return questions


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_val_split() -> pd.DataFrame:
    """Rebuild the validation split exactly as Clean_Classification.ipynb did."""
    data = pd.read_csv(DATA_PATH).applymap(lambda x: x.lower() if isinstance(x, str) else x)
    owner = data[data["record_type"] == "owner observation"].copy()
    _, val_texts, _, val_labels = train_test_split(
        owner["text"], owner["condition"], stratify=owner["condition"], test_size=0.2, random_state=42
    )
    return pd.DataFrame({"text": val_texts.values, "expected": val_labels.values, "category": "val"})


def load_clinical(sample: int) -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)
    clinical = data[(data["record_type"] == "Clinical Notes") & data["condition"].str.lower().isin(CONDITIONS)]
    if sample and sample < len(clinical):
        clinical = clinical.groupby("condition", group_keys=False).apply(
            lambda g: g.sample(sample // len(CONDITIONS), random_state=42)
        )
    return pd.DataFrame(
        {"text": clinical["text"].values, "expected": clinical["condition"].str.lower().values, "category": "clinical"}
    )


def load_hard_cases() -> pd.DataFrame:
    return pd.read_csv(HARD_CASES_PATH)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def run_classifier(texts: list[str]) -> pd.DataFrame:
    """Run the production ClassificationAgent, one message at a time, as /converse does."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_REPO)
    tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_REPO)
    agent = ClassificationAgent(model, tokenizer, device)

    rows = []
    for text in texts:
        start = time.perf_counter()
        label, confidence = agent.identify_condition(text)
        rows.append({"bert_label": label, "bert_conf": confidence, "bert_ms": (time.perf_counter() - start) * 1000})
    return pd.DataFrame(rows)


async def run_jev(texts: list[str], concurrency: int) -> pd.DataFrame:
    questions = build_questions()
    semaphore = asyncio.Semaphore(concurrency)
    retry = RetryPolicy(max_retries=3, backoff_max=2.0, timeout=15.0)

    async with AsyncTypeSafeClient() as client:

        async def ask(text: str) -> dict:
            async with semaphore:
                start = time.perf_counter()
                response = await client.system_one({"owner_message": text}, questions, retry=retry)
                elapsed = (time.perf_counter() - start) * 1000
            condition = response.choices["condition"]
            row = {
                "jev_label": condition.choice,
                "jev_conf": condition.probabilities[condition.choice],
                "jev_emergency": response.nouls["emergency"].noul,
                "jev_ms": elapsed,
                "jev_tokens": response.usage.input_tokens + response.usage.output_tokens,
            }
            for c in CONDITIONS:
                row[f"jev_has::{c}"] = response.nouls[f"has::{c}"].noul
            return row

        return pd.DataFrame(await asyncio.gather(*(ask(t) for t in texts)))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def gate(label: str, confidence: float) -> str:
    """Apply the production confidence gate; Jev's explicit no-match also becomes 'uncertain'."""
    if label == NONE_LABEL or confidence < CONFIDENCE_THRESHOLD:
        return UNCERTAIN
    return label


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return ece


def outcome(expected: str, gated: str) -> str:
    """correct / abstained / wrong, where expected may list several acceptable labels with '|'."""
    accepted = set(expected.split("|"))
    if gated == UNCERTAIN:
        return "correct" if "none" in accepted else "abstained"
    return "correct" if gated in accepted else "wrong"


def report_labelled(df: pd.DataFrame, name: str) -> None:
    print(f"\n=== {name} (n={len(df)}) ===")
    table = []
    for model in ("bert", "jev"):
        correct = (df[f"{model}_label"] == df["expected"]).to_numpy()
        gated = df[f"{model}_gated"]
        accepted = gated != UNCERTAIN
        table.append(
            {
                "model": model,
                "accuracy": accuracy_score(df["expected"], df[f"{model}_label"]),
                "macro_f1": f1_score(df["expected"], df[f"{model}_label"], average="macro", labels=CONDITIONS),
                "ECE": expected_calibration_error(df[f"{model}_conf"].to_numpy(), correct),
                "coverage@0.8": accepted.mean(),
                "acc_when_accepted": (gated[accepted] == df["expected"][accepted]).mean(),
                "median_ms": df[f"{model}_ms"].median(),
            }
        )
    print(pd.DataFrame(table).round(3).to_string(index=False))
    agree = df["bert_label"] == df["jev_label"]
    print(
        f"agreement: {agree.mean():.3f} | accuracy when both agree: "
        f"{(df['bert_label'][agree] == df['expected'][agree]).mean():.3f} | "
        f"accuracy when they disagree (bert/jev): {(df['bert_label'][~agree] == df['expected'][~agree]).mean():.3f}"
        f" / {(df['jev_label'][~agree] == df['expected'][~agree]).mean():.3f}"
    )
    print(
        f"jev emergency flag rate (expected ~0 on routine messages): {(df['jev_emergency'] >= EMERGENCY_THRESHOLD).mean():.3f}"
    )


def report_hard(df: pd.DataFrame) -> None:
    print(f"\n=== hard cases (n={len(df)}) ===")
    for model in ("bert", "jev"):
        df[f"{model}_outcome"] = [outcome(e, g) for e, g in zip(df["expected"], df[f"{model}_gated"])]
    summary = (
        df.melt(id_vars="category", value_vars=["bert_outcome", "jev_outcome"], var_name="model", value_name="outcome")
        .assign(model=lambda d: d["model"].str.replace("_outcome", ""))
        .pipe(lambda d: pd.crosstab([d["category"], d["model"]], d["outcome"]))
    )
    print(summary.to_string())

    emergencies = df["emergency"] == 1
    flagged = df["jev_emergency"] >= EMERGENCY_THRESHOLD
    print(
        f"\nemergency recall - bert: 0/{emergencies.sum()} (no urgency signal) | "
        f"jev: {(flagged & emergencies).sum()}/{emergencies.sum()} | "
        f"jev false alarms on non-emergencies: {(flagged & ~emergencies).sum()}/{(~emergencies).sum()}"
    )

    multi = df[df["category"] == "multi_condition"]
    if len(multi):
        recalls = []
        for _, row in multi.iterrows():
            truth = set(row["expected"].split("|"))
            found = {c for c in CONDITIONS if row[f"jev_has::{c}"] >= 0.5}
            recalls.append(len(truth & found) / len(truth))
        print(
            f"multi-condition label recall - bert top-1: "
            f"{np.mean([len({b} & set(e.split('|'))) / len(e.split('|')) for b, e in zip(multi['bert_label'], multi['expected'])]):.3f}"
            f" | jev per-condition nouls: {np.mean(recalls):.3f}"
        )

    print("\nmost confident BERT mistakes on hard cases:")
    mistakes = df[df["bert_outcome"] == "wrong"].nlargest(8, "bert_conf")
    for _, row in mistakes.iterrows():
        print(
            f"  [{row['bert_conf']:.2f}] {row['text'][:70]!r} -> bert={row['bert_label']}, "
            f"jev={row['jev_label']} ({row['jev_conf']:.2f}), emergency={row['jev_emergency']:.2f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sets", nargs="+", default=["val", "clinical", "hard"], choices=["val", "clinical", "hard"])
    parser.add_argument("--clinical-sample", type=int, default=200, help="0 = all ~1000 clinical notes")
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("TYPESAFE_API_KEY"):
        sys.exit("TYPESAFE_API_KEY is not set in .env")

    loaders = {"val": load_val_split, "clinical": lambda: load_clinical(args.clinical_sample), "hard": load_hard_cases}
    frames = {name: loaders[name]() for name in args.sets}
    combined = pd.concat([f.assign(eval_set=name) for name, f in frames.items()], ignore_index=True)
    texts = combined["text"].tolist()

    print(f"Running VetBERTDx on {len(texts)} messages...")
    bert = run_classifier(texts)
    print(f"Running Jev on {len(texts)} messages...")
    jev = asyncio.run(run_jev(texts, args.concurrency))

    results = pd.concat([combined, bert, jev], axis=1)
    results["bert_gated"] = [gate(l, c) for l, c in zip(results["bert_label"], results["bert_conf"])]
    results["jev_gated"] = [gate(l, c) for l, c in zip(results["jev_label"], results["jev_conf"])]

    if "val" in frames:
        report_labelled(results[results["eval_set"] == "val"].reset_index(drop=True), "validation split")
    if "clinical" in frames:
        report_labelled(results[results["eval_set"] == "clinical"].reset_index(drop=True), "clinical notes")
    if "hard" in frames:
        hard = results[results["eval_set"] == "hard"].reset_index(drop=True)
        report_hard(hard)
        results.loc[results["eval_set"] == "hard", ["bert_outcome", "jev_outcome"]] = hard[
            ["bert_outcome", "jev_outcome"]
        ].values

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"comparison_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    results.to_csv(out_path, index=False)
    total_tokens = results["jev_tokens"].sum()
    print(f"\nJev tokens used: {total_tokens:,} | per-row results: {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
