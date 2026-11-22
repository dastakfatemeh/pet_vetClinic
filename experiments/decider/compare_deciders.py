"""
Compare decision models as the triage layer: Jev (paid API) vs. open-weight deciders
(perplexity-ai/pplx-decider-v1-27b, Cloudflare/clef, Cloudflare/clef-flash), with VetBERTDx as the
production baseline.

Jev and VetBERTDx outputs are read from the saved Jev experiment results, so this script makes no
TypeSafe calls and costs nothing. Only the open models need a GPU run (run_pplx_decider.py, run_clef.py).

Steps (from the project root):
    python experiments/decider/compare_deciders.py export
    # run the GPU scripts (see README.md)
    python experiments/decider/compare_deciders.py report \
        --pred pplx=experiments/decider/inputs/pplx_job.log \
        --pred clef=experiments/decider/inputs/clef_job.log --pred clefflash=experiments/decider/inputs/clef_job.log

Each --pred is NAME=PATH, where PATH is a predictions CSV or a job log; for a log, the CSV between
=====NAME_CSV_BEGIN===== and =====NAME_CSV_END===== is extracted. NAME is the column prefix.
"""

import argparse
import io
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CONDITIONS, NONE_LABEL  # noqa: E402

HERE = Path(__file__).resolve().parent
JEV_RESULTS_DIR = HERE.parent / "jev" / "results"
INPUTS_PATH = HERE / "inputs" / "texts.csv"
RESULTS_DIR = HERE / "results"

CONFIDENCE_THRESHOLD = 0.8
EMERGENCY_THRESHOLD = 0.5
DISAGREEMENT_THRESHOLD = 0.8
UNCERTAIN = "uncertain"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_saved_results(files: list[str] | None) -> pd.DataFrame:
    """Concatenate the Jev experiment's saved per-message results in a fixed order and number the rows."""
    paths = [Path(f) for f in files] if files else sorted(JEV_RESULTS_DIR.glob("comparison_*.csv"))
    if not paths:
        sys.exit(f"No saved Jev results in {JEV_RESULTS_DIR}. Run experiments/jev/compare_classifiers.py first.")
    saved = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    dupes = saved.duplicated(subset=["eval_set", "text"])
    if dupes.any():
        print(f"dropping {dupes.sum()} duplicate messages found in more than one results file")
        saved = saved[~dupes].reset_index(drop=True)
    saved.insert(0, "row_id", range(len(saved)))
    print(
        f"loaded {len(saved)} messages from {', '.join(p.name for p in paths)}: "
        + ", ".join(f"{k}={v}" for k, v in saved["eval_set"].value_counts().items())
    )
    return saved


def read_predictions(name: str, path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8", errors="replace")
    marker = name.upper()
    match = re.search(rf"====={marker}_CSV_BEGIN=====\s*(.*?)\s*====={marker}_CSV_END=====", text, re.S)
    if match:
        text = match.group(1)
    elif "_CSV_BEGIN=====" in text:
        sys.exit(f"{path} has no ====={marker}_CSV_BEGIN===== block")
    preds = pd.read_csv(io.StringIO(text.strip()))
    missing = {f"{name}_label", f"{name}_conf", f"{name}_emergency"} - set(preds.columns)
    if missing:
        sys.exit(f"{path}: missing columns {sorted(missing)} for model '{name}'")
    return preds


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def gate(label: str, confidence: float) -> str:
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
    accepted = set(expected.split("|"))
    if gated == UNCERTAIN:
        return "correct" if "none" in accepted else "abstained"
    return "correct" if gated in accepted else "wrong"


def escalate(row: pd.Series, decider: str, use_emergency: bool, disagreement: float = DISAGREEMENT_THRESHOLD) -> bool:
    """Production gate (AI_Agents._escalation_reason) with `decider` as the triage model."""
    if use_emergency and row[f"{decider}_emergency"] >= EMERGENCY_THRESHOLD:
        return True
    if row[f"{decider}_label"] == NONE_LABEL:
        return True
    if row[f"{decider}_label"] != row["bert_label"] and row[f"{decider}_conf"] >= disagreement:
        return True
    return row["bert_conf"] < CONFIDENCE_THRESHOLD


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def report_labelled(df: pd.DataFrame, name: str, deciders: list[str]) -> None:
    print(f"\n=== {name} (n={len(df)}) ===")
    table = []
    for m in ["bert"] + deciders:
        correct = (df[f"{m}_label"] == df["expected"]).to_numpy()
        gated = df[f"{m}_gated"]
        accepted = gated != UNCERTAIN
        table.append(
            {
                "model": m,
                "accuracy": accuracy_score(df["expected"], df[f"{m}_label"]),
                "macro_f1": f1_score(df["expected"], df[f"{m}_label"], average="macro", labels=CONDITIONS),
                "ECE": expected_calibration_error(df[f"{m}_conf"].to_numpy(), correct),
                "coverage@0.8": accepted.mean(),
                "acc_when_accepted": (gated[accepted] == df["expected"][accepted]).mean(),
                "none_picks": (df[f"{m}_label"] == NONE_LABEL).mean() if m != "bert" else np.nan,
                "emergency_rate": (df[f"{m}_emergency"] >= EMERGENCY_THRESHOLD).mean() if m != "bert" else np.nan,
                "median_ms": df[f"{m}_ms"].median() if f"{m}_ms" in df else np.nan,
            }
        )
    print(pd.DataFrame(table).round(3).to_string(index=False))
    for other in deciders[1:]:
        agree = df["jev_label"] == df[f"{other}_label"]
        emerg = (df["jev_emergency"] >= EMERGENCY_THRESHOLD) == (df[f"{other}_emergency"] >= EMERGENCY_THRESHOLD)
        print(f"jev vs {other}: label agreement {agree.mean():.3f}, emergency flag agreement {emerg.mean():.3f}")


def report_hard(df: pd.DataFrame, deciders: list[str]) -> None:
    print(f"\n=== hard cases (n={len(df)}) ===")
    models = ["bert"] + deciders
    for m in models:
        df[f"{m}_outcome"] = [outcome(e, g) for e, g in zip(df["expected"], df[f"{m}_gated"])]
    summary = (
        df.melt(id_vars="category", value_vars=[f"{m}_outcome" for m in models], var_name="model", value_name="outcome")
        .assign(model=lambda d: d["model"].str.replace("_outcome", ""))
        .pipe(lambda d: pd.crosstab([d["category"], d["model"]], d["outcome"]))
    )
    print(summary.to_string())

    emergencies = df["emergency"] == 1
    print()
    for m in deciders:
        flagged = df[f"{m}_emergency"] >= EMERGENCY_THRESHOLD
        print(
            f"{m}: emergencies flagged {(flagged & emergencies).sum()}/{emergencies.sum()}, "
            f"false alarms {(flagged & ~emergencies).sum()}/{(~emergencies).sum()}"
        )

    multi = df[df["category"] == "multi_condition"]
    if len(multi):
        for m in deciders:
            if f"{m}_has::{CONDITIONS[0]}" not in df:
                continue
            recalls = []
            for _, row in multi.iterrows():
                truth = set(row["expected"].split("|"))
                found = {c for c in CONDITIONS if row[f"{m}_has::{c}"] >= 0.5}
                recalls.append(len(truth & found) / len(truth))
            print(f"{m}: multi-condition label recall (per-condition yes/no) {np.mean(recalls):.3f}")

    for other in deciders[1:]:
        print(f"\nhard cases where jev and {other} disagree on label or emergency flag:")
        diff = df[
            (df["jev_label"] != df[f"{other}_label"])
            | ((df["jev_emergency"] >= EMERGENCY_THRESHOLD) != (df[f"{other}_emergency"] >= EMERGENCY_THRESHOLD))
        ]
        for _, r in diff.iterrows():
            print(
                f"  [{r['category']}] {r['text'][:65]!r}\n"
                f"      expected={r['expected']} emergency={r['emergency']} | "
                f"jev={r['jev_label']} ({r['jev_conf']:.2f}) E={r['jev_emergency']:.2f} | "
                f"{other}={r[f'{other}_label']} ({r[f'{other}_conf']:.2f}) E={r[f'{other}_emergency']:.2f}"
            )


def report_escalation(results: pd.DataFrame, deciders: list[str]) -> None:
    """How VetBERTDx performs behind each triage model, on the labelled sets."""
    print("\n=== VetBERTDx behind each triage model (production gate) ===")
    rows = []
    for eval_set in ("val", "clinical"):
        df = results[results["eval_set"] == eval_set]
        if df.empty:
            continue
        esc = df["bert_conf"] < CONFIDENCE_THRESHOLD
        variants = [("v1: confidence gate only", esc)]
        for d in deciders:
            variants.append((f"{d}: rules 2-4 @0.8", df.apply(escalate, axis=1, decider=d, use_emergency=False)))
            if d != "jev":  # Jev's 0.8 was tuned on these sets; show the lower threshold for the open models too
                variants.append(
                    (
                        f"{d}: rules 2-4 @0.5",
                        df.apply(escalate, axis=1, decider=d, use_emergency=False, disagreement=0.5),
                    )
                )
            variants.append((f"{d}: rules 1-4 @0.8", df.apply(escalate, axis=1, decider=d, use_emergency=True)))
        for rule, esc in variants:
            answered = ~esc
            rows.append(
                {
                    "set": eval_set,
                    "triage": rule,
                    "coverage": answered.mean(),
                    "bert_acc_on_answered": (df["bert_label"][answered] == df["expected"][answered]).mean(),
                    "wrong_answers_shown": int((df["bert_label"][answered] != df["expected"][answered]).sum()),
                }
            )
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(
        "rules 2-4 match the experiments/jev README table; 1-4 adds the emergency check as in /converse. "
        "@0.5 was chosen after seeing these sets, so treat it as optimistic."
    )


def report(args: argparse.Namespace) -> None:
    preds = dict(p.split("=", 1) for p in (args.pred or []))
    if args.pplx:
        preds.setdefault("pplx", args.pplx)
    if not preds:
        sys.exit("give at least one --pred NAME=PATH")

    results = load_saved_results(args.saved)
    n_saved = len(results)
    for name, path in preds.items():
        results = results.merge(read_predictions(name, Path(path)), on="row_id", how="inner", validate="one_to_one")
    if len(results) < n_saved:
        print(f"warning: predictions cover {len(results)} of {n_saved} messages; reporting on those only")

    deciders = ["jev"] + list(preds)
    for m in ["bert"] + deciders:
        results[f"{m}_gated"] = [gate(l, c) for l, c in zip(results[f"{m}_label"], results[f"{m}_conf"])]

    for eval_set, name in (("val", "validation split"), ("clinical", "clinical notes")):
        part = results[results["eval_set"] == eval_set].reset_index(drop=True)
        if len(part):
            report_labelled(part, name, deciders)
    hard = results[results["eval_set"] == "hard"].reset_index(drop=True)
    if len(hard):
        report_hard(hard, deciders)
    report_escalation(results, deciders)

    print(
        "\nlatency: jev_ms = network round trip for 7 questions; open models' _ms = GPU time for the 2 production "
        "triage questions; bert_ms = local CPU."
    )
    print(
        f"jev tokens: {results['jev_tokens'].sum():,}"
        + "".join(f" | {d} input tokens: {results[f'{d}_tokens'].sum():,}" for d in preds if f"{d}_tokens" in results)
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"deciders_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    results.to_csv(out_path, index=False)
    print(f"per-row results: {out_path}")


def export(args: argparse.Namespace) -> None:
    saved = load_saved_results(args.saved)
    INPUTS_PATH.parent.mkdir(exist_ok=True)
    saved[["row_id", "text"]].to_csv(INPUTS_PATH, index=False)
    print(f"wrote {len(saved)} messages to {INPUTS_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_export = sub.add_parser("export", help="write the messages the open models should score")
    p_report = sub.add_parser("report", help="compare VetBERTDx, Jev and the open models")
    p_report.add_argument("--pred", action="append", help="NAME=PATH (predictions CSV or job log); repeatable")
    p_report.add_argument("--pplx", help="shortcut for --pred pplx=PATH")
    for p in (p_export, p_report):
        p.add_argument("--saved", nargs="+", help="Jev results CSVs (default: all in experiments/jev/results)")
    args = parser.parse_args()
    {"export": export, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
