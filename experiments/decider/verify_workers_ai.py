"""
Verify Clef / Clef-flash on Cloudflare Workers AI against the open-weight results from HF Jobs.

Three checks per model:
  1. Production path: the app's own TriageAgent (built by Agents/triage_config.py) answers a few messages
     through Workers AI, exactly as /converse would.
  2. Parity: the same 7 questions used on HF Jobs are sent to Workers AI for a sample of messages, and the
     hosted probabilities are compared with the open-weight ones (label agreement, probability differences,
     emergency recall on the hard cases).
  3. Contract: the raw response shape (envelope or not), token usage and latency are recorded.

Needs CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN in .env (never printed), plus the saved inputs and
HF Jobs log in experiments/decider/inputs/.

Usage (from the project root):
    python experiments/decider/verify_workers_ai.py                 # clef + clef-flash, 95 messages each
    python experiments/decider/verify_workers_ai.py --models clef --sample 0   # hard cases only

The default sample (45 hard + 25 val + 25 clinical, ~1,200 tokens each) is ~115k input tokens per model,
inside Workers AI's free daily allowance.
"""

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_ROOT))
from common import CONDITIONS, build_systemone_questions  # noqa: E402
from compare_deciders import load_saved_results, read_predictions  # noqa: E402
from Agents.triage_config import WORKERS_AI_URL, build_triage_agent, load_triage_settings  # noqa: E402

PREFIX = {"clef": "clef", "clef-flash": "clefflash"}
CLEF_LOG = HERE / "inputs" / "clef_job.log"
RESULTS_DIR = HERE / "results"
EMERGENCY_THRESHOLD = {"clef": 0.5, "clef-flash": 0.8}  # production defaults from Agents/triage_config.py


def unwrap(payload: dict) -> dict:
    if "result" in payload and "answers" not in payload:
        if payload.get("success") is False:
            raise RuntimeError(f"Workers AI error: {payload.get('errors')}")
        return payload["result"]
    return payload


def redact(obj, secrets):
    text = json.dumps(obj, indent=2)
    for s in secrets:
        if s:
            text = text.replace(s, "<redacted>")
    return text


async def production_check(model: str, env: dict, messages: list) -> None:
    print(f"\n[{model}] 1. production path (Agents.triage_config -> TriageAgent -> Workers AI)")
    settings = load_triage_settings({**env, "TRIAGE_MODEL": model})
    agent = build_triage_agent(settings, env)
    try:
        for text in messages:
            t0 = time.perf_counter()
            result = await agent.assess(text)
            ms = (time.perf_counter() - t0) * 1000
            print(
                f"   {ms:6.0f} ms  emergency={result.emergency_probability:.2f}  "
                f"{result.condition} ({result.condition_confidence:.2f})  <- {text[:60]!r}"
            )
    finally:
        await agent.close()


async def parity_run(model: str, env: dict, sample: pd.DataFrame, concurrency: int, secrets) -> pd.DataFrame:
    print(f"\n[{model}] 2. parity: {len(sample)} messages x 7 questions on Workers AI")
    url = WORKERS_AI_URL.format(account_id=env["CLOUDFLARE_ACCOUNT_ID"], model=model)
    headers = {"Authorization": f"Bearer {env['CLOUDFLARE_API_TOKEN']}", "Content-Type": "application/json"}
    questions = build_systemone_questions()
    semaphore = asyncio.Semaphore(concurrency)
    first_raw = {}

    async with httpx.AsyncClient(timeout=30, headers=headers) as client:

        async def ask(row_id: int, text: str) -> dict:
            body = {"model": model, "state": {"owner_message": text}, "questions": questions}
            async with semaphore:
                for attempt in range(4):
                    t0 = time.perf_counter()
                    response = await client.post(url, json=body)
                    ms = (time.perf_counter() - t0) * 1000
                    if response.status_code not in (429, 500, 502, 503, 504):
                        break
                    await asyncio.sleep(2**attempt)
            if response.status_code >= 400:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
            payload = response.json()
            if not first_raw:
                first_raw["body"] = payload
            data = unwrap(payload)
            answers = data["answers"]
            condition = answers["condition"]
            row = {
                "row_id": row_id,
                "wai_label": condition["choice"],
                "wai_conf": condition["probabilities"][condition["choice"]],
                "wai_emergency": answers["emergency"]["noul"],
                "wai_ms": ms,
                "wai_tokens": (data.get("usage") or {}).get("input_tokens"),
            }
            for c in CONDITIONS:
                row[f"wai_has::{c}"] = answers[f"has::{c}"]["noul"]
            return row

        rows = await asyncio.gather(*(ask(r, t) for r, t in zip(sample["row_id"], sample["text"])))

    raw = first_raw.get("body", {})
    print("   3. contract: top-level keys of a raw response:", sorted(raw))
    envelope = "result" in raw and "answers" not in raw
    print(f"      Cloudflare envelope: {'yes, payload under result' if envelope else 'no, payload at top level'}")
    RESULTS_DIR.mkdir(exist_ok=True)
    sample_path = RESULTS_DIR / f"workers_ai_{model}_sample_response.json"
    sample_path.write_text(redact(raw, secrets), encoding="utf-8")
    print(f"      raw response saved (secrets redacted): {sample_path.relative_to(PROJECT_ROOT)}")
    return pd.DataFrame(rows)


def report(model: str, merged: pd.DataFrame) -> dict:
    p = PREFIX[model]
    agree = (merged["wai_label"] == merged[f"{p}_label"]).mean()
    d_emerg = (merged["wai_emergency"] - merged[f"{p}_emergency"]).abs()
    d_conf = (merged["wai_conf"] - merged[f"{p}_conf"]).abs()
    hard = merged[merged["eval_set"] == "hard"]
    thr = EMERGENCY_THRESHOLD[model]
    emerg = hard["emergency"] == 1
    flagged_wai = hard["wai_emergency"] >= thr
    flagged_hf = hard[f"{p}_emergency"] >= thr
    labelled = merged[merged["eval_set"].isin(["val", "clinical"])]
    summary = {
        "model": model,
        "messages": len(merged),
        "label_agreement_with_hf_jobs": round(agree, 3),
        "emergency_prob_mean_abs_diff": round(d_emerg.mean(), 4),
        "emergency_prob_max_abs_diff": round(d_emerg.max(), 4),
        "condition_conf_mean_abs_diff": round(d_conf.mean(), 4),
        f"emergency_flag_agreement@{thr}": round(
            ((merged["wai_emergency"] >= thr) == (merged[f"{p}_emergency"] >= thr)).mean(), 3
        ),
        "hard_emergencies_flagged_workers_ai": f"{(flagged_wai & emerg).sum()}/{emerg.sum()}",
        "hard_emergencies_flagged_hf_jobs": f"{(flagged_hf & emerg).sum()}/{emerg.sum()}",
        "hard_false_alarms_workers_ai": f"{(flagged_wai & ~emerg).sum()}/{(~emerg).sum()}",
        "labelled_accuracy_workers_ai": round((labelled["wai_label"] == labelled["expected"]).mean(), 3),
        "labelled_accuracy_hf_jobs": round((labelled[f"{p}_label"] == labelled["expected"]).mean(), 3),
        "median_latency_ms_7q": round(statistics.median(merged["wai_ms"]), 0),
        "input_tokens_total": int(merged["wai_tokens"].fillna(0).sum()),
    }
    print(f"\n[{model}] parity vs HF Jobs open weights")
    for k, v in summary.items():
        print(f"   {k:40s} {v}")
    diff = merged[merged["wai_label"] != merged[f"{p}_label"]]
    for _, r in diff.head(8).iterrows():
        print(
            f"   label differs: {r['text'][:60]!r} hf={r[f'{p}_label']} ({r[f'{p}_conf']:.2f}) "
            f"wai={r['wai_label']} ({r['wai_conf']:.2f})"
        )
    return summary


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["clef", "clef-flash"], choices=list(PREFIX))
    parser.add_argument("--sample", type=int, default=25, help="val and clinical messages each (plus all 45 hard)")
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    env = dict(os.environ)
    missing = [k for k in ("CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN") if not env.get(k)]
    if missing:
        sys.exit(f"Missing in .env: {', '.join(missing)}")
    secrets = [env["CLOUDFLARE_API_TOKEN"], env["CLOUDFLARE_ACCOUNT_ID"]]

    saved = load_saved_results(None)
    hard = saved[saved["eval_set"] == "hard"]
    parts = [hard] + [
        saved[saved["eval_set"] == s].sample(min(args.sample, (saved["eval_set"] == s).sum()), random_state=42)
        for s in ("val", "clinical")
        if args.sample
    ]
    sample = pd.concat(parts)

    summaries = []
    for model in args.models:
        await production_check(model, env, hard["text"].head(3).tolist())
        wai = await parity_run(model, env, sample, args.concurrency, secrets)
        hf = read_predictions(PREFIX[model], CLEF_LOG)
        merged = sample.merge(hf, on="row_id").merge(wai, on="row_id")
        summaries.append(report(model, merged))
        out = RESULTS_DIR / f"workers_ai_{model}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        merged.to_csv(out, index=False)
        print(f"   per-row results: {out.relative_to(PROJECT_ROOT)}")

    print("\nsummary:")
    print(pd.DataFrame(summaries).set_index("model").T.to_string())


if __name__ == "__main__":
    asyncio.run(main())
