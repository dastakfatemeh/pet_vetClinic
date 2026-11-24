"""
Score the 445 evaluation messages with Liquid AI's hosted d1 decision model.

d1 takes the same request as Jev ({model, state, questions}) and returns the same response
({model, answers, usage}), so the 7 questions are the Jev experiment's, verbatim (common.build_systemone_questions).

Needs LIQUID_API_KEY in .env (never printed). A full run is ~300k input tokens, about $0.01 at $0.02 per M.

Usage (from the project root):
    python experiments/decider/run_d1.py                      # all 445 messages
    python experiments/decider/run_d1.py --limit 5            # smoke test
    python experiments/decider/compare_deciders.py report \
        --pred clef=experiments/decider/inputs/clef_job.log \
        --pred d1=experiments/decider/inputs/d1_predictions.csv

Output: experiments/decider/inputs/d1_predictions.csv (git-ignored), columns d1_label, d1_conf, d1_emergency,
d1_has::<condition>, d1_ms (network round trip, 7 questions), d1_tokens, d1_cost.
"""

import argparse
import asyncio
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
from common import CONDITIONS, build_systemone_questions  # noqa: E402

URL = "https://api.liquid.ai/decisions/v1/systemone"
RETRY_STATUS = (429, 500, 502, 503, 504)


async def score(texts: pd.DataFrame, key: str, model: str, concurrency: int) -> pd.DataFrame:
    questions = build_systemone_questions()
    semaphore = asyncio.Semaphore(concurrency)
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=60, headers=headers) as client:

        async def ask(row_id: int, text: str) -> dict:
            body = {"model": model, "state": {"owner_message": text}, "questions": questions}
            async with semaphore:
                for attempt in range(5):
                    t0 = time.perf_counter()
                    try:
                        response = await client.post(URL, json=body)
                    except httpx.TransportError:
                        if attempt == 4:
                            raise
                        await asyncio.sleep(2**attempt)
                        continue
                    ms = (time.perf_counter() - t0) * 1000
                    if response.status_code not in RETRY_STATUS:
                        break
                    await asyncio.sleep(2**attempt)
            if response.status_code >= 400:
                raise RuntimeError(f"row {row_id}: HTTP {response.status_code}: {response.text[:300]}")
            data = response.json()
            answers = data["answers"]
            condition = answers["condition"]
            usage = data.get("usage") or {}
            row = {
                "row_id": row_id,
                "d1_label": condition["choice"],
                "d1_conf": condition["probabilities"][condition["choice"]],
                "d1_emergency": answers["emergency"]["noul"],
                "d1_ms": ms,
                "d1_tokens": usage.get("input_tokens"),
                "d1_cost": usage.get("cost"),
            }
            for c in CONDITIONS:
                row[f"d1_has::{c}"] = answers[f"has::{c}"]["noul"]
            return row

        rows = await asyncio.gather(*(ask(r, t) for r, t in zip(texts["row_id"], texts["text"])))
    return pd.DataFrame(rows).sort_values("row_id")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", default=str(HERE / "inputs" / "texts.csv"))
    parser.add_argument("--model", default="d1")
    parser.add_argument("--limit", type=int, help="score only the first N messages")
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    key = os.environ.get("LIQUID_API_KEY")
    if not key:
        sys.exit("Missing LIQUID_API_KEY in .env")

    texts = pd.read_csv(args.inputs)
    if args.limit:
        texts = texts.head(args.limit)
    print(f"scoring {len(texts)} messages with {args.model} at {URL}")
    t0 = time.perf_counter()
    preds = asyncio.run(score(texts, key, args.model, args.concurrency))
    elapsed = time.perf_counter() - t0

    out = Path(args.inputs).with_name("d1_predictions.csv")
    preds.to_csv(out, index=False)
    print(
        f"done in {elapsed:.0f} s: median latency {statistics.median(preds['d1_ms']):.0f} ms (7 questions), "
        f"{int(preds['d1_tokens'].sum()):,} input tokens, cost ${preds['d1_cost'].sum():.4f}"
    )
    print(f"predictions: {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
