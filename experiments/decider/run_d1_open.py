# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch==2.11.0", "torchvision==0.26.0", "transformers==5.24.0",
#   "accelerate", "huggingface-hub", "safetensors", "pillow", "pandas>=2.0",
# ]
# ///
"""
Run Liquid AI's open-weight d1 decision models on the comparison messages.

    LiquidAI/d1-3B   3.1B (LFM2.5-VL-3B), ~6.2 GB BF16 weights, needs transformers>=5.14
    LiquidAI/d1-2B   1.9B (LFM3-VL-2B),   ~3.9 GB BF16 weights, needs transformers>=5.19 (d1-3B's successor)

Both are small enough for a 24 GB GPU (HF Jobs l4x1). The models ship their own code (trust_remote_code) and
expose model.system_one(state, questions), which takes the Jev question schema. So they get the Jev
experiment's questions verbatim with state = {"owner_message": text}, like Clef.

Each model's predictions are written as CSV and also printed to stdout between markers, so a cloud job's
logs are enough to recover them. Several --model values run one after another in the same job.

Usage:
    hf jobs uv run --flavor l4x1 -v ./experiments/decider:/work experiments/decider/run_d1_open.py \
        --inputs /work/inputs/texts.csv --model LiquidAI/d1-3B --model LiquidAI/d1-2B
"""

import argparse
import gc
import importlib.util
import io
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModel

PREFIXES = {"LiquidAI/d1-3B": "d13b", "LiquidAI/d1-2B": "d12b"}


def load_common(here: Path):
    for candidate in (here / "common.py", Path("/work/common.py")):
        if candidate.exists():
            spec = importlib.util.spec_from_file_location("common", candidate)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    sys.exit("common.py not found next to the script or at /work/common.py")


def timed(fn, device: str):
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return result, (time.perf_counter() - t0) * 1000


def run_model(model_id: str, inputs: pd.DataFrame, common, args) -> pd.DataFrame:
    prefix = PREFIXES.get(model_id, model_id.split("/")[-1].replace("-", "").lower())
    start = time.perf_counter()
    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    model = AutoModel.from_pretrained(model_id, revision=args.revision, trust_remote_code=True, dtype=dtype)
    model = model.to(args.device).eval()
    print(f"[{prefix}] {model_id} loaded in {time.perf_counter() - start:.0f}s", flush=True)
    if args.device == "cuda":
        print(f"[{prefix}] GPU memory after load: {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)

    questions = common.build_systemone_questions()
    triage_questions = {k: questions[k] for k in ("emergency", "condition")}  # what production TriageAgent asks

    rows = []
    with torch.inference_mode():
        for i, (row_id, text) in enumerate(zip(inputs["row_id"], inputs["text"])):
            state = {"owner_message": text}
            _, triage_ms = timed(lambda: model.system_one(state, triage_questions), args.device)
            response = model.system_one(state, questions)
            if i == 0:
                print(f"[{prefix}] first response: {response}", flush=True)
            answers = response["answers"]
            condition = answers["condition"]
            row = {
                "row_id": row_id,
                f"{prefix}_label": condition["choice"],
                f"{prefix}_conf": condition["probabilities"][condition["choice"]],
                f"{prefix}_emergency": answers["emergency"]["noul"],
                f"{prefix}_ms": triage_ms,
                f"{prefix}_tokens": response["usage"]["input_tokens"],
            }
            for c in common.CONDITIONS:
                row[f"{prefix}_has::{c}"] = answers[f"has::{c}"]["noul"]
            rows.append(row)
            if (i + 1) % 50 == 0:
                print(f"[{prefix}] {i + 1}/{len(inputs)} done, {time.perf_counter() - start:.0f}s elapsed", flush=True)

    if args.device == "cuda":
        print(f"[{prefix}] peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    result = pd.DataFrame(rows)
    print(
        f"[{prefix}] total {time.perf_counter() - start:.0f}s, median triage latency "
        f"{result[f'{prefix}_ms'].median():.0f} ms, input tokens {result[f'{prefix}_tokens'].sum():,}",
        flush=True,
    )

    out_path = Path(args.inputs).with_name(f"{prefix}_predictions.csv")
    try:
        result.to_csv(out_path, index=False)
        print(f"[{prefix}] wrote {out_path}")
    except OSError as e:
        print(f"[{prefix}] could not write {out_path} ({e}); use the CSV printed below")
    buffer = io.StringIO()
    result.to_csv(buffer, index=False)
    marker = prefix.upper()
    print(f"====={marker}_CSV_BEGIN=====\n{buffer.getvalue().strip()}\n====={marker}_CSV_END=====", flush=True)

    model = None  # release the reference so the GPU memory can be freed for the next model
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", required=True, help="CSV with row_id,text")
    parser.add_argument("--model", action="append", required=True, help="repeat to run several models")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0, help="only the first N messages (smoke test)")
    args = parser.parse_args()

    common = load_common(Path(__file__).resolve().parent)
    inputs = pd.read_csv(args.inputs)
    if args.limit:
        inputs = inputs.head(args.limit)
    print(f"{len(inputs)} messages, models: {', '.join(args.model)}", flush=True)
    if args.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    failed = []
    for model_id in args.model:
        try:
            run_model(model_id, inputs, common, args)
        except Exception as e:  # keep going so one model's failure doesn't lose the other's results
            print(f"FAILED {model_id}: {type(e).__name__}: {e}", flush=True)
            failed.append(model_id)
    if failed:
        sys.exit(f"failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
