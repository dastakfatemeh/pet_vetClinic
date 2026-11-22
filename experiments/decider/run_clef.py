# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch==2.11.0", "torchvision==0.26.0", "transformers==5.10.2",
#   "accelerate", "huggingface-hub", "safetensors", "pillow", "pandas>=2.0",
# ]
# ///
"""
Run Cloudflare's open-weight Clef decision models on the comparison messages. GPU only.

    Cloudflare/clef        27B (Qwen3.8-27B), ~54 GB BF16 weights -> 80 GB GPU
    Cloudflare/clef-flash  9.4B (Qwen3.5-9B), ~19 GB BF16 weights -> 24 GB+ GPU

Clef is Jev-compatible, so it gets the Jev experiment's questions verbatim with state = {"owner_message": text},
and all 7 questions are answered in one forward pass. Torch/transformers are pinned to the versions the model
cards were tested with.

Each model's predictions are written as CSV and also printed to stdout between markers, so a cloud job's
logs are enough to recover them. Several --model values run one after another in the same job.

Usage:
    uv run experiments/decider/run_clef.py --inputs experiments/decider/inputs/texts.csv --model Cloudflare/clef-flash
    hf jobs uv run --flavor a100-large -v ./experiments/decider:/work experiments/decider/run_clef.py \
        --inputs /work/inputs/texts.csv --model Cloudflare/clef-flash --model Cloudflare/clef
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
from huggingface_hub import snapshot_download

PREFIXES = {"Cloudflare/clef": "clef", "Cloudflare/clef-flash": "clefflash"}


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
    prefix = PREFIXES.get(model_id, model_id.split("/")[-1].replace("-", ""))
    start = time.perf_counter()
    path = snapshot_download(model_id, revision=args.revision)
    sys.path.insert(0, path)
    import joint_schema_model as jsm  # noqa: E402  (shipped in the model repo)

    model, processor = jsm.load_release_model(path, device=args.device)
    print(f"[{prefix}] {model_id} loaded in {time.perf_counter() - start:.0f}s", flush=True)
    if args.device == "cuda":
        print(f"[{prefix}] GPU memory after load: {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)

    questions = common.build_systemone_questions()
    triage_questions = {k: questions[k] for k in ("emergency", "condition")}  # what production TriageAgent asks
    name = model_id.split("/")[-1]

    rows = []
    for i, (row_id, text) in enumerate(zip(inputs["row_id"], inputs["text"])):
        state = {"owner_message": text}
        _, triage_ms = timed(
            lambda: jsm.systemone(model, processor, {"model": name, "state": state, "questions": triage_questions}),
            args.device,
        )
        response = jsm.systemone(model, processor, {"model": name, "state": state, "questions": questions})
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

    # Free the GPU for the next model
    model = processor = None  # release references so the GPU memory can be freed
    sys.path.remove(path)
    sys.modules.pop("joint_schema_model", None)
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
