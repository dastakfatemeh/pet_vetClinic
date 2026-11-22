# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch==2.14.0", "torchvision==0.29.0", "transformers==5.17.0",
#   "accelerate==1.15.0", "huggingface-hub==1.31.0",
#   "pillow==12.3.0", "safetensors==0.8.0", "pandas>=2.0",
# ]
# ///
"""
Run perplexity-ai/pplx-decider-v1-27b on the comparison messages. GPU only.

The model needs about 49 GiB for its BF16 weights, so it needs an 80 GB-class GPU (A100 80GB, H100, H200,
RTX PRO 6000). This script does nothing except inference; compare_deciders.py does all the scoring locally.

Input:  a CSV with columns row_id, text   (made by `python experiments/decider/compare_deciders.py export`)
Output: a CSV with one row per message and pplx_* columns. It is also printed to stdout between
        markers, so a cloud job's logs are enough to recover it.

Usage:
    uv run experiments/decider/run_pplx_decider.py --inputs experiments/decider/inputs/texts.csv
    hf jobs uv run --flavor a100-large --secrets HF_TOKEN -v ./experiments/decider:/work \
        experiments/decider/run_pplx_decider.py --inputs /work/inputs/texts.csv
"""

import argparse
import importlib.util
import io
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import hf_hub_download

MODEL_ID = "perplexity-ai/pplx-decider-v1-27b"
CSV_BEGIN, CSV_END = "=====PPLX_CSV_BEGIN=====", "=====PPLX_CSV_END====="


def load_common(here: Path):
    """common.py lives next to this script; when only the script was uploaded, fall back to the mounted copy."""
    for candidate in (here / "common.py", Path("/work/common.py")):
        if candidate.exists():
            spec = importlib.util.spec_from_file_location("common", candidate)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    sys.exit("common.py not found next to the script or at /work/common.py")


def load_decider(model_id: str, revision: str, device: str):
    """Use the model repo's own inference.py so prompting and calibration match the release exactly."""
    path = hf_hub_download(model_id, "inference.py", revision=revision)
    spec = importlib.util.spec_from_file_location("pplx_inference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    decider = module.Decider.from_pretrained(model_id, revision=revision, device=device)
    from autojev.model import answer  # importable once from_pretrained has added the source dir to sys.path

    return decider, answer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", required=True, help="CSV with row_id,text")
    parser.add_argument("--out", default=None, help="output CSV (default: pplx_predictions.csv next to inputs)")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0, help="only the first N messages (smoke test)")
    args = parser.parse_args()

    common = load_common(Path(__file__).resolve().parent)
    questions = common.build_questions()

    inputs = pd.read_csv(args.inputs)
    if args.limit:
        inputs = inputs.head(args.limit)
    print(f"{len(inputs)} messages x {len(questions)} questions on {args.device}", flush=True)
    if args.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    start = time.perf_counter()
    decider, answer = load_decider(args.model, args.revision, args.device)
    print(f"model loaded in {time.perf_counter() - start:.0f}s", flush=True)

    names = list(questions)
    triage_names = ["emergency", "condition"]  # what production TriageAgent asks
    rows = []
    for i, (row_id, text) in enumerate(zip(inputs["row_id"], inputs["text"])):
        # Production-like timing: the two triage questions in one batch
        if args.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        decider.model.predict([{"state": text, "question": questions[n]} for n in triage_names], batch_size=2)
        if args.device == "cuda":
            torch.cuda.synchronize()
        triage_ms = (time.perf_counter() - t0) * 1000

        probs = decider.model.predict([{"state": text, "question": questions[n]} for n in names], batch_size=len(names))
        answers = {n: answer(questions[n], p) for n, p in zip(names, probs)}

        condition = answers["condition"]
        out = {
            "row_id": row_id,
            "pplx_label": condition["choice"],
            "pplx_conf": condition["probabilities"][condition["choice"]],
            "pplx_rescaled_conf": condition["confidence"],
            "pplx_emergency": answers["emergency"]["noul"],
            "pplx_ms": triage_ms,
        }
        for c in common.CONDITIONS:
            out[f"pplx_has::{c}"] = answers[f"has::{c}"]["noul"]
        rows.append(out)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(inputs)} done, {time.perf_counter() - start:.0f}s elapsed", flush=True)

    result = pd.DataFrame(rows)
    out_path = Path(args.out) if args.out else Path(args.inputs).with_name("pplx_predictions.csv")
    try:
        result.to_csv(out_path, index=False)
        print(f"wrote {out_path}")
    except OSError as e:  # read-only mount on a cloud job; stdout copy below still has everything
        print(f"could not write {out_path} ({e}); use the CSV printed below")

    buffer = io.StringIO()
    result.to_csv(buffer, index=False)
    print(CSV_BEGIN)
    print(buffer.getvalue().strip())
    print(CSV_END, flush=True)
    print(f"total {time.perf_counter() - start:.0f}s, median triage latency {result['pplx_ms'].median():.0f} ms")


if __name__ == "__main__":
    main()
