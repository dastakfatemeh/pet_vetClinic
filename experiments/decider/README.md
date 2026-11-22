# Jev vs. open-weight deciders: Triage Comparison

Jev ([TypeSafe](https://docs.typesafe.ai)) is the triage model in `/converse` (see [the Jev experiment](../jev/README.md)), and it is a paid API. This experiment checks whether a free, open-weight decision model can replace it. Three candidates have been tested, all Apache-2.0 and built on Qwen:

| Model | Size | Base | Notes |
|---|---|---|---|
| [`perplexity-ai/pplx-decider-v1-27b`](https://huggingface.co/perplexity-ai/pplx-decider-v1-27b) | 27B | Qwen3.8-27B | One question per forward pass |
| [`Cloudflare/clef`](https://huggingface.co/Cloudflare/clef) | 27B | Qwen3.8-27B | Jev-compatible request format; also hosted on Workers AI ([blog](https://blog.cloudflare.com/clef-decision-models/)) |
| [`Cloudflare/clef-flash`](https://huggingface.co/Cloudflare/clef-flash) | 9.4B | Qwen3.5-9B | Same as Clef, small enough for a 24 GB GPU |

All three answer the same question types as Jev (`choice` with probabilities, and `noul` yes/no probabilities), so each one gets the Jev experiment's 7 questions and can replace Jev inside `Agents/TriageAgent.py`.

## Short answer

**Clef (27B) is the best free replacement for Jev.**
- **Safety:** it matches Jev, with 12/12 emergencies flagged and 10/10 out-of-scope messages sent to an appointment.
- **Classification:** it is the most accurate decision model on both labelled sets (0.845 val, 0.835 clinical) and has the best-calibrated probabilities of any model tested, VetBERTDx included.
- **Checking VetBERTDx:** with a 0.5 disagreement threshold it matches Jev exactly on validation (9 wrong answers shown, 82% answered).
- **Hosting:** it's on Workers AI at $0.24 per million input tokens, so no GPU is needed. We verified the hosted endpoint live on 2026-11-22: it matches the open weights on 97% of labels and on every hard-case emergency ([Workers AI verification](#workers-ai-verification)).

**pplx-decider** is as safe, but weaker at checking VetBERTDx and slower. **Clef-flash** is the cheapest option, but it over-flags emergencies at the 0.5 threshold. It needs its emergency threshold raised to 0.8, and it lets more of VetBERTDx's mistakes through.

All thresholds marked @0.5 or @0.8 below were picked after looking at these same messages. They need checking on fresh data before production.

## Results

Every model saw the same 445 messages (val 200, clinical 200, hard 45). Jev and VetBERTDx outputs come from the saved Jev run, and the open models ran on HF Jobs `a100-large` (A100-SXM4-80GB):

| Run | Job | GPU time | Cost |
|---|---|---|---|
| pplx-decider | `6afa940029dd05ff0369c0d7` | 703 s | ~$0.49 |
| Clef + Clef-flash | `6afa9c0329dd05ff0369c8d1` | 896 s | ~$0.62 |
| Smoke tests (2) | | 254 s | ~$0.18 |

### As a classifier (context; VetBERTDx stays the classifier in production)
| Set | Model | Accuracy | Macro F1 | Calibration error ↓ | Accepted at 0.8 | Accuracy when accepted |
|---|---|---|---|---|---|---|
| val | VetBERTDx | **0.890** | **0.889** | 0.068 | 93.5% | 0.909 |
| val | Jev | 0.810 | 0.830 | 0.118 | 84.5% | 0.905 |
| val | pplx-decider | 0.790 | 0.823 | 0.112 | 75.0% | 0.887 |
| val | Clef | 0.845 | 0.848 | **0.035** | 68.5% | 0.905 |
| val | Clef-flash | 0.845 | 0.851 | 0.061 | 62.0% | 0.895 |
| clinical | VetBERTDx | 0.770 | 0.772 | 0.113 | 74.0% | 0.872 |
| clinical | Jev | 0.805 | 0.870 | 0.121 | 76.0% | 0.961 |
| clinical | pplx-decider | 0.755 | 0.842 | 0.131 | 67.0% | 0.985 |
| clinical | Clef | **0.835** | **0.880** | **0.045** | 69.5% | **0.986** |
| clinical | Clef-flash | 0.820 | 0.862 | 0.048 | 64.0% | 0.969 |

The Clef models give lower, better-calibrated probabilities. Their stated confidence matches how often they are right, which is why fewer of their answers clear the fixed 0.8 bar. Label agreement with Jev: pplx 93%, Clef 92–93%, Clef-flash 89%.

### Safety checks (hard cases)
| Check | Jev | pplx-decider | Clef | Clef-flash |
|---|---|---|---|---|
| Emergencies flagged (≥ 0.5) | 12 / 12 | 12 / 12 | 12 / 12 | 12 / 12 |
| False emergency alarms (≥ 0.5) | 0 / 33 | 1 / 33 | 1 / 33 | 4 / 33 |
| Emergency flag rate on routine val messages | 13% | 12% | 14% | **27%** |
| Out-of-scope sent to an appointment | 10 / 10 | 10 / 10 | 10 / 10 | 10 / 10 |
| Non-English, negation, long context | all correct | all correct | all correct | all correct |
| Slang / typos correct | 6 / 6 | 5 / 6 | 5 / 6 | 3 / 6 (3 abstained) |
| Multi-condition labels found (per-condition yes/no) | 100% | 100% | 93% | 93% |

**Emergency threshold.** Every model scored every real emergency at least 0.86 (Jev 0.92, pplx 0.91, Clef 0.95, Clef-flash 0.86), so the threshold can rise without missing any of them:

| Emergency threshold | Jev | pplx-decider | Clef | Clef-flash |
|---|---|---|---|---|
| 0.5 (current) | 12/12, 0 FA, val 13% | 12/12, 1 FA, val 12% | 12/12, 1 FA, val 14% | 12/12, 4 FA, val 27% |
| 0.7 | 12/12, 0 FA, val 6% | 12/12, 0 FA, val 6% | 12/12, 0 FA, val 8% | 12/12, 1 FA, val 16% |
| 0.8 | 12/12, 0 FA, val 4% | 12/12, 0 FA, val 4% | 12/12, 0 FA, val 6% | 12/12, 0 FA, val 10% |

FA means false alarms out of 33 non-emergency hard cases. The hard-case emergencies are clear-cut, so borderline real emergencies could score lower. A higher threshold needs a veterinarian's review before it ships.

**Notable disagreements with Jev**
- *"Dog was stung multiple times and now his face is swelling up"*: pplx and Clef give the expected label, `skin irritations`, where Jev said "none of these". All models flag the emergency.
- *"Cat has been straining in the litter box for hours"*: Clef-flash says `digestive issues` (0.81), while the others say "none of these". The emergency is still flagged (0.95). The Jev README already notes that constipation would make `digestive issues` defensible.
- *"Cloudy film over one of my dog's eyes"* and *"hamster's teeth look really long"*: Clef-flash raised false emergency alarms (0.59–0.60). Both cases are still correctly out of scope.

### VetBERTDx behind each triage model (the deciding table)
Rules 2–4 of the production gate: escalate when the triage model says "none of these", or when it is at least X confident in a different condition than VetBERTDx, or when VetBERTDx is below 0.8.

| Triage model | Val answered | Val accuracy | Val wrong shown | Clinical answered | Clinical accuracy | Clinical wrong shown |
|---|---|---|---|---|---|---|
| None (v1) | 93.5% | 0.909 | 17 | 74.0% | 0.872 | 19 |
| **Jev @0.8** | 82.5% | **0.945** | **9** | 66.0% | 0.947 | 7 |
| pplx @0.8 | 81.5% | 0.914 | 14 | 63.0% | 0.944 | 7 |
| pplx @0.5 | 78.5% | 0.936 | 10 | 61.5% | **0.967** | **4** |
| Clef @0.8 | 88.0% | 0.926 | 13 | 67.0% | 0.948 | 7 |
| **Clef @0.5** | 82.0% | **0.945** | **9** | 65.0% | 0.954 | 6 |
| Clef-flash @0.8 | 90.0% | 0.911 | 16 | 69.0% | 0.920 | 11 |
| Clef-flash @0.5 | 85.5% | 0.918 | 14 | 66.5% | 0.940 | 8 |

Clef at 0.5 matches Jev on validation and is slightly better on clinical notes. The open models need the lower threshold because their probabilities are lower and better calibrated. Their disagreements with VetBERTDx are usually right, but often stated at 0.5–0.75 rather than above 0.8.

### Cost and latency
| | Jev | pplx-decider | Clef | Clef-flash |
|---|---|---|---|---|
| Self-hosted GPU | n/a | 80 GB (49 GiB weights) | 80 GB (51.6 GiB peak) | **24 GB (18.0 GiB peak)** |
| Hosted API | TypeSafe, per token | none | Workers AI, $0.24 per M input tokens | Workers AI, $0.038 per M input tokens |
| Tokens per message (7 questions) | ~1,140 | n/a | ~1,190 | ~1,190 |
| This evaluation on the hosted API | ~508k tokens, already paid | n/a | ~$0.13 | ~$0.02 |
| Median triage latency (2 questions), our A100 | ~104 ms (network) | 340 ms | 478 ms | 241 ms |
| Median latency on Workers AI, 7 questions, measured from a home connection | n/a | n/a | **202 ms** | **148 ms** |

Workers AI prices come from Cloudflare's [pricing page](https://developers.cloudflare.com/workers-ai/platform/pricing/), checked 2026-11-16, with no output-token charge for Clef. The free tier is 10,000 neurons a day. At 21,818 neurons per million Clef input tokens, that is about 458k tokens a day, or roughly 380 messages a day with all 7 questions. Production asks only 2 questions, so it fits more. The live check measured ~1,190 input tokens per message with 7 questions, the same as the open weights.

Our A100 latencies are pessimistic. transformers fell back to slow reference code because `flash-linear-attention` and `causal-conv1d` were not installed. The hosted latencies above are real round trips, including the network.

### Workers AI verification
Everything above was measured with the open weights on HF Jobs. Since `/converse` defaults to Clef on Workers AI, we checked the hosted endpoint live on 2026-11-22 with `verify_workers_ai.py`. It sent the same 7 questions for 95 messages (all 45 hard cases plus 25 validation and 25 clinical, seed 42) and compared the answers message by message.

| Check | Clef on Workers AI | Clef-flash on Workers AI |
|---|---|---|
| App's own `TriageAgent` works end to end | ✅ 135–522 ms per call | ✅ 127–327 ms per call |
| Same label as the open weights | 96.8% | 97.9% |
| Emergency probability difference, mean / max | 0.018 / 0.188 | 0.037 / 0.249 |
| Same emergency flag (at the model's threshold) | 98.9% (at 0.5) | 97.9% (at 0.8) |
| Hard-case emergencies flagged | 12 / 12 (open weights: 12 / 12) | 12 / 12 (open weights: 12 / 12) |
| Hard-case false alarms | 1 / 33 (open weights: 1 / 33) | 0 / 33 (open weights: 0 / 33) |
| Accuracy on the 50 labelled messages | 0.78 (open weights: 0.80) | 0.82 (open weights: 0.82) |
| Response format | `{"result": {model, answers, usage}, "success", "errors", "messages"}`, which `SystemOneHTTPBackend` handles | Same |
| Input tokens for 95 messages | 113,364 | 113,364 |

Every label difference was on a message where the open weights were already unsure (top probability 0.39–0.60). Examples:
- *"Old dog is stiff getting up in the morning and also has an itchy..."*: mobility 0.52 vs skin 0.53
- *"Cat has ear mites and has scratched the skin behind her ears"*: parasites 0.50 vs ear infections 0.51
- *"he's got these lil black specks in his fur n wont stop scratching"*: parasites 0.60 on the open weights vs skin irritations 0.79 hosted. This is the one where the hosted version gave a worse answer with more confidence.

Small differences like these are expected from different serving hardware and numeric precision. None of them changed a safety outcome on the hard cases. The whole check used about 280k input tokens in one day (including a first hard-cases-only run of Clef), inside the free daily allowance.

## Verdict
- **Replace Jev with Clef** if the goal is to drop TypeSafe. Call it on Workers AI with a 0.5 disagreement threshold, and consider an emergency threshold of 0.7–0.8 after veterinary review. It matched or beat Jev on everything measured here.
- **Clef-flash** is the budget option, about 6× cheaper on Workers AI and able to run on a 24 GB GPU. It needs an emergency threshold of 0.8, and it shows 14 wrong validation answers against Jev's 9.
- **pplx-decider** works, but it has no hosted API, needs an 80 GB GPU, and checks VetBERTDx less well than Clef.

## Next steps
- [x] Check the Workers AI Clef endpoint on a sample, to confirm hosted probabilities match the open weights and to measure real latency (2026-11-22, see above)
- [x] Make `TriageAgent` backend-agnostic (TypeSafe, Workers AI or self-hosted) behind the same `TriageResult`. Chosen with `TRIAGE_MODEL`, Clef by default
- [ ] Validate the 0.5 disagreement and 0.7–0.8 emergency thresholds on fresh messages, written by someone other than the author of the hard cases
- [ ] Test the self-hosted path (`TRIAGE_PROVIDER=self_hosted`, SGLang `/v1/systemone`). It is implemented but has not been run
- [ ] Have a veterinarian review the emergency threshold and the cases the models disagree on
- [ ] Re-measure self-hosted latency with `flash-linear-attention` and `causal-conv1d` installed

## How it works

1. `compare_deciders.py export` collects the 445 messages from the saved Jev run into `inputs/texts.csv`.
2. A GPU script runs each open model on those messages and prints its predictions as CSV in the job log:
   - `run_pplx_decider.py` uses the model repo's own `inference.py`, so prompting and calibration match the release.
   - `run_clef.py` uses the model repo's own `joint_schema_model.py`, and runs several Clef models one after the other in one job.
3. `compare_deciders.py report` merges the predictions with the saved VetBERTDx and Jev outputs and prints every table above.

| File | Purpose |
|---|---|
| `common.py` | The condition descriptions and 7 questions, in both formats |
| `run_pplx_decider.py` | pplx-decider inference (self-contained `uv` script, Python 3.12) |
| `run_clef.py` | Clef and Clef-flash inference (self-contained `uv` script, pinned to the card's tested torch 2.11 and transformers 5.10.2) |
| `compare_deciders.py` | Export inputs and score results locally. No API calls |
| `verify_workers_ai.py` | Live check of Clef and Clef-flash on Cloudflare Workers AI against the HF Jobs results |

**Question wording:**
- Clef gets the Jev questions verbatim, with state `{"owner_message": text}`.
- pplx-decider takes the message as plain text after `State:`, so its questions say "the message in State" instead of `` `owner_message` ``.
- The condition descriptions are identical for all models.

## Running

From the project root:

```bash
python experiments/decider/compare_deciders.py export

# pplx-decider (80 GB GPU)
hf jobs uv run --detach --flavor a100-large --timeout 60m -v ./experiments/decider:/work \
    experiments/decider/run_pplx_decider.py --inputs /work/inputs/texts.csv

# Clef-flash then Clef in one job (Clef-flash alone fits on l4x1 / a10g-large)
hf jobs uv run --detach --flavor a100-large --timeout 45m -v ./experiments/decider:/work \
    experiments/decider/run_clef.py --inputs /work/inputs/texts.csv \
    --model Cloudflare/clef-flash --model Cloudflare/clef

# The mount is read-only, so save each log; the report reads the CSVs out of them
PYTHONIOENCODING=utf-8 hf jobs logs <pplx-job-id> > experiments/decider/inputs/pplx_job.log
PYTHONIOENCODING=utf-8 hf jobs logs <clef-job-id> > experiments/decider/inputs/clef_job.log

python experiments/decider/compare_deciders.py report \
    --pplx experiments/decider/inputs/pplx_job.log \
    --pred clefflash=experiments/decider/inputs/clef_job.log \
    --pred clef=experiments/decider/inputs/clef_job.log
```

- Add `--limit 5` to a GPU script for a smoke test first.
- **Git Bash on Windows:** prefix `hf jobs` commands with `MSYS_NO_PATHCONV=1`, or `/work` gets rewritten to a Windows path.
- **No token needed:** all three models are public. The "unauthenticated requests" warning is harmless, and downloads ran at about 2 GB/s.
- **Bucket:** the local folder is synced to the `jobs-artifacts` bucket in your HF account.
- **Without HF Jobs:** on any machine with a big enough GPU, run the same scripts with `uv run` and the local `--inputs` path.

### Workers AI check
Needs `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN` in `.env`. Create the token with Cloudflare's "Workers AI" template, which grants only Workers AI Read and Edit. It also needs the saved inputs and `clef_job.log` from the steps above.

```bash
python experiments/decider/verify_workers_ai.py                         # Clef + Clef-flash, 95 messages each
python experiments/decider/verify_workers_ai.py --models clef --sample 0   # hard cases only
```

The default run uses about 115k input tokens per model, inside the free daily allowance. The token and account ID are never printed. A redacted sample response is saved to `results/`.

## Caveats
- Same author bias as the Jev experiment: one person wrote the hard cases and the condition descriptions.
- 45 hard cases and 12 emergencies show the pattern but can't pin exact rates.
- Thresholds marked @0.5 or 0.7–0.8 were chosen after seeing these results.
- The model cards' own benchmarks (Jev Decision Index and others) are general-purpose and say nothing specific about veterinary triage.
- Newer versions exist or may appear (`pplx-decider-v1.1-27b`). Pass `--model` to test them with the same scripts.

## Adding another decision model
Write a runner that prints `row_id`, `<name>_label`, `<name>_conf`, `<name>_emergency`, `<name>_ms` and `<name>_has::<condition>` for the exported messages (optionally `<name>_tokens`), between `=====<NAME>_CSV_BEGIN=====` and `=====<NAME>_CSV_END=====` markers. Then pass `--pred <name>=<log or csv>` to `report`.
