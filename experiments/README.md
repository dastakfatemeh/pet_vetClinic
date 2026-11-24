# Experiments: Adding a Decision-Model Triage Layer

This folder holds the experiments behind one idea: **put a decision model next to the fine-tuned classifier, so `/converse` can recognise messages the classifier can't handle safely.** These are emergencies, conditions outside the five categories, and confident mistakes.

The classifier stays the primary model. The decision model only decides when the classifier's answer should *not* be shown to the owner.

| Folder | Question it answers |
|---|---|
| [`jev/`](./jev/README.md) | Does a decision model (Jev, TypeSafe's paid API) add anything to the fine-tuned classifier? |
| [`decider/`](./decider/README.md) | Can a free or cheaper decision model (pplx-decider, Clef, Clef-flash, Liquid AI d1) replace Jev? |

The architecture with the new triage layer is drawn in [`overview_diagram.md`](./overview_diagram.md).

## Why a decision model

The classifier is a softmax over five conditions, so it always picks one of them, even for a poisoning, a urinary blockage or a question in French. In the hard-case tests it gave some of its highest confidence scores to messages that needed urgent care:

| Owner message | Classifier says | Decision model says |
|---|---|---|
| "Cat has been straining in the litter box for hours and nothing is coming out" | digestive issues (1.00) | none of these, emergency 0.97 |
| "My cat ate part of a lily from a bouquet and is drooling" | digestive issues (0.99) | none of these, emergency 0.97 |
| "My dog ate a whole bar of dark chocolate" | digestive issues (0.97) | none of these, emergency 0.94 |

A decision model answers typed questions with probabilities: a yes/no probability for "is this an emergency?", and a probability over the five conditions plus "none of these". It needs no training on this dataset. One request gives the two signals the classifier can't produce.

## Models used by the agents

| Agent | Model | Origin | Fine-tuned for this project? |
|---|---|---|---|
| Classification | [`fdastak/model_classification`](https://huggingface.co/fdastak/model_classification) (BERT-base, 108M) | Fine-tuned by the repo owner from [`havocy28/VetBERTDx`](https://huggingface.co/havocy28/VetBERTDx), itself VetBERT trained on 39 veterinary diagnosis classes | **Yes**: 800 owner observations, 5 conditions (`Clean_Classification.ipynb`) |
| Retrieval | [`havocy28/VetBERT`](https://huggingface.co/havocy28/VetBERT) embeddings + Qdrant | havocy28 | No |
| Communication | [`google/flan-t5-base`](https://huggingface.co/google/flan-t5-base) | Google | No |
| **Triage (new)** | [`Cloudflare/clef`](https://huggingface.co/Cloudflare/clef) by default; also Clef-flash, Liquid AI d1 or Jev | Cloudflare / Liquid AI / TypeSafe | No, zero-shot with short condition descriptions |

## Agents: before and after

| | Before (`main`) | After (this contribution) |
|---|---|---|
| Classification Agent | Fine-tuned VetBERTDx | Unchanged |
| Retrieval Agent | VetBERT + Qdrant | Unchanged |
| Communication Agent | Flan-T5-base | Unchanged |
| **Triage Agent** | None | **New**: runs at the same time as the classifier. The model is chosen in `.env` with `TRIAGE_MODEL` (`clef` by default, or `clef-flash`, `d1`, `jev`, `none`) |
| Escalation gate | 1 rule: classifier confidence below 0.8 → offer an appointment | 4 rules, first match wins (below) |
| If triage is unavailable | n/a | Logs a warning and behaves exactly like `main` |

The new escalation gate:

| Order | Check | Response |
|---|---|---|
| 1 | Emergency probability ≥ the model's emergency threshold | Urgent-care message, skip retrieval |
| 2 | Decision model picks "none of these" | Offer an appointment (`out_of_scope`) |
| 3 | Decision model ≥ its disagreement threshold on a different condition | Offer an appointment (`model_disagreement`) |
| 4 | Classifier confidence below 0.8 | Offer an appointment (`low_confidence`, the original rule) |

The files under `Agents/` for classification, retrieval, communication and embeddings are unchanged. The new behaviour lives in `Agents/TriageAgent.py`, `Agents/triage_config.py` and `AI_Agents.py`.

## Results

### The classifier on its own (owner's notebook, validation split, n = 200)
| Condition | Precision | Recall | F1 |
|---|---|---|---|
| digestive issues | 0.97 | 0.93 | 0.95 |
| ear infections | 0.85 | 1.00 | 0.92 |
| mobility problems | 0.93 | 0.93 | 0.93 |
| parasites | 0.84 | 0.80 | 0.82 |
| skin irritations | 0.86 | 0.80 | 0.83 |
| **Accuracy** | | | **0.89** |

We rebuilt the same split and got the same result (accuracy 0.890, macro F1 0.889).

### The whole pipeline, before and after
| Measure | Before: classifier + 0.8 gate | After, with Jev | After, with Clef (default) |
|---|---|---|---|
| Validation: messages answered | 93.5% | 82.5% | 82.0% |
| Validation: accuracy of the answers given | 0.909 | 0.945 | 0.945 |
| Validation: wrong answers shown (of 200) | 17 | 9 | 9 |
| Clinical notes: accuracy of the answers given | 0.872 | 0.947 | 0.954 |
| Clinical notes: wrong answers shown (of 200) | 19 | 7 | 6 |
| Emergencies flagged (hand-written hard cases) | **0 / 12** | 12 / 12 | 12 / 12 |
| Out-of-scope messages handled correctly | 5 / 10 | 10 / 10 | 10 / 10 |
| False emergency alarms (non-emergency hard cases) | n/a | 0 / 33 | 1 / 33 |

The trade-off: about 11% more messages get an appointment offer instead of an answer. In return, owners see roughly half as many wrong answers, and emergencies are no longer answered as routine conditions.

"Clinical notes" are 200 clinical-style notes from the same dataset that the classifier was never trained on. They test a different writing style. The 45 hard cases cover emergencies, out-of-scope questions, several conditions at once, slang and typos, negation, non-English and long messages.

### Choosing the decision model
| | Jev | Clef 27B | Clef-flash 9.4B | Liquid AI d1 | pplx-decider 27B | d1-3B / d1-2B |
|---|---|---|---|---|---|---|
| Emergencies flagged | 12 / 12 | 12 / 12 | 12 / 12 | 12 / 12 | 12 / 12 | 12 / 12 |
| False alarms at the 0.5 emergency threshold | 0 / 33 | 1 / 33 | 4 / 33 | 1 / 33 | 1 / 33 | 1 / 33 |
| Validation: wrong answers shown (best threshold) | 9 | **9** | 14 | 10 | 10 | 10 / 12 |
| Clinical notes: wrong answers shown (best threshold) | 7 | **6** | 8 | 7 | 4 | 4 / 5, but they answer only 52–60% |
| Accuracy as a classifier (val / clinical) | 0.810 / 0.805 | **0.845 / 0.835** | 0.845 / 0.820 | 0.830 / 0.810 | 0.790 / 0.755 | 0.680 / 0.630 and 0.755 / 0.725 |
| Calibration error, val (lower is better) | 0.118 | 0.035 | 0.061 | **0.033** | 0.112 | 0.084 / 0.068 |
| Weights | Closed | Apache-2.0 | Apache-2.0 | Closed | Apache-2.0 | LFM Open License |
| Hosted option | TypeSafe, paid per token | Workers AI, $0.24 per M input tokens | Workers AI, $0.038 per M input tokens | Liquid AI, **$0.02 per M input tokens** | None | None |
| Self-hosting | n/a | 80 GB GPU | 24 GB GPU | n/a | 80 GB GPU | 6 GB / 4 GB GPU |
| Default thresholds (emergency / disagreement) | 0.5 / 0.8 | 0.5 / 0.5 | 0.8 / 0.5 | 0.5 / 0.5 | — | — |

Clef matched Jev on safety and on catching the classifier's mistakes, and it is the best-calibrated model tested. It is free to self-host and cheap to call on Workers AI, so it is the default. The open models were evaluated on Hugging Face Jobs (A100 80GB) for about $1.30 in total. Full numbers and methods are in [`decider/README.md`](./decider/README.md).

Liquid AI's hosted d1, added on 2026-11-23, is about as good as Clef and the cheapest option. It is offered as `TRIAGE_MODEL=d1`, but Clef stays the default because d1's weights are closed. The small open d1-3B and d1-2B fit on a laptop-class GPU, but they are too unsure for this task: they send about half the clinical notes to an appointment, and d1-2B scored one real emergency at only 0.50. The whole d1 evaluation cost about $0.08.

## Observations on the classifier (for the repo owner)

- **The model keeps VetBERTDx's 39-label output layer.** It was trained only on labels 0–4, and `ClassificationAgent` maps those to the five conditions. A top prediction on one of the other 34 outputs returns `"unknown condition"`. Re-initialising the head with `num_labels=5` and a matching `id2label` would remove that case.
- **The Hub config still lists VetBERTDx's label names** ("abscess", "adverse reaction to drug", ...), so the Hugging Face widget and `pipeline()` show those names instead of the five conditions.
- **Two copies exist on the Hub:** `fdastak/model_classification`, which the app loads, and `fdastak/model_calssification`, created automatically from the Trainer's `output_dir` name.
- **The classifier is strong in its own domain** (0.89 on owner observations). It drops to 0.77 on clinical-style notes, where the decision models do better. That is why the triage layer checks it rather than replacing it.

## Caveats

- The dataset looks synthetic, and some labels are ambiguous. Scores measure agreement with its labels, not clinical correctness.
- The hard cases and the condition descriptions were written by the same person, which favours the decision models. 45 hard cases show the pattern but don't pin exact rates.
- The Clef thresholds were chosen on these evaluation messages. They need checking on real traffic.
- A veterinarian has not reviewed the hard-case labels, the emergency flags or the urgent-care wording.
- Clef was evaluated with its open weights on HF Jobs, then checked live on Workers AI on 95 messages. The hosted model agreed on 97% of labels and on every hard-case emergency, but it is not bit-for-bit identical ([details](./decider/README.md#workers-ai-verification)).
- The self-hosted path (`TRIAGE_PROVIDER=self_hosted`) is implemented but has not been run.
- Liquid AI's d1 price ($0.02 per M input tokens) comes from the cost reported in its own API responses. Hosted d1 is closed, so Liquid can change it at any time.

## Running it

### Where Clef runs
Clef 27B needs about 52 GB of GPU memory, so it does **not** run on a laptop. It doesn't have to: by default the app calls it on Cloudflare Workers AI, and the local machine only makes an HTTPS request.

| Option | What you need | Cost | Status |
|---|---|---|---|
| **Clef on Workers AI (default)** | Free Cloudflare account, account ID and a "Workers AI" API token | Free daily allowance, then $0.24 per M input tokens | ✅ Verified live |
| Clef-flash on Workers AI | Same, plus `TRIAGE_MODEL=clef-flash` | About 6× cheaper | ✅ Verified live |
| Clef self-hosted | A rented 80 GB GPU (24 GB for Clef-flash) running SGLang, plus `TRIAGE_PROVIDER=self_hosted` and `CLEF_BASE_URL` | ~$2–3/hour while running | Implemented, not run |
| Liquid AI d1 | `TRIAGE_MODEL=d1` and a Liquid AI API key (`LIQUID_API_KEY`) | $0.02 per M input tokens, about 12× cheaper than Clef | ✅ Verified live |
| Jev | `TRIAGE_MODEL=jev` and a TypeSafe API key (Python ≥ 3.10) | Paid per token | ✅ Used in the Jev experiment |
| No triage | `TRIAGE_MODEL=none` | Free | Same behaviour as `main` |

Triage never blocks the app. If keys are missing, the internet is down, or a call fails or takes longer than 3 s, `/converse` logs a warning and answers with the classifier alone, exactly like `main`. The call runs at the same time as the classifier, so it adds little delay (about 130–250 ms measured on Workers AI).

### Running the whole app locally
Everything except the triage model runs on the local machine, on CPU: about 1 GB of model downloads and 2–3 GB of memory, plus Qdrant in Docker.

1. Start Qdrant: `docker run -p 6333:6333 -d --name qdrant qdrant/qdrant`
2. Load the 1,000 clinical notes into Qdrant by running `Vector_DB.ipynb` once. Without this, `/converse` never finds similar cases.
3. Install dependencies: `pip install -r requirements.txt`
4. Create a free Cloudflare account. Copy the account ID from the dashboard, then create an API token with the **"Workers AI" template** (permissions: Workers AI Read and Edit only; set an expiry).
5. Create `.env` in the project root. It is git-ignored, so never commit it:
   ```plaintext
   HUGGING_FACE_TOKEN=...
   TRIAGE_MODEL=clef                 # clef (default) | clef-flash | d1 | jev | none
   CLOUDFLARE_ACCOUNT_ID=...
   CLOUDFLARE_API_TOKEN=...
   ```
6. Start the API with `python -m uvicorn AI_Agents:app --reload`. The log should show `Triage enabled: model=clef, provider=workers_ai`.
7. Open `http://127.0.0.1:8000/docs` and try `POST /converse` with, for example, `{"user_input": "My dog ate a whole bar of dark chocolate"}`. The response should come back with `condition_identified: "emergency"` and `triage.model: "clef"`.

The app has no user interface; the Swagger page at `/docs` is the way to try it. Messages sent to Workers AI leave the local machine (Cloudflare lists zero data retention for Clef). That is fine for sample messages, but check before using real pet-owner data.

### Reproducing the experiments
- [`jev/README.md`](./jev/README.md) needs a TypeSafe key.
- [`decider/README.md`](./decider/README.md) needs a GPU with 24–80 GB, or Hugging Face Jobs (about $1.40 for every model). The Workers AI check needs only the Cloudflare keys, and hosted d1 needs only a Liquid AI key.
