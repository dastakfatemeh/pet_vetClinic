# Jev vs. VetBERTDx: Classifier Comparison

This experiment checks whether [Jev](https://docs.typesafe.ai) (TypeSafe's System One model) improves on the production VetBERTDx classifier. Jev returns typed judgments with probabilities, so we can compare the two directly on classification and also use Jev for checks the classifier can't do: spotting emergencies and messages outside the five supported conditions.

**Short answer:** Jev doesn't replace VetBERTDx on accuracy. It is a strong **safety layer** in front of it. That finding led to the triage checks now running in `/converse` (see [Outcome](#outcome)).

## 📚 Table of Contents
- [Setup](#setup)
- [Running](#running)
- [Method](#method)
- [Results](#results)
- [Caveats](#caveats)
- [Outcome](#outcome)
- [Next Steps](#next-steps)

## 🔧 Setup

1. Python **3.10+** (`typesafe-sdk` does not support 3.9).
2. Install dependencies from the project root:
   ```bash
   pip install -r requirements.txt
   pip install -r experiments/jev/requirements.txt
   ```
3. Add both keys to `.env` at the project root:
   ```
   HUGGING_FACE_TOKEN=hf_...      # a "Read" token is enough; all models are public
   TYPESAFE_API_KEY=...
   ```

Qdrant is **not** needed. The experiment only compares the classification step.

## 🚀 Running

From the project root:

```bash
# All three evaluation sets (val, clinical sample of 200, hard cases)
python experiments/jev/compare_classifiers.py

# Selected sets / full clinical notes
python experiments/jev/compare_classifiers.py --sets val hard
python experiments/jev/compare_classifiers.py --sets clinical --clinical-sample 0
```

The script prints the metric tables and writes one row per message, with both models' outputs, to `experiments/jev/results/comparison_<timestamp>.csv`. The `results/` folder is git-ignored.

| File | Purpose |
|---|---|
| `compare_classifiers.py` | Runs both models and reports metrics |
| `hard_cases.csv` | 45 hand-written hard cases with expected label and emergency flag |
| `requirements.txt` | Extra dependencies for the experiment |

## 🔬 Method

### Models
| | VetBERTDx (production) | Jev (zero-shot) |
|---|---|---|
| How it runs | `fdastak/model_classification` through the real `ClassificationAgent`, one message at a time, on local CPU | `jev-latest` through `typesafe-sdk`, 8 concurrent requests |
| Training on this data | Fine-tuned on the owner observations | None; uses short generic descriptions of each condition |
| Output | Softmax over 5 conditions | Choice over 5 conditions **+ "none of these"**, an emergency yes/no probability, and a yes/no probability for each condition |

The condition descriptions were written from general veterinary knowledge, not copied from the dataset, so test text doesn't leak into the prompts.

### Evaluation sets
| Set | n | Description |
|---|---|---|
| `val` | 200 | The notebook's held-out split, rebuilt exactly: owner observations only, lowercased, stratified 20%, `random_state=42` |
| `clinical` | 200 | Clinical notes (40 per condition), which the classifier never saw. Tests a different writing style |
| `hard` | 45 | Hand-written: emergencies (12), out-of-scope (10), multi-condition (5), slang/typos (6), negation (6), non-English (2), long context (4) |

### Production gate
Both models go through the same gate as `/converse`: a prediction with confidence below **0.8** (or Jev choosing "none of these") becomes `uncertain`, and the owner is offered an appointment.

## 📊 Results

Runs from 2026-10-25, `jev-1.14.0`.

### Validation split (in-distribution)
| Model | Accuracy | Macro F1 | Calibration error (lower is better) | Accepted at 0.8 gate | Accuracy when accepted | Median latency |
|---|---|---|---|---|---|---|
| VetBERTDx | **0.890** | **0.889** | **0.068** | 93.5% | 0.909 | 24 ms |
| Jev | 0.810 | 0.830 | 0.118 | 84.5% | 0.905 | 104 ms |

- VetBERTDx wins here, as expected, because it learned this dataset's labelling conventions.
- 12 of Jev's 38 misses were "none of these" picks, often reasonable ones. For example, *"dog ate grapes, now vomiting"* is a poisoning, and *"facial nerve paralysis"* is neurological.
- Several "errors" are ambiguous labels. *"Dark, crumbly stuff in her ears"* is labelled `parasites` (ear mites), and Jev says `ear infections`. Both are defensible.
- The two models agreed on 82% of messages, and when they agreed they were right **93.3%** of the time.

### Clinical notes (different writing style)
| Model | Accuracy | Macro F1 | Calibration error | Accepted at 0.8 gate | Accuracy when accepted |
|---|---|---|---|---|---|
| VetBERTDx | 0.770 | 0.772 | 0.113 | 74.0% | 0.872 |
| Jev | **0.805** | **0.870** | 0.121 | 76.0% | **0.961** |

Once the text looks different from the training data, VetBERTDx drops and Jev pulls ahead. Accuracy when both models agreed: **96.6%**.

### Hard cases
| Category | VetBERTDx correct / abstained / wrong | Jev correct / abstained / wrong |
|---|---|---|
| Emergency (12) | 7 / 1 / 4 | **11 / 1 / 0** |
| Out-of-scope (10) | 5 / 0 / 5 | **10 / 0 / 0** |
| Multi-condition (5) | 5 / 0 / 0 | 3 / 2 / 0 |
| Slang / typos (6) | 4 / 1 / 1 | **6 / 0 / 0** |
| Negation / mixed (6) | 5 / 0 / 1 | **6 / 0 / 0** |
| Non-English (2) | 0 / 0 / 2 | **2 / 0 / 0** |
| Long context (4) | 4 / 0 / 0 | 4 / 0 / 0 |

| Safety metric | VetBERTDx | Jev |
|---|---|---|
| Emergencies flagged | 0 / 12 (no urgency signal exists) | **12 / 12** |
| False emergency alarms among the 33 other hard cases | n/a | **0 / 33** |
| Labels found in multi-condition cases | 47% (top-1 only) | **100%** (per-condition checks) |

**VetBERTDx's most confident mistakes**, all of which pass the 0.8 gate:

| Message | VetBERTDx | Jev |
|---|---|---|
| "Cat has been straining in the litter box for hours and nothing is coming out" | digestive issues (1.00) | none of these, emergency 0.97 |
| "Drinking much more water than usual and peeing all the time" | digestive issues (1.00) | none of these |
| "My cat ate part of a lily from a bouquet and is drooling" | digestive issues (0.99) | none of these, emergency 0.97 |
| "Vet ruled out fleas, yet my dog still has itchy red bumps" | parasites (0.98) | skin irritations |
| "Mon chat vomit tous les matins depuis une semaine" | parasites (0.98) | digestive issues |
| "My dog ate a whole bar of dark chocolate" | digestive issues (0.97) | none of these, emergency 0.94 |

### Emergency flags on routine messages
Jev flagged 13% of validation messages and 8% of clinical notes as emergencies. On review most of these look **genuinely urgent**: unproductive retching (possible bloat), *"rabbit stopped eating hay and seems lethargic"* (gut stasis), *"dog ate grapes"*, *"rabbit has a broken leg"*. A veterinarian should confirm this.

### Cost and latency
- Jev: ~1,100 tokens per message with the 7 questions used here. The full 445-message evaluation used ~508k tokens.
- Production triage asks only 2 questions (emergency + condition), so it costs much less.
- Median latency: 104 ms for Jev (network) vs. 24 ms for VetBERTDx (local CPU). In `/converse` the two run at the same time.

### Choosing the escalation rule
Using the saved results, we compared rules for when to override the classifier. Coverage is the share of messages answered rather than escalated; accuracy is VetBERTDx's accuracy on those answered messages.

| Escalate when... | Val coverage | Val accuracy | Clinical coverage | Clinical accuracy |
|---|---|---|---|---|
| Jev says "none of these" | 88.0% | 0.909 | 68.5% | 0.912 |
| Any disagreement | 79.0% | 0.943 | 65.0% | **0.962** |
| **"None of these", or Jev ≥ 0.8 confident in a different condition** | **82.5%** | **0.945** | **66.0%** | 0.947 |

We chose the last rule: it matches "any disagreement" on accuracy for validation while answering more messages.

## ⚠️ Caveats

- **The hard cases favour Jev somewhat.** The same person wrote the hard cases and Jev's condition descriptions, and the "none of these" description names urinary problems and poisoning. Treat hard-case results as directional.
- **Small sample.** 45 hard cases is enough to show the pattern, not to fix exact rates.
- **Labels are unreviewed.** Hard-case labels and emergency flags need review by a veterinarian. Some labels are debatable: we labelled "straining in the litter box" as out-of-scope, though constipation would be `digestive issues`.
- **Dataset quality.** The 2,000 rows look synthetic and contain ambiguous labels, plus one stray `Medical Guidence` row. VetBERTDx's validation score reflects agreement with these labels, not clinical correctness.
- **Thresholds** (0.8 gate, 0.5 emergency) were not tuned on real traffic.
- **Typed output isn't proof of correctness.** Jev always returns a valid label, but that label still needs validation in this domain.

## ✅ Outcome

Based on these findings, `/converse` now runs a **triage layer** (`Agents/TriageAgent.py`) at the same time as VetBERTDx:

| Order | Check | Action |
|---|---|---|
| 1 | Jev emergency probability ≥ 0.5 | Urgent-care message, skip retrieval |
| 2 | Jev picks "none of these" | Offer appointment (`out_of_scope`) |
| 3 | Jev ≥ 0.8 confident in a different condition | Offer appointment (`model_disagreement`) |
| 4 | VetBERTDx confidence < 0.8 | Offer appointment (`low_confidence`, unchanged) |

VetBERTDx stays the primary classifier. If TypeSafe is unavailable (no key, SDK missing, error or timeout), the pipeline logs a warning and behaves as before.

## 🔭 Next Steps

- [ ] Have a veterinarian review the hard-case labels, the emergency flags and the urgent-care wording
- [ ] Have someone else write a fresh hard-case set, to remove the author bias
- [ ] Log triage scores in production and tune the thresholds on real traffic
- [ ] Alert when triage falls back (an outage silently disables the emergency check)
- [ ] Use Jev to rerank retrieved cases instead of the fixed 0.8 cosine threshold. The current hard-negative mining means at most one case passes that threshold
- [ ] Use Jev to audit dataset labels and flag rows where it strongly disagrees
