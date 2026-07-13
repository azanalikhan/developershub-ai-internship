# Task 5: Auto Tagging Support Tickets Using LLM

## Objective
Automatically tag free-text support tickets with the top-3 most probable
category tags using an LLM, comparing **zero-shot** vs **few-shot** prompting.

---

## Files in this folder — what each one does, and the order to use them

| Order | File | What it does | Run directly? |
|---|---|---|---|
| 1 | `requirements.txt` | anthropic, groq, and openai SDKs + pandas, scikit-learn | Once, before anything else |
| 2 | `data/support_tickets.csv` | 10 sample support tickets with human-labeled ground-truth tags | N/A — just data |
| 3 | `tagging.py` | Tags every ticket with both zero-shot and few-shot prompting, saves `models/predictions.csv` | Yes — run first |
| 4 | `evaluate.py` | Scores both strategies against `true_tags`, saves `models/comparison_metrics.json` | Yes — run after tagging |

**Run order: `tagging.py` -> `evaluate.py`**

---

## Step-by-step: how to actually run it

### 0. Install dependencies (once)
```powershell
python -m pip install -r requirements.txt
```

### 1. Set your LLM backend and API key
Supports three backends via `LLM_BACKEND`:

| Backend | `LLM_BACKEND` value | API key env var | Model used |
|---|---|---|---|
| Groq | `groq` | `GROQ_API_KEY` | `llama-3.3-70b-versatile` |
| Anthropic | `anthropic` (default) | `ANTHROPIC_API_KEY` | `claude-sonnet-4-6` |
| OpenAI | `openai` | `OPENAI_API_KEY` | `gpt-4o-mini` |

```powershell
$env:LLM_BACKEND = "groq"
$env:GROQ_API_KEY = "gsk_..."
```

### 2. Tag all tickets
```powershell
python tagging.py
```
Tags all 10 sample tickets with both zero-shot and few-shot prompting, saves
`models/predictions.csv`, and prints a side-by-side comparison table.

### 3. Score both strategies
```powershell
python evaluate.py
```
Compares `zero_shot_tags` and `few_shot_tags` against `true_tags` using
precision@3, recall, and top-1 match rate. Saves `models/comparison_metrics.json`.

### 4. Tag a single ad-hoc ticket (optional)
```powershell
python tagging.py --single "My invoice has the wrong tax amount for this month."
```

---

## Methodology / Approach
1. **Dataset** — `data/support_tickets.csv`: 10 free-text support tickets
   with human-labeled ground-truth tags.
2. **Tag taxonomy** — a fixed 15-tag label set (`TAG_TAXONOMY` in
   `tagging.py`: Billing, Bug, Login Issue, Feature Request, etc.), so the
   LLM always chooses from a controlled vocabulary instead of inventing
   arbitrary tags.
3. **Zero-shot prompting** — ticket text + tag taxonomy sent to the LLM with
   instructions to return the top 3 most probable tags, ranked, no labeled
   examples included.
4. **Few-shot prompting** — the same prompt, preceded by 3 hand-labeled
   example tickets, to steer the model toward the taxonomy's labeling
   convention.
5. **Structured output** — the model responds with strict JSON
   (`{"tags": [...]}`), parsed and validated against the taxonomy — any tag
   not in the fixed list gets silently dropped.
6. **Comparison** — `evaluate.py` scores both strategies using
   **precision@3**, **recall**, and **top-1 match rate**.

## Key Results / Observations

Final comparison, averaged across all 10 tickets:

| Metric | Zero-shot | Few-shot |
|---|---|---|
| Precision@3 | 0.700 | **0.733** |
| Recall | 0.933 | **0.967** |
| Top-1 match rate | 1.000 | 1.000 |

Both strategies got the single most-confident tag right on every ticket
(top-1 match rate = 1.0). Few-shot's advantage shows up specifically in the
**2nd and 3rd ranked tags** — the harder, more ambiguous calls — where it
edges out zero-shot on both precision and recall.

Both strategies are also **generous rather than stingy**: recall (0.93-0.97)
is consistently higher than precision (0.70-0.73), meaning the model rarely
misses a true tag, but tends to add one extra plausible-but-not-labeled tag
to round out its top 3 (e.g. adding "Security" to a login-issue ticket, or
"Performance" to a slow-loading dashboard ticket that was only labeled
"Bug"). This is arguably reasonable over-tagging rather than a real error —
several of these "extra" tags are defensible reads of the ticket that just
weren't in the original 2-tag ground truth.

**The clearest example of few-shot actually correcting a zero-shot mistake**
— ticket 10:
> *"Getting a 'permission denied' error when trying to invite teammates to my workspace."*
> - True tags: `Bug, Account, Team Management`
> - Zero-shot predicted: `Team Management, Security, Login Issue` — only
>   1 of 3 correct. It misread this as a login/security problem.
> - Few-shot predicted: `Team Management, Bug, Security` — 2 of 3 correct,
>   correctly identifying it as a **Bug** rather than a login issue.

This is a case where the labeled examples in the few-shot prompt (which
included a similarly-worded "keeps getting logged out... Bug" example)
appear to have steered the model toward recognizing "permission denied
during an action" as a bug pattern, rather than defaulting to the more
surface-level "permission" -> "Login Issue"/"Security" association zero-shot
made.

Two tickets (5 and 6) were tagged with the **exact correct tag set** by both
strategies, showing that for tickets with unambiguous, clearly-stated
categories (e.g. explicit "API... 500 errors... since this morning" or
explicit "cancel my subscription... refund"), zero-shot alone is already
sufficient — the extra few-shot examples mainly help on ambiguous edge cases
like ticket 10, not on clear-cut ones.

## Files
- `tagging.py` — zero-shot + few-shot tagging via Groq/Anthropic/OpenAI, CLI + batch modes
- `evaluate.py` — scores predictions against ground truth, compares strategies
- `data/support_tickets.csv` — sample labeled ticket dataset
- `models/` — saved predictions + comparison metrics (generated after running)
