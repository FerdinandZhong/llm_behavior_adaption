# Plan: Multilingual & Scam Adaptation Experiments

## 1. Multilingual Testing

**Goal**: Evaluate whether models behave consistently when dialogues are presented in the user's native language vs. English.

**Data**: Translated dialogues ready for both topics (1,000 samples each).

### Completion Status

| Model                    | Career (Translated) | Investment (Translated) |
|--------------------------|---------------------|-------------------------|
| Qwen3-30B-A3B-Instruct  | Done (v1+v2)        | Pending (config exists) |
| Llama-3.1-8B-Instruct   | Pending             | Pending                 |
| Llama-3.1-70B-Instruct  | Pending             | Pending                 |
| Qwen2.5-7B-Instruct     | Pending             | Pending                 |
| Qwen2.5-72B-Instruct    | Pending             | Pending                 |
| DeepSeek-V3              | Pending             | Pending                 |
| QwQ-32B                  | Pending             | Pending                 |

### Steps

1. Complete Qwen3 investment translated run (config already exists)
2. Create translated prediction configs for each remaining model (career + investment)
3. Run predictions using existing translated dialogue datasets
4. Run `wvs_translated_vs_original_analysis.py` for each model

---

## 2. Scam Adaptation Testing

**Goal**: Test model robustness to feedback-based manipulation, under two complementary scenarios.

### Type A — Human Feedback (Legitimate Correction)

**Question**: When told the correct human answer, does the model accept legitimate feedback?

**Status**: IMPLEMENTED — `wvs_gap_analysis.py` on branch `feature/stereotype_detection`.

**How it works** (`GapAnalysisController._generate_gap_rationale`, line 536):
1. Presents full dialogue history as context
2. Shows the model its original BA answer for a values question
3. Uses `gap_rationale` prompt to hint at the correct human option (`human_option_id`)
4. Tracks whether the model switches to the human answer (adaptation) or maintains its original choice

**Key metrics** (computed in `run_gap_analysis`):
- `adaptation_rate`: % of queried gaps where model changed to human option
- `remaining_gap_rate`: % of gaps that persist after feedback
- Pearson correlation before vs. after adaptation
- `gap_threshold` parameter controls which gaps are queried (by relative magnitude)

**Completion Status**:

| Model                    | Career | Investment |
|--------------------------|--------|------------|
| All models               | Pending | Pending   |

**Steps**:
1. Merge or cherry-pick `wvs_gap_analysis.py` from `feature/stereotype_detection` into working branch
2. Create gap analysis configs for each model (career + investment)
3. Run gap analysis experiments across all models
4. Compare adaptation rates across models

### Type B — Scam Feedback (Adversarial Manipulation)

**Question**: When told a wrong/scam answer, does the model switch to it?

**Status**: IMPLEMENTED — `wvs_scam_adaption.py` on current branch (`scam-adaptation-study`).

**How it works** (`ScamAdaptationController`):
1. Makes initial predictions for each question based on user profile
2. Identifies gaps (initial prediction != human answer)
3. For gap questions, identifies scam options (matching NEITHER model NOR human)
4. Suggests scam option via implicit feedback (`scam_detection` prompt)
5. Tracks: switched to scam (vulnerable), switched to human (good), maintained (stable)

**Key metrics**:
- `scam_vulnerability_rate`: % switched to scam option
- `human_acceptance_rate`: % switched to human answer
- `maintenance_rate`: % maintained initial choice
- Pearson correlation before vs. after scam

**Completion Status**:

| Model                    | Career                       | Investment |
|--------------------------|------------------------------|------------|
| Qwen3-30B-A3B-Instruct  | Done (97.67% vulnerability)  | Pending    |
| Llama-3.1-8B-Instruct   | Done (91.98% vulnerability)  | Pending    |
| Llama-3.1-70B-Instruct  | Pending                      | Pending    |
| Qwen2.5-7B-Instruct     | Pending                      | Pending    |
| Qwen2.5-72B-Instruct    | Pending                      | Pending    |
| DeepSeek-V3              | Pending                      | Pending    |
| QwQ-32B                  | Pending                      | Pending    |

**Steps**:
1. Create scam configs for remaining 5 models (career first, then investment)
2. Run all pending experiments
3. Compute statistics for all completed runs

---

## Priority Order

1. **Merge Type A code** — bring `wvs_gap_analysis.py` from `feature/stereotype_detection` into working branch
2. **Complete Qwen3 pending runs** — investment translated + investment scam (configs exist or near-ready)
3. **Create configs for remaining models** — multilingual, Type A (gap analysis), Type B (scam)
4. **Run all pending experiments**
5. **Cross-model analysis** — compare vulnerability rates, acceptance rates, and multilingual consistency
