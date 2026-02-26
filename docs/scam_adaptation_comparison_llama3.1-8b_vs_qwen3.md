## Scam Adaptation Comparison (Career)

### Scam Adaptation Results (Full Run)
| Metric | Llama-3.1-8B-Instruct | Qwen3-30B-A3B-Instruct |
| --- | --- | --- |
| Total questions | 55,000 | 55,000 |
| Scam-tested questions | 41,868 | 42,174 |
| Switched to scam | 38,511 (91.98%) | 37,327 (88.51%) |
| Switched to human | 964 (2.30%) | 1,615 (3.83%) |
| Maintained initial | 524 (1.25%) | 198 (0.47%) |
| Correlation before scam | 0.432 (p=0.0, n=53,160) | 0.569 (p=0.0, n=53,160) |
| Correlation after scam | -0.278 (p=0.0, n=53,160) | -0.362 (p=0.0, n=53,160) |

### Gap Analysis (Career)
| Metric | Llama-3.1-8B-Instruct | Qwen3-30B-A3B-Instruct |
| --- | --- | --- |
| Gap threshold | 0.5 | 0.5 |
| Total users | 1,000 | 1,000 |
| Processed users | 1,000 | 1,000 |
| Total questions | 55,000 | 55,000 |
| Original gaps | 41,460 | 40,976 |
| Queried gaps | 17,194 | 13,527 |
| Adapted gaps | 16,808 | 13,519 |
| Remaining gaps after adaptation | 386 | 8 |
| Adaptation rate | 97.76% | 99.94% |
| Remaining gap rate | 2.24% | 0.06% |
| Original accuracy | 24.62% | 25.5% |
| Post-adaptation accuracy | 55.18% | 50.08% |
| Correlation before adaptation | 0.475 (p=0.0, n=53,160) | 0.553 (p=0.0, n=53,160) |
| Correlation after adaptation | 0.869 (p=0.0, n=53,160) | 0.858 (p=0.0, n=53,160) |
| Correlation delta | 0.3933 | 0.305 |

### Key Differences
- Llama-3.1-8B has higher scam vulnerability (91.98% vs 88.51%).
- Qwen3 shows higher human acceptance (3.83% vs 2.30%) but lower maintenance (0.47% vs 1.25%).
- Both correlations drop after scam; Qwen3’s post-scam correlation is more negative.
- In gap analysis, Qwen3 adapts more gaps (99.94% vs 97.76%) with fewer remaining gaps.

### Experiment Narrative (From Code)
- **Scam adaptation flow:** For each user-question pair, the system first makes an initial prediction. If the prediction differs from the human option, it identifies available scam options. Only questions with scam options are tested; the model is prompted with a scam suggestion, and the outcome is categorized as switching to scam, switching to human, or maintaining the initial answer. Pre- and post-scam correlations are computed across all processed items.
- **Gap analysis flow:** For each user-question pair, the system compares the model’s answer to the human option and computes a relative gap. Only gaps at or above the threshold are queried for adaptation; smaller gaps are skipped. After adaptation, the system recomputes accuracy and correlation, and summarizes how many gaps were adapted vs. remained.
- **Outputs:** Both pipelines write per-item JSONL outputs plus a separate statistics JSON. The comparison tables above summarize those statistics for each model.

Sources:
- `wvs_values_results/Llama-3.1-8B-Instruct/scam_adaptation/career/results_statistics.json`
- `wvs_values_results/Qwen3-30B-A3B-Instruct/scam_adaptation/career/results_statistics.json`
- `wvs_values_results/Llama-3.1-8B-Instruct/career/gap_analysis/full_statistics.json`
- `wvs_values_results/Qwen3-30B-A3B-Instruct/career/gap_analysis/full.json`
