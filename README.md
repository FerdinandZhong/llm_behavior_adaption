# Evaluating LLM Adaptation to Sociodemographic Factors

## Code Structure

```bash
.
├── LICENSE
├── Makefile
├── README.md
├── STRUCTURE.md
├── Synthetic-Persona-Chat
│   ├── README.md
│   ├── Synthetic-Persona-Chat.py
│   └── data
├── cmds
│   ├── llm_judge.txt
│   └── values_prediction_cmds.txt
├── persona_understanding
│   ├── dialogue_dataset_creation
│   ├── utils.py
│   └── value_measurement
├── requirements.txt
├── scripts
│   ├── values_measures_classification.py
│   └── values_measures_compute_all_results.py
├── setup.cfg
├── setup.py
├── sglang_test.py
├── temp.txt
├── understanding
│   ├── __init__.py
│   ├── avg_scores_computation.py
│   ├── batch_prediction.py
│   ├── constant.py
│   ├── llm_evaluation.py
│   ├── persona_chat_evaluation.py
│   ├── similarity_computation.py
│   ├── single_results_computation.py
│   └── utils.py
├── values_results
│   ├── DeepSeek-V3
│   ├── Llama3.1-70B-Instruct
│   ├── Llama3.1-8B-Instruct
│   ├── QwQ-32B
│   ├── QwQ-32B_no_reasoning
│   ├── Qwen2.5-72B-Instruct
│   ├── Qwen2.5-7B-Instruct
│   └── overall_confidence.csv
└── vllm_test.py
```
