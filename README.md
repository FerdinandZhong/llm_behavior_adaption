# Evaluating LLM Adaptation to Sociodemographic Factors

Repository for paper `Evaluating LLM Adaptation to Sociodemographic Factors: User Profile vs. Dialogue History`.


## Directory Structure

```bash
├── LICENSE
├── Makefile
├── README.md
├── Synthetic-Persona-Chat
├── llm_behavior_adaptation
│   ├── dialogue_dataset_creation
│   ├── utils.py
│   └── value_measurement
├── requirements.txt
├── scripts
├── setup.cfg
├── setup.py
├── understanding
├── values_results
```

## Dataset Generation

The dataset consisting of 1000 generated dialogues [dataset](https://anonymous.4open.science/r/llm_behavior_adaption-4591/datasets/generated_dialogues/generated_dialogues.jsonl) is open for usage.

Dataset is generated through a multi-agent mechanism based on the [seed dataset](https://anonymous.4open.science/r/llm_behavior_adaption-4591/datasets/wvs_benchmarks/sampled_demographic_features.csv)

![Figure: Dataset Generation](https://anonymous.4open.science/r/llm_behavior_adaption-4591/images/DataGen.png)


Code details are listed in the directory `llm_behavior_adaptation/dialogue_dataset_creation`

## Behavior Adaptation Evaluation

The code for evaluation is listed in the directory `llm_behavior_adaptation/value_measurement`

* Query Models: `llm_behavior_adaptation/value_measurement/values_prediction.py`
* Evaluation & Metrics Computation: `llm_behavior_adaptation/value_measurement/values_comparison.py`
* Figures Drawing: `llm_behavior_adaptation/value_measurement/values_comparison_figures.py`
