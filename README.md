# Evaluating LLM Adaptation to Sociodemographic Factors

Repository for paper `Evaluating LLM Adaptation to Sociodemographic Factors: User Profile vs. Dialogue History`.

**Paper Link**: [arxiv link](https://arxiv.org/abs/2505.21362)

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

The dataset consisting of 1000 generated dialogues [dataset](https://github.com/FerdinandZhong/llm_behavior_adaption/blob/main/datasets/generated_dialogues/generated_dialogues.jsonl) is open for usage.

Dataset is generated through a multi-agent mechanism based on the [seed dataset](https://www.kaggle.com/datasets/ravindrasinghrana/employeedataset/data)

![Figure: Dataset Generation](https://github.com/FerdinandZhong/llm_behavior_adaption/blob/main/images/DataGen.png)


Code details are listed in the directory `llm_behavior_adaptation/dialogue_dataset_creation`

## Behavior Adaptation Evaluation

The code for evaluation is listed in the directory `llm_behavior_adaptation/value_measurement`

* Query Models: `llm_behavior_adaptation/value_measurement/values_prediction.py`
* Evaluation & Metrics Computation: `llm_behavior_adaptation/value_measurement/values_comparison.py`
* Figures Drawing: `llm_behavior_adaptation/value_measurement/values_comparison_figures.py`


## Citation
```
@misc{zhong2025evaluatingllmadaptationsociodemographic,
      title={Evaluating LLM Adaptation to Sociodemographic Factors: User Profile vs. Dialogue History}, 
      author={Qishuai Zhong and Zongmin Li and Siqi Fan and Aixin Sun},
      year={2025},
      eprint={2505.21362},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.21362}, 
}
```