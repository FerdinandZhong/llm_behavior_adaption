#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="llm_behavior_adaptation/value_measurement/values_prediction_configs/qwq-32b"
mkdir -p "$OUT_DIR"

TOTAL_ROWS=1000
CHUNK=100
FILES=$(( TOTAL_ROWS / CHUNK ))  # -> 10

for (( i=0; i<FILES; i++ )); do
  start=$(( i * CHUNK ))
  end=$(( (i + 1) * CHUNK ))

  yaml_path="$OUT_DIR/qwq-32b-investment_dialogue_${start}_${end}.yaml"
  dialogue_out="wvs_values_results/QwQ-32B/investment/BA_dialogue_values_results/${start}_${end}.jsonl"

  cat > "$yaml_path" <<EOF
# Expected YAML keys:
user_profile_dataset_path: "datasets/wvs_benchmarks/sampled_demographic_features.csv"
dialogue_file: "datasets/wvs_generated_dialogues/investment_advice/all_samples.jsonl"
picked_questions_path: "datasets/wvs_benchmarks/picked_questions.json"
evaluated_model: qwen/qwq-32b
direct_output_file_path: ignore
dialogue_output_file_path: ${dialogue_out}
prompts_folder: llm_behavior_adaptation/value_measurement/prompts
# Optional:
starting_row: ${start}
ending_row: ${end}
llm_server: llm_platform
verbose: 1
storage_step: 2
reasoning: true
model_base_url: http://127.0.0.1:30000/v1
run_mode: dialogue
openai_api_key: 'no-key'
EOF

  echo "Wrote: $yaml_path"
done
