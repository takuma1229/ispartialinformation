#!/bin/bash

# Array of experiment types
declare -a experiment_types=(
    "normal"
    "former-only"
    "latter-only"
    "exclude_mo"
    "exclude_koso"
    "exclude_function_words"
    "exclude_content_words"
    # "exclude_particle"
    "exclude_ha_and_ga"
    "exclude_negation"
    "convert_content_words_to_dummy"
    "convert_function_words_to_dummy"
    "convert_all_words_to_dummy"
    "exclude_connectives"
    "convert_connectives_to_dummy"
    # "exclude_iru_aru_oku"
)

# Base command
base_command="python src/run_multiple_experiments.py --model_name tohoku-nlp/bert-base-japanese-v3"

# Loop through each experiment type
for experiment_type in "${experiment_types[@]}"
do
    echo "Running experiment: $experiment_type"
    $base_command --experiment_type=$experiment_type

    # Check if the last command succeeded
    if [ $? -ne 0 ]; then
        echo "Error occurred during execution of experiment: $experiment_type"
        exit 1
    fi
done

echo "All experiments completed successfully."
