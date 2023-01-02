#!/usr/bin/env bash

set -eu

layout="post"
subdir="blog/coronary_artery_disease_investigation"

../notebook_convert.py \
    --nbpath coronary-artery-disease-investigation.ipynb \
    --date "2023-01-02" \
    --layout ${layout} \
    --subdir ${subdir} \
    --description "." \
    --tags "Logistic Function" "Logistic Regression" "Machine Learning" "Cross-Entropy" "Classification" "Gradient Descent" "Neural Networks" "Notebook"
