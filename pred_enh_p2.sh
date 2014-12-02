#!/bin/bash

POS="data/fasta/vt_hm_pos.fa"
NEG="data/fasta/vt_hm_neg.fa"
TF="data/feature_bed/TF_binding_sites.bed"

echo "Predicting general enhancer activity"
echo "pos: $POS, neg: $NEG, tf: $TF"
python src/enhancer_clf.py $POS $NEG --tf=TF

