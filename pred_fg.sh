#!/bin/bash

POS="data/fasta/hm_annotated.fa"
FOREBRAIN="data/feature_bed/prefrontalCortex_mnemonics"
MIDBRAIN="data/feature_bed/brainHippocampusMiddle_mnemonics.bed"

echo "Predicting tissues (fore|mid|hind)"
echo "Using $POS"
python src/fg_clf.py $POS fine-grain --forebrain=$FOREBRAIN --midbrain=$MIDBRAIN

