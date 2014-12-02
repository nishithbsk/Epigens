#!/bin/bash

POS="data/fasta/hm_annotated.fa"
HEART="data/feature_bed/heart_mnemonics.bed"
LIVER="data/feature_bed/liver_mnemonics.bed"
BRAIN="data/feature_bed/brain_mnemonics.bed"

echo "Predicting tissues (limb|brain|heart|neural|other)"
echo "Using $POS"
python src/tissue_clf.py $POS --heart=$HEART --liver=$LIVER --brain=$BRAIN

