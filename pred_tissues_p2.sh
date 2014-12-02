#!/bin/bash

POS="data/fasta/hm_annotated.fa"
HEART="data/out/heart.txt"
LIVER="data/out/limbs.bed"
BRAIN="data/out/brain.bed"

echo "Predicting tissues (limb|brain|heart|neural|other)"
echo "Using $POS"
python src/tissue_clf.py $POS --heart=$HEART --liver=$LIVER --brain=$BRAIN

