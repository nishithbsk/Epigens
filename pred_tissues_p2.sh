#!/bin/bash

POS="data/fasta/hm_annotated.fa"
HEART="data/out/heart.npy"
LIVER="data/out/limbs.npy"
BRAIN="data/out/brain.npy"

echo "Predicting tissues (limb|brain|heart|neural|other)"
echo "Using $POS"
python src/tissue_clf.py $POS --heart=$HEART --liver=$LIVER --brain=$BRAIN

