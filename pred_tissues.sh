#!/bin/bash

POS="data/fasta/hm_annotated.fa"

echo "Predicting tissues (limb|brain|heart|neural|other)"
echo "Using $POS"
python src/enhancer_clf.py $POS none tissue

