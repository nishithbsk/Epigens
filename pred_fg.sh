#!/bin/bash

POS="data/fasta/hm_annotated.fa"

echo "Predicting tissues (fore|mid|hind)"
echo "Using $POS"
python src/enhancer_clf.py $POS none fine-grain

