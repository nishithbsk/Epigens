#!/bin/bash

POS="data/fasta/vt_hm_pos.fa"
NEG="data/fasta/vt_hm_neg.fa"
POS_TF="data/out/pos_tf.txt"
NEG_TF="data/out/neg_tf.txt"

echo "Predicting general enhancer activity"
echo "pos: $POS, neg: $NEG, tf: $TF"

python src/enhancer_clf.py $POS $NEG --pos_tf=$POS_TF --neg_tf=$NEG_TF

