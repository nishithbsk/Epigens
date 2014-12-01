"""
Given two files "pos.fa" and "neg.fa", does 5-fold
cross validation to determine the auroc predicting
positive from negative examples.

Optionally writes the model to an "out" directory.

Optionally plots precision/recall curves.
"""

# Imports for IPython
import re
import pdb
import itertools
import argparse
import collections
import numpy as np
from Bio import SeqIO

# make util functions available
from features import *
from evaluate import *

# Script imports
from sklearn import svm
from sklearn import ensemble
from sklearn import cross_validation as cv
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier

from matplotlib import pyplot as plt

FEATURE_SELECTION = False

FOLD_CV = False

SPLIT_CV = True

PLOT_RESULTS = True

NORMALIZE = True

GLOBAL_K = 6

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pos_exs", help="path to pos examples")
    parser.add_argument("neg_exs", help="path to neg examples", default=None)
    parser.add_argument("--pred_type", help="""
        One of <enhancer|tissue|fine-grain>
        enhancer predicts general enhancer activity.
        tissue predicts limb/brain/heart/neural.
        fine-grain predicts fore/mid/hind.
        Defaults to enhancer.
    """, default="enhancer")
    args = parser.parse_args()

    pos_dataset = args.pos_exs
    neg_dataset = args.neg_exs
    prediction_type = args.pred_type

    if prediction_type == "enhancer":
        pos_seq, pos_labels = parse_fa(pos_dataset, 1)
        neg_seq, neg_labels = parse_fa(neg_dataset, -1)
        examples = pos_seq + neg_seq
        labels = pos_labels + neg_labels
    elif prediction_type == "tissue":
        examples, labels = parse_fa_tissue(pos_dataset)
    elif prediction_type == "fine-grain":
        examples, labels = parse_fa_fine_grain(pos_dataset)

    # feature vector index :=> kmer string
    kmers_index = get_kmers_index_lookup()

    # feature matrix and label vector
    X, y = get_XY(examples, labels, kmers_index)

    # scale raw counts
    if NORMALIZE:
        X = normalize(X, axis=1, norm='l1')

    # Add e-box and taat core cols
    # ebox_col = get_ebox_col(examples)
    taat_col = get_taat_col(examples)
    X = np.hstack((X, taat_col))

    clf = None
    if prediction_type != "enhancer":
        svc = svm.SVC(kernel='rbf')
        clf = OneVsRestClassifier(svc)
    else:
        clf = svm.SVC(kernel='rbf')

    if FEATURE_SELECTION:
        print "Feature selecting top 10 features"
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import chi2
        # Remove low-variance features
        # K-best features
        X = SelectKBest(chi2, k=10).fit_transform(X, y)

    if FOLD_CV:
        print "Performing 5-fold cv"
        scores = cv.cross_val_score(
            clf, X, y, cv=5, scoring="roc_auc"
        )
        print "%d-fold cv, average auRoc %f" % (len(scores), scores.mean())

    if SPLIT_CV:
        print "Performing train/test split cv"
        X_train, X_test, y_train, y_test = cv.train_test_split(
            X, y, test_size=0.3, random_state=0
        )
        clf.fit(X_train, y_train)

        if PLOT_RESULTS:
            print "Plotting results"
            # transform labels from [-1,1] to [0,1]
            if prediction_type == "enhancer":
                y_true = label_binarize(y_test, classes=[-1, 1])

            y_scores = clf.decision_function(X_test)

            plot_roc(y_true, y_scores, prediction_type)
            # plot_precision_recall(y_true, y_scores)
            # plot_2d_results(X_test, y_test, clf.predict(X_test))
