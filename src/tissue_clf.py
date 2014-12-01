"""
Base script. Intended to be forked and modified.

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
from plot import *

# Script imports
from sklearn import svm
from sklearn import ensemble
from sklearn import cross_validation as cv
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from matplotlib import pyplot as plt

FEATURE_SELECTION = False

FOLD_CV = True

PLOT_RESULTS = False

NORMALIZE = True

GLOBAL_K = 6


# ==========================================================


tissues = ["limb", "heart"]


def filter_fn(d):
    for raw in d.split("|")[4:]:
        for part in tissues:
            if part in raw:
                return True
    return False


def one_v_all(d):
    label = [0, 0, 0, 0]
    for raw in d.split("|")[4:]:
        line = raw.strip()
        if "brain" in line:
            label[0] = 1
        if "limb" in line:
            label[1] = 1
        if "heart" in line:
            label[2] = 1
        if "neural" in line:
            label[3] = 1
    return label


def one_v_one(d):
    for raw in d.split("|")[4:]:
        if tissues[0] in raw:
            return 1
        elif tissues[1] in raw:
            return -1
        else:
            raise Exception("Shouldn't have more than two classes here")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pos_exs", help="path to pos examples")
    args = parser.parse_args()

    pos_dataset = args.pos_exs
    examples, labels = parse_fa_tissue(pos_dataset, one_v_all, filter_fn)

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

    clf = svm.SVC(kernel='linear')
    # clf = OneVsRestClassifier(svc)

    if FEATURE_SELECTION:
        print "Feature selecting top 10 features"
        # Remove low-variance features
        # K-best features
        X = SelectKBest(chi2, k=600).fit_transform(X, y)

    if FOLD_CV:
        print "Performing 5-fold cv"
        scores = cv.cross_val_score(
            clf, X, y, cv=5, scoring="roc_auc"
        )
        print "%d-fold cv, average auRoc %f" % (len(scores), scores.mean())

    if PLOT_RESULTS:
        X_train, X_test, y_train, y_test = cv.train_test_split(
            X, y, test_size=0.3, random_state=0
        )
        clf.fit(X_train, y_train)

        print "Plotting results"
        y_scores = clf.decision_function(X_test)
        plot_roc(y_test, y_scores, "ROC Tissue", 
            out="figures/roc-curve-tis.png")
        # plot_precision_recall(y_true, y_scores)
        # plot_2d_results(X_test, y_test, clf.predict(X_test))
        print "Done plotting"
