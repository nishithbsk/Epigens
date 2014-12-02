"""
Given an annotated fasta file of enhancer tissues,
trains and evaluates classifier to distinguish selected
tissues.

Classifiers used are one-vs-one to distinguish btwn
two tissues, and one-vs-all for the rest.
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

FOLD_CV = False

PLOT_RESULTS = not FOLD_CV

NORMALIZE = True

GLOBAL_K = 6


tissues = ["limb", "brain"]


def filter_fn(d):
    for raw in d.split("|")[4:]:
        for part in tissues:
            if part in raw:
                return True
    return False


def one_v_all(d):
    """ converts description to a multilabel
    np array to use in a one-v-all classifier """
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
    """ Treats the first two entries in brain-parts
    as the labels for a one-v-one clf """
    for raw in d.split("|")[4:]:
        if tissues[0] in raw:
            return 1
        elif tissues[1] in raw:
            return -1
    raise Exception("Shouldn't have more than two classes here")


def extract_seq(d):
    return


import time
if __name__ == "__main__":
    start = time.clock()
    parser = argparse.ArgumentParser()
    parser.add_argument("pos_exs", help="path to pos examples")
    parser.add_argument("--heart_mnemonic", help="path to heart mnemonic")
    parser.add_argument("--liver_mnemonic", help="path to liver mnemonic")
    parser.add_argument("--brain_mnemonic", help="path to brain mnemonic")
    args = parser.parse_args()

    pos_dataset = args.pos_exs
    heart_data = None
    liver_data = None
    brain_data = None

    if hasattr(args, "heart_mnemonic"):
        heart_data = args.heart_mnemonic
    if hasattr(args, "liver_mnemonic"):
        liver_data = args.liver_mnemonic
    if hasattr(args, "brain_mnemonic"):
        brain_data = args.brain_mnemonic

    examples, labels, descriptions = parse_fa_tissue(
        pos_dataset, one_v_one, filter_fn
    )

    # feature vector index :=> kmer string
    kmers_index = get_kmers_index_lookup()

    # feature matrix and label vector
    X, y = get_XY(examples, labels, kmers_index)

    # scale raw counts
    if NORMALIZE:
        X = normalize(X, axis=1, norm='l1')

    # Add extra features
    if heart_data:
        rows = [extract_extra_features_heart(extract_seq(d), heart_data)
            for d in descriptions])
        heart_X = np.concatenate(rows)



    # Add e-box and taat core cols
    # ebox_col = get_ebox_col(examples)
    # taat_col = get_taat_col(examples)
    # X = np.hstack((X, taat_col))

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
            out="figures/roc-curve-tis-limb-v-brain.png")
        # plot_precision_recall(y_true, y_scores)
        # plot_2d_results(X_test, y_test, clf.predict(X_test))
        print "Done plotting"

    end = time.clock()
    print "time taken : %fs" % (end - start)
