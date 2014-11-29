"""
Given two files "pos.fa" and "neg.fa", does 5-fold
cross validation to determine the auroc predicting
positive from negative examples.

Optionally writes the model to an "out" directory.

Optionally plots precision/recall curves.
"""

import re
import pdb
import itertools
import argparse
import numpy as np

from Bio import SeqIO
from sklearn import svm
from sklearn import ensemble
from sklearn import cross_validation as cv
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier

from matplotlib import pyplot as plt

# === Config ===

FEATURE_SELECTION = False

FOLD_CV = True

SPLIT_CV = False

PLOT_RESULTS = False

NORMALIZE = True

GLOBAL_K = 6


# === Preprocessing ===

def reverse(input_str):
    """ Simple reverse string """
    list_form = list(input_str)
    list_form.reverse()
    return list_form


def cmpl_base_pair(x):
    """ Get complementary base pair """
    if x == 'A':
        return 'T'
    elif x == 'C':
        return 'G'
    elif x == 'T':
        return 'A'
    elif x == 'G':
        return 'C'
    else:
        return 'N'


def neg_strand(pos_strand):
    """ Given pos_strand sequence, returns complementary
    negative sequence """
    return "".join(map(cmpl_base_pair, reverse(pos_strand)))


def gen_kmers(seq, k=GLOBAL_K):
    """ Returns generator for all kmers in given DNA sequence.
    May contain duplicates """
    return (seq[i:i + k] for i in range(len(seq) - k + 1))


def locfd(description):
    """ converts descriptions to locations. Input
    is a fasta descr. delimited by "|". Whether
    the region is enhanced is denoted by the ?th
    entry in the array """
    return description.split("|")[4:]


def lts(label):
    """ converts label to sign """
    return 1 if label == "positive" else -1


def lfd(description):
    """ converts descriptions to labels. Input
    is a fasta descr. delimited by "|". Whether
    the region is enhanced is denoted by the 4th
    entry in the array """
    return lts(description.split("|")[3].strip())


"""
Dist:
    forebrain => 309
    midbrain => 253
    hindbrain => 236
    neural tube => 170
    limb => 162
    other => 596
"""

def lftd_binary(description):
    """ converts descriptions to multi-label
    tissue label.
    Indices correspond to:
        brain => 0
        limb => 1
        heart => 2
        neural => 3
    """
    for raw in description.split("|")[4:]:
        line = raw.strip()
        # if "brain" in line:
        #     return 1
        if "limb" in line:
            return 1
        # if "heart" in line:
        #     return 1
        # if "neural" in line:
        #     return 1
    return -1


def lftd(description):
    """ converts descriptions to multi-label
    tissue label.
    Indices correspond to:
        brain => 0
        limb => 1
        heart => 2
        neural => 3
    """
    label = [0, 0]
    for raw in description.split("|")[4:]:
        line = raw.strip()
        if "brain" in line:
            label[0] = 1
        if "limb" in line:
            label[1] = 1
        # if "heart" in line:
        #     label[2] = 1
        # if "neural" in line:
        #     label[3] = 1
    return label


def lfbd(description):
    """ converts descriptions to multi-label
    brain label.
    Indices in label correspond to:
        1 => "forebrain"
        2 => "midbrain"
        3 => "hindbrain"
    """
    label = [0, 0, 0]
    for raw in description.split("|")[4:]:
        line = raw.strip()
        if "midbrain" in line:
            label[0] = 1
        if "forebrain" in line:
            label[1] = 1
        if "hindbrain" in line:
            label[2] = 1
    return label


def parse_fa(path, label):
    """ Given a fasta file that represents a label
    class, returns a pair of (sequence, label) numpy
    arrays. Useful for constructing X/y for training """
    fasta_file = open(path)
    human_fasta_seq = SeqIO.parse(fasta_file, 'fasta')
    seqs = []
    _labels = []

    for entry in human_fasta_seq:
        seqs.append(str(entry.seq).replace("n", "").lower())
        _labels.append(float(label))

    _labels = np.array(_labels)
    return (seqs, _labels)


def parse_fa_tissue(path):
    """ Given a fasta file that represents positive
    labels, returns a pair of (sequence, label) numpy
    arrays. Useful for constructing X/y for training """
    fasta_file = open(path)
    human_fasta_seq = SeqIO.parse(fasta_file, 'fasta')

    seqs = []
    _labels = []

    for entry in human_fasta_seq:
        seqs.append(str(entry.seq).replace("n", "").lower())
        # _labels.append(lftd_binary(entry.description))
        _labels.append(lftd(entry.description))

    _labels = np.array(_labels)
    return (seqs, _labels)


def parse_fa_fine_grain(path):
    """ Given a fasta file that represents positive
    labels, returns a pair of (sequence, label) numpy
    arrays. Useful for constructing X/y for training """
    fasta_file = open(path)
    human_fasta_seq = SeqIO.parse(fasta_file, 'fasta')
    seqs = []
    _labels = []

    for entry in human_fasta_seq:
        seqs.append(str(entry.seq).replace("n", "").lower())
        _labels.append(lfbd(entry.description))

    _labels = np.array(_labels)
    return (seqs, _labels)


def get_kmer_counts(seq, ref):
    """ Given example sequence and a reference table mapping
    kmers to indices, returns a numpy array representing one row
    of the feature vector.

    NOTE Finds kmers on both strands, avoids double-counting.
    This is an assumption made due to Figure 1 of the Lee et al. paper
    which describes a feature vector x with such counts:

        5' AAAAAA 3'  |>> x1
        3' TTTTTT 5'  |
        -------------------------
        ...
        -------------------------
        5' TTTAAA 3'  |>> xn
        3' AAATTT 5'  |
        -------------------------

    Based on the last entry, if "TTTAAA" is the same on both strands,
    don't count it.
    """
    row = np.zeros(len(ref))
    pos_kmers = gen_kmers(seq)
    for kmer in pos_kmers:
        if kmer != neg_strand(kmer):
            idx = ref[kmer]
            row[idx] += 1
    return row


def make_index_dict_from_list(l):
    """ Given list, creates dictionary where keys
    are contents of array and value is index of
    original array """
    return dict([(x, i) for i, x in enumerate(l)])


def get_kmers_index_lookup():
    """ Builds up mapping of index to a unique kmer """
    global GLOBAL_K
    all_kmers = [''.join(x) for x in itertools.product("atcg", repeat=GLOBAL_K)]
    return make_index_dict_from_list(list(set(all_kmers)))


def get_XY(examples, labels, kmer_index):
    X = np.vstack([get_kmer_counts(x, kmer_index) for x in examples])
    y = np.array(labels, dtype=np.int32)
    print "train. matrix dims (X): ", X.shape
    print "num labels (y): ", len(y)
    if len(labels.shape) == 1:
        print "+ ", len(np.where(y == 1)[0])
        print "- ", len(np.where(y == -1)[0])
    print "------------------------------------"
    return (X, y)


def get_taat_col(examples):
    regex = "^([atgc])+(taat)([atcg])+$"
    expr = lambda x: 1.0 if re.match(regex, x) else 0.0
    return np.array(map(expr, examples)).reshape(len(examples), 1)


def get_ebox_col(examples):
    regex = "ca[atcg]{2}tg"
    expr = lambda x: 1.0 if re.search(regex, x) else 0.0
    return np.array(map(expr, examples)).reshape(len(examples), 1)


# === Prediction ===


def plot_2d_results(X, y, preds):
    pca = PCA(n_components=2)
    X_r = pca.fit(X).transform(X)

    # Plot scatter
    plt.figure()
    cs = "cm"
    cats = [1, -1]
    target_names = ["positive", "negative"]
    for c, i, target_name in zip(cs, cats, target_names):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c, label=target_name)
    plt.legend()
    plt.title("PCA of 2d data")
    plt.savefig("figures/data-scatter.png")

    # Plot mispredictions
    plt.figure()
    diff = np.array([1 if y_test[i] == preds[i] else 0 for i in range(len(y_test))])
    cs = "rg"
    cats = [0, 1]
    target_names = ["incorrect", "correct"]
    for c, i, target_name in zip(cs, cats, target_names):
        plt.scatter(X_r[diff == i, 0], X_r[diff == i, 1], c=c, label=target_name)
        plt.legend()
        plt.title("PCA of correct/incorrect predictions")
    # plt.show()
    plt.savefig("figures/residual-scatter.png")


def plot_precision_recall(y_test, y_scores):
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
    plt.figure()
    plt.plot(recall, precision, 'g-')
    plt.title("Precision-Recall Curve")
    plt.savefig("figures/pr-curve.png")


def plot_roc(y_test, y_score):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(1):
        fpr[i], tpr[i], _ = roc_curve(y_test, y_score)
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # Plot of a ROC curve for a specific class
    plt.figure()
    plt.plot(fpr[0], tpr[0], label='ROC curve (area = %0.2f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC, Kmer counts used to predict general enhancer functionality')

    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig("roc-curve.png")


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
    # taat_col = get_taat_col(examples)
    # X = np.hstack((X, ebox_col, taat_col))

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
        if prediction_type != "enhancer":
            svc = clf
            clf = OneVsRestClassifier(svc)
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
        clf.score(X_test, y_test)

        if PLOT_RESULTS:
            print "Plotting results"
            # transform labels from [-1,1] to [0,1]
            _y_test = label_binarize(y_test, classes=[-1, 1])
            y_scores = clf.decision_function(X_test)

            plot_roc(_y_test, y_scores)
            plot_precision_recall(_y_test, y_scores)
            plot_2d_results(X_test, y_test, clf.predict(X_test))
