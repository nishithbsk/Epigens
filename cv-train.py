"""
Entry point. Load, extract counts of kmers
as features,trains an assembly of classifiers, and
serializes outputs to files.
"""
import pandas as pd
import numpy as np
import itertools
import pdb
import re
# import pickle

from Bio import SeqIO
from sklearn import svm
from sklearn import cross_validation
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.externals import joblib
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

from matplotlib import pyplot as plt


# === Config ===

TRAIN_EXPERIMENTAL = True

SHOULD_SPLIT = True

SPLIT_PLOT_RESULTS = False

FEATURE_SELECTION = False

CONTINUOUS_FEATURES = True

FOLD_CV = False

NORMALIZE = True

VISTA_TABLE_SRC = "data/vistaTable20141113.txt"

FASTA_HUMAN_SRC = "data/humanRegions.fasta"

TRAIN_DATA_DST = "out/%s.train"

POS_DATASET = "data/fasta/pos.fa"

NEG_DATASET = "data/fasta/neg.fa"

GLOBAL_K = 6


# === Preprocessing ===


def reverse(st):
    a = list(st)
    a.reverse()
    return a


def cmpl_base_pair(x):
    if x == 'A': return 'T'
    elif x == 'C': return 'G'
    elif x == 'T': return 'A'
    elif x == 'G': return 'C'
    else: return 'N'


def neg_strand(pos_strand):
    """ Given pos_strand sequence, returns complementary
    negative sequence """
    return "".join(map(cmpl_base_pair, reverse(pos_strand)))


def gen_kmers(seq, k=GLOBAL_K):
    """ Returns generator for all kmers in given DNA sequence.
    May contain duplicates """
    return (seq[i:i+k] for i in range(len(seq) - k + 1))


def set_kmers(seq, k=GLOBAL_K):
    """ Returns kmers but with dups remove """
    return set([x for x in gen_kmers(seq, k)])


def set_kmers_reducer(accumSet, seq):
    """ Reducer form of set_kmers """
    return accumSet.union(set_kmers(seq))


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


def load_vista_db(path):
    """ Loads the vista table for reference.
    Returns a Pandas DataFrame for easy access.

    Useful when annotating kmers with histones data
    """

    VT_f = open(path)
    _df = pd.read_csv(
        VT_f, delimiter="\t",
        colNames=["id", "bracketing_genes", "human_coords", "mouse_coords"])
    print "--> vista table shape loaded from %s: %s" % (path, str(_df.shape))
    return _df


def load_named_seq(path):
    """ Returns two lists of tuples The first is (id<int>, seq<str>),
    representing each entry of the fasta file.
    The second is (id<int>, label<str>) representing the labels
    associated with each id (positive/negative)"""

    FH_f = open(path)
    human_fasta_seq = SeqIO.parse(FH_f, 'fasta')
    _named_sequences = []
    _named_descriptions = []
    _named_locations = []

    # TODO Should we be using "lower" here? What does it mean
    # for a FASTA letter to be capitalized?
    for x in human_fasta_seq:
        _named_sequences.append((x.id, str(x.seq).lower()))
        _named_descriptions.append((x.id, lfd(x.description)))
        _named_locations.append((x.id, locfd(x.description)))

    print "--> num sequences read from %s: %d" % (path, len(_named_sequences))
    return (_named_sequences, _named_descriptions, _named_locations)


def parse_fa(path, label):
    FH_f = open(path)
    human_fasta_seq = SeqIO.parse(FH_f, 'fasta')
    seqs = []
    labels = []

    counter = 0

    # TODO Should we be using "lower" here? What does it mean
    # for a FASTA letter to be capitalized?
    for x in human_fasta_seq:
        seqs.append((x.id, str(x.seq).lower()))
        labels.append((x.id, float(label)))
        counter += 1
        if counter >= 2096:
            break

    return (seqs, labels)


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
            if CONTINUOUS_FEATURES:
                row[idx] += 1
            else:
                row[idx] = 1
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


def get_XY(examples, labels_mapping, kmer_index):
    all_seqs = [x[1] for x in examples]
    X = np.vstack([get_kmer_counts(x, kmer_index) for x in all_seqs])
    y = np.array([x[1] for x in labels_mapping])
    print "train. matrix dims (X): ", X.shape
    print "num labels (y): ", len(y)
    print "+ ", len([1 for y_i in y if y_i == 1])
    print "- ", len([1 for y_i in y if y_i == -1])
    print "------------------------------------"
    return (X, y)


def get_TAAT_core_col(examples):
    all_seqs = [x[1] for x in examples]
    regex = "^([atgc])+(taat)([atcg])+$"
    expr = lambda x: 1.0 if re.match(regex, x) else 0.0
    return np.array(map(expr, all_seqs)).reshape(len(all_seqs), 1)


def get_Ebox_col(examples):
    all_seqs = [x[1] for x in examples]
    regex = "ca[atcg]{2}tg"
    expr = lambda x: 1.0 if re.search(regex, x) else 0.0
    return np.array(map(expr, all_seqs)).reshape(len(all_seqs), 1)


def get_locations_to_y_tIndex(locations):
    """ locations_to_y_tIndex is a dictionary that maps location
    (eg. hindbrain) to  indices into the y_t vector. """

    locations_to_y_tIndex = {
        'forebrain': [],
        'hindbrain': [],
        'limb': [],
        'rest': []
    }

    cutoff = (8 * len(locations)) / 10

    index = 0
    for (x, y) in locations[cutoff:]:
        if len(y) > 0:
            for location in y:
                if "forebrain" in location:
                    locations_to_y_tIndex['forebrain'].append(index)
                if "hindbrain" in location:
                    locations_to_y_tIndex['hindbrain'].append(index)
                if "limb" in location:
                    locations_to_y_tIndex['limb'].append(index)
                if "forebrain" not in location and \
                        "hindbrain" not in location and \
                        "limb" not in location:
                    if index not in locations_to_y_tIndex['rest']:
                        locations_to_y_tIndex['rest'].append(index)
        index += 1
    return locations_to_y_tIndex


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


# feature vector index :=> kmer string
kmers_index = get_kmers_index_lookup()


if TRAIN_EXPERIMENTAL:
    pos_seq, pos_lmap = parse_fa(POS_DATASET, 1)
    neg_seq, neg_lmap = parse_fa(NEG_DATASET, -1)

    examples = pos_seq + neg_seq
    labels = pos_lmap + neg_lmap
    X, y = get_XY(examples, labels, kmers_index)

    if NORMALIZE:
        X = normalize(X, axis=1, norm='l1')

    # Add e-box and taat core cols
    # X = np.hstack((
    #     X,
    #     get_Ebox_col(train_ex),
    #     get_TAAT_core_col(train_ex)
    # ))

    clf = svm.SVC(kernel='rbf', C=1)

    if NORMALIZE:
        X = normalize(X_train, axis=1, norm='l1')

    if FEATURE_SELECTION:
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import chi2
        # Remove low-variance features
        # K-best features
        X = SelectKBest(chi2, k=60).fit_transform(X, y)

    if SHOULD_SPLIT:
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(
            X, y, test_size=0.3, random_state=0)
        clf.fit(X_train, y_train)
        clf.score(X_test, y_test)

        # transform labels from [-1,1] to [0,1]
        _y_test = label_binarize(y_test, classes=[-1, 1])
        y_scores = clf.decision_function(X_test)

        scores = cross_validation.cross_val_score(clf, X_train, y_train, cv=5)
        print "%d-fold cv, average accuracy %f" % (len(scores), scores.mean())


        if SPLIT_PLOT_RESULTS:
            plot_roc(_y_test, y_scores)
            plot_precision_recall(_y_test, y_scores)
            plot_2d_results(X_test, y_test, clf.predict(X_test))


else:
    examples, labels, locations = load_named_seq(FASTA_HUMAN_SRC)
    _X, _y = get_XY(examples, labels, kmers_index)

    # X = np.hstack((
    #     X,
    #     get_Ebox_col(examples),
    #     get_TAAT_core_col(examples)
    # ))

    clf = svm.SVC(kernel='linear', C=1)

    X = _X
    X = VarianceThreshold(threshold=2.0).fit_transform(X)
    y = _y

    if NORMALIZE:
        X = normalize(X, axis=1, norm='l1')

    if FEATURE_SELECTION:
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import chi2
        # Remove low-variance features
        # K-best features
        X = SelectKBest(chi2, k=60).fit_transform(X, y)

    if SHOULD_SPLIT:
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(
            X, y, test_size=0.3, random_state=0)
        clf.fit(X_train, y_train)
        clf.score(X_test, y_test)

        # transform labels from [-1,1] to [0,1]
        _y_test = label_binarize(y_test, classes=[-1, 1])
        y_scores = clf.decision_function(X_test)

        if SPLIT_PLOT_RESULTS:
            plot_roc(_y_test, y_scores)
            plot_precision_recall(_y_test, y_scores)
            plot_2d_results(X_test, y_test, clf.predict(X_test))

    if FOLD_CV:
        scores = cross_validation.cross_val_score(clf, X, y, cv=5)
        print "%d-fold cv, average accuracy %f" % (len(scores), scores.mean())


    # == K-fold cross validation ==
    # scores = cross_validation.cross_val_score(clf, X, y, cv=5)
    # print "%d-fold cv, average accuracy %f" % (len(scores), scores.mean())

    # === Plot ===
    # Compute ROC curve and ROC area for each class


    # Plot ROC curve
    # plt.figure()
    # plt.plot(fpr["micro"], tpr["micro"],
    #          label='micro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc["micro"]))
    # for i in range(2):
    #     plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
    #                                    ''.format(i, roc_auc[i]))

    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Some extension of Receiver operating characteristic to multi-class')
    # plt.legend(loc="lower right")
    # plt.show()

    # === Serialize ===
    # s = pickle.dumps(clf)
    # with open(TRAIN_DATA_DST % "master", "w") as model_outfile:
    #   model_outfile.write(s)

    # joblib.dump(clf, TRAIN_DATA_DST % "master")
