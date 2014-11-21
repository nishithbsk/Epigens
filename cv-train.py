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

from Bio import SeqIO
from sklearn import svm
from sklearn import cross_validation

# import pickle
from sklearn.externals import joblib

# Config
VISTA_TABLE_SRC = "data/vistaTable20141113.txt"

FASTA_HUMAN_SRC = "data/humanRegions.fasta"

TRAIN_DATA_DST = "out/%s.train"

POS_DATASET = "papers/pos.fa"

NEG_DATASET = "papers/neg.fa"

GLOBAL_K = 6


# Parse
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
    """ Returns two lists of tuples. The first is (id<int>, seq<str>),
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


def normalize(v):
    """ Returns normalized v with length 1 """
    return v / np.linalg.norm(v)


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
    return normalize(row)


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


# feature vector index :=> kmer string
kmers_index = get_kmers_index_lookup()


# === Train on experimental dataset ===
# pos_seq, pos_lmap = parse_fa(POS_DATASET, 1)
# neg_seq, neg_lmap = parse_fa(NEG_DATASET, -1)

# train_ex = pos_seq + neg_seq
# train_labels = pos_lmap + neg_lmap
# X_train, y_train = get_XY(train_ex, train_labels, kmers_index)

# # Add e-box and taat core cols
# X_train = np.hstack((
#     X_train,
#     get_Ebox_col(train_ex),
#     get_TAAT_core_col(train_ex)
# ))

# clf = svm.SVC(kernel='rbf', C=1)
# clf.fit(X_train, y_train)

# === Test on our own ===
examples, labels, locations = load_named_seq(FASTA_HUMAN_SRC)
X, y = get_XY(test_ex, test_labels, kmers_index)

X = np.hstack((
    X,
    get_Ebox_col(examples),
    get_TAAT_core_col(examples)
))

X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    X, y, test_size=0.1, random_state=0)

clf = svm.SVC(kernel='linear', C=1)
clf.fit(X, y)
score = clf.score(X, y)


# == K-fold cross validation ==
# scores = cross_validation.cross_val_score(clf, X, y, cv=5)
# print "%d-fold cv, average accuracy %f" % (len(scores), scores.mean())

# === Plot ===
# TODO

# === Serialize ===
# s = pickle.dumps(clf)
# with open(TRAIN_DATA_DST % "master", "w") as model_outfile:
#   model_outfile.write(s)

# joblib.dump(clf, TRAIN_DATA_DST % "master")
