"""
Entry point. Load, extract counts of kmers
as features,trains an assembly of classifiers, and
serializes outputs to files.
"""
import pandas as pd
import numpy as np

from Bio import SeqIO
from sklearn import svm

# import pickle
from sklearn.externals import joblib

# Config
VISTA_TABLE_SRC = "data/vistaTable20141113.txt"

FASTA_HUMAN_SRC = "data/humanRegions.fasta"

TRAIN_DATA_DST = "out/%s.train"

GLOBAL_K = 7


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

    # TODO Should we be using "lower" here? What does it mean
    # for a FASTA letter to be capitalized?
    for x in human_fasta_seq:
        _named_sequences.append((x.id, str(x.seq).lower()))
        _named_descriptions.append((x.id, lfd(x.description)))

    print "--> num sequences read from %s: %d" % (path, len(_named_sequences))
    return (_named_sequences, _named_descriptions)


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


def get_kmers_index_lookup(examples):
    """ Given list of examples, builds the index
    mapping of kmers """
    all_seqs = [x[1] for x in examples]
    all_kmers = list(reduce(set_kmers_reducer, all_seqs, set()))
    return make_index_dict_from_list(all_kmers)


def get_XY(examples, labels_mapping, kmer_index):
    all_seqs = [x[1] for x in examples]
    X = np.vstack([get_kmer_counts(x, kmer_index) for x in all_seqs])
    y = np.array([x[1] for x in labels_mapping])
    print "train. matrix dims (X): ", X.shape
    print "num labels (y): ", len(y)
    print "------------------------------------"
    return (X, y)


# Load / train

examples, labels_mapping = load_named_seq(FASTA_HUMAN_SRC)
kmers_index = get_kmers_index_lookup(examples)
X, y = get_XY(examples, train_labels, kmers_index)

test_examples = examples[cutoff:]
test_labels = labels_mapping[cutoff:]
testX, testy = get_XY(test_examples, test_labels, kmers_index)

results = clf.predict(testX)
num_same = len([x for i, x in enumerate(results) if x == testy[i]])
accuracy = 1.0 * num_same / len(results)


# TODO Plot data, draw, svm so we have a rough approximation

# Serialize output to string

# s = pickle.dumps(clf)
# with open(TRAIN_DATA_DST % "master", "w") as model_outfile:
#   model_outfile.write(s)

joblib.dump(clf, TRAIN_DATA_DST % "master")
