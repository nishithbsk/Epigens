"""
Feat. extraction for parts 1/2
"""
import re
import numpy as np
import itertools
import pybedtools
from Bio import SeqIO

GLOBAL_K = 6


# === Part 1: Kmer classifiers ===


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

    seqs = np.array(seqs)
    _labels = np.array(_labels)
    return (seqs, _labels)


def parse_fa_tissue(path, label_fn, filter_fn):
    """
    Given
        fasta file that represents positive labels
        label extracting fn given description

    Returns
        a pair of (sequence, label) numpy
        arrays. Useful for constructing X/y for training
    """
    fasta_file = open(path)
    human_fasta_seq = SeqIO.parse(fasta_file, 'fasta')

    seqs = []
    _labels = []

    for entry in human_fasta_seq:
        if filter_fn(entry.description):
            seqs.append(str(entry.seq).replace("n", "").lower())
            _labels.append(label_fn(entry.description))

    seqs = np.array(seqs)
    _labels = np.array(_labels)
    return (seqs, _labels)


def parse_fa_fine_grain(path, label_fn, filter_fn):
    """
    Given
        fasta file that represents positive labels
        label extracting fn given description

    Returns
        a pair of (sequence, label) numpy
        arrays. Useful for constructing X/y for training
    """
    fasta_file = open(path)
    human_fasta_seq = SeqIO.parse(fasta_file, 'fasta')
    seqs = []
    _labels = []

    for entry in human_fasta_seq:
        if filter_fn(entry.description):
            seqs.append(str(entry.seq).replace("n", "").lower())
            _labels.append(label_fn(entry.description))

    seqs = np.array(seqs)
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


# == Part 2: Additional features ==


def seq_to_bed(seq):
    description = seq[0].split("|")[1]
    chromName = description.split(":")[0]

    chromRange = description.split(":")[1]
    chromStart = chromRange.split("-")[0]
    chromEnd = chromRange.split("-")[1]

    bedLine = [chromName, chromStart, chromEnd]
    bedLine = '\t'.join(bedLine)
    return bedLine


def extract_extra_features_2(seq, TF_name):
    """ Given single sequence, returns
    a list of additional features as listed
    by Kristin. """
    extra_features = []

    bedline = seq_to_bed(seq)
    seqFile = pybedtools.BedTool(bedline, from_string=True)
    TF_binding_sites = pybedtools.BedTool(TF_name)
    intersections = TF_binding_sites.intersect(seqFile)

    TFDictionary = {'P300' : 0, 'TCF' : 0, 'TBF' : 0}
    for intersection in intersections:
        TF = intersection.fields[-1]

        if('P300' in TF):
            TFDictionary['P300'] = 1
        elif('TCF' in TF):
            TFDictionary['TCF'] = 1
        elif('TBF' in TF):
            TFDictionary['TBF'] = 1

    extra_features = TFDictionary.values()
    return np.array(extra_features)


def extract_extra_features_1(seq, heart):
    """ Given single sequence, returns
    a list of additional features as listed
    by Kristin. """
    extra_features = []
    bedline = seq_to_bed(seq)
    seqFile = pybedtools.BedTool(bedline)
    mnemonicsFile = pybedtools.BedTool('data/feature_bed/heart_mnemonics.bed')
    intersections = mnemonicsFile.intersect(seqFile)

    statesDictionary = {'Enh' : 0, 'EnhG' : 0, 'Het' : 0, 'TxWk' : 0}
    for intersection in intersections:
        state = intersection.fields[-1]

        if('Enh' in state):
            statesDictionary['Enh'] = 1
        elif('EnhG' in state):
            statesDictionary['EnhG'] = 1
        elif('Het' in state):
            statesDictionary['Het'] = 1
        elif('TxWk' in state):
            statesDictionary['TxWk'] = 1

    extra_features = statesDictionary.values()
    return np.array(extra_features)
