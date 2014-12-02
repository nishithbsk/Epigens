"""
Generates a list of labels using rows
from a fasta file downloaded from USCS
Genome Browser.

Writes out to second arg
"""

import sys
import numpy as np
from Bio import SeqIO
from features import *


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print """
        Arguments:
            First is fasta file,
            second is path to tf binding sites,
            third is path to write to
        """
        sys.exit(1)

    inpath = sys.argv[1]
    tfpath = sys.argv[2]
    outpath = sys.argv[3]

    seqfile = SeqIO.parse(inpath, "fasta")
    labels = []
    for seq in seqfile:
        label = extra_feat_enhancers(seq, tfpath)
        labels.append(extract_bed(label))

    result = np.array(labels)
    result.save(outpath)
