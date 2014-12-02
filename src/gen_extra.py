"""
Generates a list of row features using rows
from a fasta file downloaded from USCS
Genome Browser.

Outputs to a matrix of choice.
"""

import sys
import time
import numpy as np
from Bio import SeqIO
from features import *


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print """
        Arguments:
            First is fasta file,
            second is path to bedfile you want to intersect with
            third is path to write to
            fourth is the operation. Can be (tf|heart|limb|
        """
        sys.exit(1)

    print sys.argv
    inpath = sys.argv[1]
    tfpath = sys.argv[2]
    outpath = sys.argv[3]

    seqfile = [x for x in SeqIO.parse(inpath, "fasta")]
    labels = []
    total = len(seqfile)
    count = 0

    start = time.time()
    for x in seqfile:
        bedtool = extract_bed(x.description)
        label = extract_feat_enhancers(bedtool, tfpath, converted=True)
        labels.append(label)
        count += 1
        if count % 5 == 0:
            diff = time.time() - start
            print "%d out of %d processed, took %.2fs" % (count, total, diff)
            start = time.time()

    result = np.array(labels)
    result.tofile(outpath)
