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
    if len(sys.argv) < 5:
        print """
        Arguments: <fasta> <bedfile> <outfile> <op>
            First is fasta file,
            second is path to bedfile you want to intersect with
            third is path to write to
            fourth is the operation. Can be (tf|<something else>)
        """
        sys.exit(1)

    inpath = sys.argv[1]
    bedpath = sys.argv[2]
    outpath = sys.argv[3]
    op = sys.argv[4]

    seqfile = [x for x in SeqIO.parse(inpath, "fasta")]
    labels = []
    total = len(seqfile)
    count = 0

    start = time.time()
    for x in seqfile:
        if op == "tf":
            # raw fasta file from USCS genome table, so have
            # to convert at this step first
            bedtool = description_to_bed(x.description)
            label = extract_feat_1(bedtool, bedpath, converted=True)
        else:
            # annotated fasta file, so can use seq_to_bed
            if op in x.description:
                print "found %s in %s" % (op, x.description)
                label = extract_feat_23(x.description, bedpath)
            else:
                label = np.zeros(3)
        labels.append(label)
        count += 1
        if count % 5 == 0:
            break
            diff = time.time() - start
            print "%d out of %d processed, took %.2fs" % (count, total, diff)
            start = time.time()

    result = np.array(labels)
    np.save(outpath, result)
