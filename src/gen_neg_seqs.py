"""
Given a bedfile of positive examples, generates
a "<name>-pos.fa" and "<name>-neg.fa" files for
classifiers.

Positives are enhancer regions, aka
the original bedfile, and negatives are the randomly
shuffled regions.

The negative examples may not overlap
the positive enhancer regions, but may overlap one
another, though the chances are small as they are allowed
to be shuffled across the entire genome.
"""

import os
import sys
import tempfile
import pybedtools


def gen_neg_seqs(argv):
    """ Given a fatsa sequence, shuffles sequence around the genome
    and outputs the shuffled bedtools. """
    bedfilename = argv[0]
    shuffledfilename = argv[1]
    bt = pybedtools.BedTool(bedfilename)
    tempf = open("temp.bed", "wb")
    tempf.write(str(bt))
    tempf.close()
    if len(argv) > 2:
        genome = argv[2]
    else:
        genome = 'hg19'
    shuffled = bt.shuffle(genome=genome, excl=tempf.name)
    os.remove("temp.bed")

    out = open(shuffledfilename, "w")
    out.write(str(shuffled))
    out.close()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print "Usage: gen_neg_seqs.py positive_examples.bed vt-human <genome>"
        print "First arg is path to positive examples file"
        print "Second arg is out path to write to"
        print "[Optional] Third arg is genome to use. One of hg19|mm9"
        sys.exit(1)
    else:
        gen_neg_seqs(sys.argv[1:])
        print "Done"

