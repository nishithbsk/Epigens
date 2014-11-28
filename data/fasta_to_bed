#!/usr/bin/python

import csv
import os
from Bio import SeqIO


if __name__ == "__main__":
    rowcount = 0
    for fpath in os.listdir("fasta"):
        if fpath.endswith(".fa"):
            fname, ext = fpath.split(".")
            bedfile = "bed/%s.bed" % fname
            print "Outputting %s to %s" % (("fasta/" + fname, bedfile))
            with open(os.path.join("fasta", fpath)) as f, open(bedfile, "w") as out:
                writer = csv.writer(out, delimiter="\t", lineterminator="\n")
                fasta_file = SeqIO.parse(f, 'fasta')
                for entry in fasta_file:
                    writer.writerow(entry.id.split("."))
                    rowcount += 1
                print "%d rows written" % rowcount
