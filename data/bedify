#!/usr/bin/python
import csv
import sys
import pdb


def convert_to_bed_entry(line):
    chrom, rest = line.split(":")
    rest = rest.strip()
    start, end = rest.split("-")
    start = start.replace(",", "")
    end = end.replace(",", "")
    return [chrom, start, end]


if __name__ == "__main__":
    rowcount = 0
    with open(sys.argv[1]) as fin, open(sys.argv[2], "wb") as fout:
        writer = csv.writer(fout, delimiter="\t", lineterminator='\n')
        for line in fin:
            try:
                writer.writerow(convert_to_bed_entry(line))
                rowcount += 1
            except:
                pass
    print "Wrote out %d rows" % rowcount
