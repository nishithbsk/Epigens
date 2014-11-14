epigensML
=========

CS 273a: Predicting Tissue Specific Enhancer Activity from Epigenetic Marks and Sequence

## Authors

Nishith K., Kristin M., Jim Z. (ataki[ataki](https://github.com/ataki))

## Summary

For reference, we're using the paper by Lee et al. in papers/2197 as a guide.
Their analysis uses only genome sequence features plus a set of general
sequence features in order to detect enhancer regions.

They use an SVM with hard boundary to separate +/- regions. Their SVM was
immune to kernel choice, which indicated linear separability across a 
high dimensional dataset (feature space of all possible kmers in regions of
interest).

## Milestone

For the milestone, we have the current checklist:

- [x] Process Human FASTA data
- [x] Generate kmer count feature vector
- [x] Train SVM with default RBF kernel
- [ ] Generate additional features from Lee et al paper
- [ ] Generate point scatter plot using PCA of top features (like Fig 1 of Lee
  et al)
- [ ] Generate precision/recall/false-positive curves (like Fig 2)
- [ ] K-fold cross validation using sklearn

## Final

We shall brainstorm additional features to generate from the genomic sequence
which would help us. We will also apply more evaluation techniques from
standard machine learning, and take a look at alternative classifiers, as
well as attempt to make stonger predicitions (multiclass instead of binary).

