epigensML
=========

CS 273a: Predicting Tissue Specific Enhancer Activity from Epigenetic Marks and Sequence

## Authors

Nishith K., Kristin M., Jim Z. ([ataki](https://github.com/ataki))

## Summary

For reference, we're using the paper by Lee et al. in papers/2197 as a guide.
Their analysis uses only genome sequence features plus a set of general
sequence features in order to detect enhancer regions.

They use an SVM with hard boundary to separate +/- regions. Their SVM was
immune to kernel choice, which indicated linear separability across a 
high dimensional dataset (feature space of all possible kmers in regions of
interest).

## Description of Data



## Milestone

For the milestone, we have completed the current checklist:

- [x] Process Human FASTA data
- [x] Generate kmer count feature vector
- [x] Train SVM with default RBF kernel
- [x] Generate additional features from Lee et al paper
- [x] Generate point scatter plot using PCA of top features (like Fig 1 of Lee
  et al)
- [x] Generate precision/recall/false-positive curves (like Fig 2)
- [x] K-fold cross validation using sklearn

## Notes on Results

Initial results (sklearn's cv train/test split). Normalizing vector counts 
help to keep the average accuracy at >0.6.

- accuracy = ~0.61 (no feature sel, linear svm, 6mer features)
- roc = 0.65

VISTA only predicts enhancers at 11.5 days(? wks) after birth. They say
that enhancer activity can still be present at a later stage in development,
i.e. our "pos" labels for enhancer activity is only good for a snapshot at 
11.5 days and is not predictive of enhancer activity in general.

Using E-box and TAAT-cores didn't help. Will need to do some more hacking
on these features to see if they can be combined to be useful.

## Final

We shall brainstorm additional features to generate from the genomic sequence
which would help us. We will also apply more evaluation techniques from
standard machine learning, and take a look at alternative classifiers, as
well as attempt to make stonger predicitions (multiclass instead of binary).

