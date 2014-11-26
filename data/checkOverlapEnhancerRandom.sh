cat bed/enh_es_200.bed bed/enh_es_800.bed bed/enh_fb.bed bed/enh_lb.bed bed/enh_mb.bed bed/enh_neuact_200.bed bed/enh_neuact_800.bed > all_enh_notbed

cat bed/random12000.bed bed/random4000.bed bed/random5240.bed > all_random_notbed

python bedify.py all_enh_notbed all_enh.bed
python bedify.py all_random_notbed all_random.bed

bedtools intersect -a all_enh.bed -b all_random.bed > possibleOverlap.bed

wc -l possibleOverlap.bed

rm -f all_enh_notbed all_random_notbed all_enh.bed all_random.bed possibleOverlap.bed
