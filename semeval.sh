#!/bin/bash
#$1 filename $2 fileid $3 basepath
filename=$1
fileid=$2
basepath=$3
perl $basepath/maxdiff_to_scale.pl $basepath/../benchmarks/semeval/Phase2Answers/Phase2Answers-$fileid.txt $basepath/../work/semeval-tmp/TurkerScaled-$fileid.txt
perl $basepath/score_scale.pl $basepath/../work/semeval-tmp/TurkerScaled-$fileid.txt $filename $basepath/../work/semeval-tmp/SpearmanRandomScaled-$fileid.txt
perl $basepath/scale_to_maxdiff.pl $basepath/../benchmarks/semeval/Phase2Questions/Phase2Questions-$fileid.txt $filename $basepath/../work/semeval-tmp/MaxDiff-$fileid.txt
perl $basepath/score_maxdiff.pl $basepath/../benchmarks/semeval/Phase2Answers/Phase2Answers-$fileid.txt $basepath/../work/semeval-tmp/MaxDiff-$fileid.txt $basepath/../work/semeval-tmp/MaxDiffFinal-$fileid.txt