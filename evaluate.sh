#!/bin/bash

THIS_DIR=$(pwd)
cd ./data/SemEval-2015-task-13-v1.0/scorer
RESULT="$THIS_DIR/output/result.txt"
for file in $THIS_DIR/output/*.txt
do
    echo "$file" >> "$RESULT"
    java Scorer ../keys/gold_keys/EN/semeval-2015-task-13-en-WSD.key "$file" >> "$RESULT"
done
