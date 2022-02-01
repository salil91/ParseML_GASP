#!/bin/bash

pwd;
echo 'Job Started: ml_train_test_split'
date;

echo 'Working directory:' $1
echo 'Percentage for Training Set:' $2

export PATH=/home/salil.bavdekar/.conda/envs/ai_gasp/bin:$PATH
python ml_train_test_split.py $1 $2

echo 'Done.'
date;
