#!/bin/bash
echo Experiment 1 &&
python3 camembert_weighted.py '3' 'augm-1' 'cuda:0' > class-3_augm-1.log &&
echo Experiment 2 &&
python3 camembert_weighted.py '3' 'augm-2' 'cuda:0' > class-3_augm-2.log &&
echo Experiment 3 &&
python3 camembert_weighted.py '3' 'augm-3' 'cuda:0' > class-3_augm-3.log &&
echo Experiment 4 &&
python3 camembert_weighted.py '3' 'augm-4' 'cuda:0' > class-3_augm-4.log &&
echo Experiment 5 &&
python3 camembert_weighted.py '3' 'augm-5' 'cuda:0' > class-3_augm-5.log &&
echo Experiment 6 &&
python3 camembert_weighted.py '3' 'augm-6' 'cuda:0' > class-3_augm-6.log &&
echo Experiment 7 &&
python3 camembert_weighted.py '3' 'augm-7' 'cuda:0' > class-3_augm-7.log &&
echo Experiment 8 &&
python3 camembert_weighted.py '3' 'augm-8' 'cuda:0' > class-3_augm-8.log &&
echo Experiment 9 &&
python3 camembert_weighted.py '3' 'augm-9' 'cuda:0' > class-3_augm-9.log 
