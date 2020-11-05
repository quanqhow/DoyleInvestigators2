#! /bin/sh

python driver_synonyms.py 0 0.2 ../data/Christie_10.txt ../data/Christie_10_syn.txt ../training/doyle_50dim_350part.bin
python driver_synonyms.py 0 0.2 ../data/Doyle_10.txt ../data/Doyle_10_syn.txt ../training/doyle_50dim_350part.bin
python driver_synonyms.py 0 0.2 ../data/Rinehart_10.txt ../data/Rinehart_10_syn.txt ../training/doyle_50dim_350part.bin

python driver_synonyms.py 1 0.2 ../data/Christie_10.txt ../data/Christie_10_syn_tag.txt ../training/doyle_50dim_350part.bin
python driver_synonyms.py 1 0.2 ../data/Doyle_10.txt ../data/Doyle_10_syn_tag.txt ../training/doyle_50dim_350part.bin
python driver_synonyms.py 1 0.2 ../data/Rinehart_10.txt ../data/Rinehart_10_syn_tag.txt ../training/doyle_50dim_350part.bin
