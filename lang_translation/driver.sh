#! /bin/sh

python driver_translate.py uk ../data/Rinehart_10.txt ../data/Rinehart_10_uk.txt
python driver_translate.py us ../data/Doyle_10.txt ../data/Doyle_10_us.txt
python driver_translate.py us ../data/Christie_10.txt ../data/Christie_10_us.txt

python driver_translate.py uk ../data/Rinehart_10.txt ../data/Rinehart_10_uk_tag.txt 1
python driver_translate.py us ../data/Doyle_10.txt ../data/Doyle_10_us_tag.txt 1
python driver_translate.py us ../data/Christie_10.txt ../data/Christie_10_us_tag.txt 1
