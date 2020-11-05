#! /bin/sh

python driver_create_json.py 350 perturbed_langtranslation_rinehart_350.json ../data/Rinehart_10_uk.txt rinehart ../data/Christie_10_us.txt christie ../data/Doyle_10_us.txt doyle
python driver_create_json.py 1400 perturbed_langtranslation_rinehart_1400.json ../data/Rinehart_10_uk.txt rinehart ../data/Christie_10_us.txt christie ../data/Doyle_10_us.txt doyle
python driver_create_json.py 3500 perturbed_langtranslation_rinehart_3500.json ../data/Rinehart_10_uk.txt rinehart ../data/Christie_10_us.txt christie ../data/Doyle_10_us.txt doyle

python driver_create_json.py 350 perturbed_synonym_rinehart_350.json ../data/Rinehart_10_syn.txt rinehart ../data/Christie_10_syn.txt christie ../data/Doyle_10_syn.txt doyle
python driver_create_json.py 1400 perturbed_synonym_rinehart_1400.json ../data/Rinehart_10_syn.txt rinehart ../data/Christie_10_syn.txt christie ../data/Doyle_10_syn.txt doyle
python driver_create_json.py 3500 perturbed_synonym_rinehart_3500.json ../data/Rinehart_10_syn.txt rinehart ../data/Christie_10_syn.txt christie ../data/Doyle_10_syn.txt doyle
