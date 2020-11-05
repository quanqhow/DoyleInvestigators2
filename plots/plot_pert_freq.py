import json
import matplotlib.pyplot as plt
import numpy as np


def PF_plot(file):
    # input the jason file of data dictionary
    with open(file, 'r') as f:
        data = json.load(f)

    plt.bar(np.arange(len(data.keys())), [100*data.get(key, 0) for key in data.keys()], align='center', fc='0.5')
    plt.xlabel("Documents #")
    plt.ylabel('%Perturbed Words')
    plt.title('Perturbation Frequency (Author Name)') #update author name to your demand
    return plt.show()


if __name__ == '__main__':
	PF_plot('perturb_rate_synonym_rinehart_350.json')
	PF_plot('perturb_rate_synonym_rinehart_1400.json')
	PF_plot('perturb_rate_synonym_rinehart_3500.json')

	PF_plot('perturb_rate_langtranslation_rinehart_350.json')
	PF_plot('perturb_rate_langtranslation_rinehart_1400.json')
	PF_plot('perturb_rate_langtranslation_rinehart_3500.json')
