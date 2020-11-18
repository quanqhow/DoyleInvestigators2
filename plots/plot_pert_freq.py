import json
import matplotlib.pyplot as plt
import numpy as np


def PF_plot(fn: str, author_name: str = None):
    with open(fn) as fd:
        data = json.load(fd)

    plt.bar(np.arange(len(data.keys())), [100 * data.get(key, 0) for key in data.keys()], align='center', fc='0.5')
    plt.xlabel("Document #")
    plt.ylabel('Perturbed Words (%)')
    if author_name:
        plt.title(f'Perturbation Frequency ({author_name})')
    else:
        plt.title('Perturbation Frequency')
    return plt.show()


if __name__ == '__main__':
    author_name = 'Rinehart'
    PF_plot('perturb_rate_synonym_rinehart_350.json', author_name)
    PF_plot('perturb_rate_synonym_rinehart_1400.json', author_name)
    PF_plot('perturb_rate_synonym_rinehart_3500.json', author_name)

    PF_plot('perturb_rate_langtranslation_rinehart_350.json', author_name)
    PF_plot('perturb_rate_langtranslation_rinehart_1400.json', author_name)
    PF_plot('perturb_rate_langtranslation_rinehart_3500.json', author_name)
