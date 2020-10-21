import scipy.sparse as sp
import pandas as pd
from tqdm import tqdm
import pickle
from collections import defaultdict, OrderedDict
import numpy as np
import os


def pk_save(obj, file_path):
    return pickle.dump(obj, open(file_path, 'wb'))


def pk_load(file_path):
    if os.path.exists(file_path):
        return pickle.load(open(file_path, 'rb'))
    else:
        return None


# output_files
x_output_file = 'ind.decagon.allx'
graph_output_file = 'ind.decagon.graph'
# output graph
multi_graph = {}

voc = OrderedDict()
multi_graph_i = pk_load(graph_output_file)
for type, dict_ls in multi_graph_i.items():
    if type not in multi_graph.keys():
        multi_graph[type] = defaultdict(list)
    for drug_k, drug_ls in dict_ls.items():
        if drug_k not in voc.keys():
            voc[drug_k] = len(voc)
        for drug_v in drug_ls:
            if drug_v not in voc.keys():
                voc[drug_v] = len(voc)
            multi_graph[type][voc[drug_k]].append(voc[drug_v])

# save graph and node feature
pk_save(multi_graph, graph_output_file)

n_drugs = len(voc)
drug_feat = sp.identity(n_drugs)
pk_save(drug_feat, x_output_file)
