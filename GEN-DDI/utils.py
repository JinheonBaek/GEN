import os
import math
import time
import copy
import pickle
import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, average_precision_score


#################################
#   Data Load and Pre-process   #
#################################

def load_data(file_path):
    '''
        argument:
            file_path: ./Dataset/raw_data/FB15k-237

        return:
            entity2id, relation2id, train_triplets, valid_triplets, test_triplets
    '''

    print("load data from {}".format(file_path))

    with open(os.path.join(file_path, 'entity2id.txt')) as f:
        entity2id = dict()

        for line in f:
            entity, eid = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(file_path, 'relation2id.txt')) as f:
        relation2id = dict()

        for line in f:
            relation, rid = line.strip().split('\t')
            relation2id[relation] = int(rid)

    train_triplets = read_triplets(os.path.join(file_path, 'train.txt'), entity2id, relation2id)
    valid_triplets = read_triplets(os.path.join(file_path, 'valid.txt'), entity2id, relation2id)
    test_triplets = read_triplets(os.path.join(file_path, 'test.txt'), entity2id, relation2id)

    print('num_entity: {}'.format(len(entity2id)))
    print('num_relation: {}'.format(len(relation2id)))
    print('num_train_triples: {}'.format(len(train_triplets)))
    print('num_valid_triples: {}'.format(len(valid_triplets)))
    print('num_test_triples: {}'.format(len(test_triplets)))

    return entity2id, relation2id, train_triplets, valid_triplets, test_triplets

def read_triplets(file_path, entity2id, relation2id):

    triplets = []

    with open(file_path) as f:
        for line in f:
            head, relation, tail = line.strip().split('\t')
            triplets.append((entity2id[head], relation2id[relation], entity2id[tail]))

    return np.array(triplets)

def load_processed_data(file_path):

    with open(os.path.join(file_path, 'filtered_triplets.pickle'), 'rb') as f:
        filtered_triplets = pickle.load(f)

    with open(os.path.join(file_path, 'meta_train_task_triplets.pickle'), 'rb') as f:
        meta_train_task_triplets = pickle.load(f)

    with open(os.path.join(file_path, 'meta_valid_task_triplets.pickle'), 'rb') as f:
        meta_valid_task_triplets = pickle.load(f)

    with open(os.path.join(file_path, 'meta_test_task_triplets.pickle'), 'rb') as f:
        meta_test_task_triplets = pickle.load(f)

    with open(os.path.join(file_path, 'meta_train_task_entity_to_triplets.pickle'), 'rb') as f:
        meta_train_task_entity_to_triplets = pickle.load(f)

    with open(os.path.join(file_path, 'meta_valid_task_entity_to_triplets.pickle'), 'rb') as f:
        meta_valid_task_entity_to_triplets = pickle.load(f)

    with open(os.path.join(file_path, 'meta_test_task_entity_to_triplets.pickle'), 'rb') as f:
        meta_test_task_entity_to_triplets = pickle.load(f)

    return filtered_triplets, meta_train_task_triplets, meta_valid_task_triplets, meta_test_task_triplets, \
            meta_train_task_entity_to_triplets, meta_valid_task_entity_to_triplets, meta_test_task_entity_to_triplets


#################
#   Cal Score   #
#################

def metric_report(y, y_prob):
    rocs = []
    prs = []

    ks = [1]
    pr_score_at_ks = []
    for k in ks:
        pr_at_k = []
        for i in range(y_prob.shape[0]):
            # forloop samples
            y_prob_index_topk = np.argsort(y_prob[i])[::-1][:k]
            inter = set(y_prob_index_topk) & set(y[i].nonzero()[0])
            pr_ith = len(inter) / k
            pr_at_k.append(pr_ith)
        pr_score_at_k = np.mean(pr_at_k)
        pr_score_at_ks.append(pr_score_at_k)

    for i in range(y.shape[1]):
        if(sum(y[:, i]) < 1):
            continue
        try:
            roc = roc_auc_score(y[:, i], y_prob[:, i])
            rocs.append(roc)
        except ValueError:
            print(len(y[:, i]))

        prauc = average_precision_score(y[:, i], y_prob[:, i])
        prs.append(prauc)

    roc_auc = sum(rocs)/len(rocs)
    pr_auc = sum(prs)/len(prs)

    return {
        'pr': pr_auc,
        'roc': roc_auc,
        'acc': pr_score_at_ks[0]
    }
