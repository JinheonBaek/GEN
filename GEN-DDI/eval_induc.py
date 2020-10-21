import os
import time
import random
import argparse
import numpy as np

from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

from models import InducGEN

class Trainer(object):

    def __init__(self, args):

        super(Trainer, self).__init__()

        self.args = args
        self.exp_name = self.experiment_name(args)

        self.use_cuda = args.gpu >= 0 and torch.cuda.is_available()
        if self.use_cuda:
            torch.cuda.set_device(args.gpu)

        self.filtered_triplets, self.meta_train_task_triplets, self.meta_valid_task_triplets, self.meta_test_task_triplets, \
        self.meta_train_task_entity_to_triplets, self.meta_valid_task_entity_to_triplets, self.meta_test_task_entity_to_triplets \
            = utils.load_processed_data('./Dataset/processed_data/{}'.format(args.data))

        self.meta_task_entity = np.concatenate((list(self.meta_train_task_entity_to_triplets.keys()),
                                            list(self.meta_valid_task_entity_to_triplets.keys()),
                                            list(self.meta_test_task_entity_to_triplets.keys())))

        self.meta_task_test_entity = torch.LongTensor(np.array(list(self.meta_test_task_entity_to_triplets.keys())))

        self.load_pretrain_embedding(data = args.data, model = args.pre_train_model)
        self.load_model(model = args.model)

        if self.use_cuda:
            self.model.cuda()
            self.meta_task_test_entity = self.meta_task_test_entity.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.args.lr, weight_decay = self.args.weight_decay)

    def load_pretrain_embedding(self, data, model):

        self.embedding_size = int(self.args.pre_train_emb_size)

        if self.args.pre_train:
            entity_file_name = './Pretraining/{}/{}_entity.npy'.format(self.args.data, self.args.pre_train_model)
            
            self.pretrain_entity_embedding = torch.Tensor(np.load(entity_file_name))
            self.pretrain_relation_embedding = None

        else:
            self.pretrain_entity_embedding = None
            self.pretrain_relation_embedding = None

    def load_model(self, model):

        num_entities = 1861
        num_relations = 113

        if self.args.model == 'InducGEN':

            self.model = InducGEN(self.embedding_size, self.embedding_size, num_entities, num_relations,
                            args = self.args, entity_embedding = self.pretrain_entity_embedding, relation_embedding = self.pretrain_relation_embedding)
        else:

            raise ValueError("Model Name <{}> is Wrong".format(self.args.model))

        meta_task_entity = torch.LongTensor(self.meta_task_entity)
        self.model.entity_embedding.weight.data[meta_task_entity] = torch.zeros(len(meta_task_entity), self.embedding_size)

    def train(self):

        checkpoint = torch.load('{}/best_mrr_model.pth'.format(self.exp_name), map_location='cuda:{}'.format(args.gpu))
        self.model.load_state_dict(checkpoint['state_dict'])

        print("Using best epoch: {}, {}".format(checkpoint['epoch'], self.exp_name))

        # Final Test
        results = {}

        total_results, total_induc_results, total_trans_results = self.eval(eval_type='test')

        results['total_prs'] = total_results['pr']
        results['total_rocs'] = total_results['roc']
        results['total_p@1s'] = total_results['acc']

        results['total_induc_prs'] = total_induc_results['pr']
        results['total_induc_rocs'] = total_induc_results['roc']
        results['total_induc_p@1s'] = total_induc_results['acc']

        results['total_trans_prs'] = total_trans_results['pr']
        results['total_trans_rocs'] = total_trans_results['roc']
        results['total_trans_p@1s'] = total_trans_results['acc']

        tqdm.write("Total PR (filtered): {:.6f}".format(results['total_prs']))
        tqdm.write("Total ROC (filtered) {:.6f}".format(results['total_rocs']))
        tqdm.write("Total P@1s (filtered) {:.6f}".format(results['total_p@1s']))

        tqdm.write("Total Induc PR (filtered): {:.6f}".format(results['total_induc_prs']))
        tqdm.write("Total Induc ROC (filtered) {:.6f}".format(results['total_induc_rocs']))
        tqdm.write("Total Induc P@1s (filtered) {:.6f}".format(results['total_induc_p@1s']))
        
        tqdm.write("Total Trans PR (filtered): {:.6f}".format(results['total_trans_prs']))
        tqdm.write("Total Trans ROC (filtered) {:.6f}".format(results['total_trans_rocs']))
        tqdm.write("Total Trans P@1s (filtered) {:.6f}".format(results['total_trans_p@1s']))
        

    def eval(self, eval_type='test'):

        self.model.eval()

        if eval_type == 'valid':
            test_task_dict = self.meta_valid_task_entity_to_triplets
            test_task_pool = list(self.meta_valid_task_entity_to_triplets.keys())

        elif eval_type == 'test':
            test_task_dict = self.meta_test_task_entity_to_triplets
            test_task_pool = list(self.meta_test_task_entity_to_triplets.keys())

        else:
            raise ValueError("Eval Type <{}> is Wrong".format(eval_type))

        total_ys = []
        total_y_probs = []

        induc_ys = []
        induc_y_probs = []
        
        trans_ys = []
        trans_y_probs = []

        for task_entity in tqdm(test_task_pool):

            my_total_triplets = []
            my_induc_triplets = []
            my_trans_triplets = []

            task_triplets = test_task_dict[task_entity]
            task_triplets = np.array(task_triplets)
            task_heads, task_relations, task_tails = task_triplets.transpose()

            train_task_triplets = task_triplets[:self.args.few]
            test_task_triplets =  task_triplets[self.args.few:]

            if (len(task_triplets)) - self.args.few < 1:
                continue

            task_entity_embedding = self.model(task_entity, train_task_triplets, self.use_cuda)

            test_task_triplets = torch.LongTensor(test_task_triplets)
            if self.use_cuda:
                test_task_triplets = test_task_triplets.cuda()

            for test_triplet in test_task_triplets:

                is_trans = self.is_trans(test_task_pool, test_triplet)

                my_total_triplets.append(test_triplet)

                if is_trans:
                    my_trans_triplets.append(test_triplet)
                else:
                    my_induc_triplets.append(test_triplet)
            
            if len(my_total_triplets)>0:
                my_total_triplets = torch.stack(my_total_triplets, dim=0)
                y_prob, y = self.model.predict(task_entity, task_entity_embedding, my_total_triplets, target=None, use_cuda=self.use_cuda)
                total_y_probs.append(y_prob.detach().cpu())
                total_ys.append(y.detach().cpu())

            if len(my_induc_triplets)>0:
                my_induc_triplets = torch.stack(my_induc_triplets, dim=0)
                y_prob, y = self.model.predict(task_entity, task_entity_embedding, my_induc_triplets, target=None, use_cuda=self.use_cuda)
                induc_y_probs.append(y_prob.detach().cpu())
                induc_ys.append(y.detach().cpu())

            if len(my_trans_triplets)>0:
                my_trans_triplets = torch.stack(my_trans_triplets, dim=0)
                y_prob, y = self.model.predict(task_entity, task_entity_embedding, my_trans_triplets, target=None, use_cuda=self.use_cuda)
                trans_y_probs.append(y_prob.detach().cpu())
                trans_ys.append(y.detach().cpu())


        total_y_prob = torch.cat(total_y_probs, dim=0).detach().cpu().numpy()
        total_y = torch.cat(total_ys, dim=0).detach().cpu().numpy()
        total_results = utils.metric_report(total_y, total_y_prob)

        induc_y_prob = torch.cat(induc_y_probs, dim=0).detach().cpu().numpy()
        induc_y = torch.cat(induc_ys, dim=0).detach().cpu().numpy()
        induc_results = utils.metric_report(induc_y, induc_y_prob)

        trans_y_prob = torch.cat(trans_y_probs, dim=0).detach().cpu().numpy()
        trans_y = torch.cat(trans_ys, dim=0).detach().cpu().numpy()
        trans_results = utils.metric_report(trans_y, trans_y_prob)

        return total_results, induc_results, trans_results

    def is_trans(self, total_task_entity, test_triplet):

        is_trans = False

        if (test_triplet[0] in total_task_entity) and (test_triplet[2] in total_task_entity):

            is_trans = True

        return is_trans

    def experiment_name(self, args):

        exp_name = os.path.join('./checkpoints', self.args.exp_name)

        return exp_name


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Meta-KGNN')

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--debug", action='store_true')

    parser.add_argument("--exp-name", type=str, default='DeepDDI_Induc')
    parser.add_argument("--data", type=str, default='DeepDDI')

    parser.add_argument("--few", type=int, default=3)
    parser.add_argument("--n-epochs", type=int, default=10000)
    parser.add_argument("--evaluate-every", type=int, default=100)

    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", default=0.0, type=float)

    parser.add_argument("--grad-norm", type=float, default=1.0)
    parser.add_argument("--bases", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--pre-train", action='store_true')
    parser.add_argument("--fine-tune", action='store_true')

    parser.add_argument("--pre-train-model", type=str, default='DDI')
    parser.add_argument("--pre-train-emb-size", type=str, default='100')
    parser.add_argument("--model", type=str, default='TransGEN')

    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    trainer = Trainer(args)

    trainer.train()
    print(args)
