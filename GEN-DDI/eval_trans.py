import os
import time
import copy
import random
import argparse
import numpy as np

from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

from models import TransGEN

class Trainer(object):

    def __init__(self, args):

        super(Trainer, self).__init__()

        self.args = args
        self.exp_name = self.experiment_name(args)

        self.best_mrr = 0

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

        if self.args.model == 'TransGEN':

            self.model = TransGEN(self.embedding_size, self.embedding_size, num_entities, num_relations,
                            args = self.args, entity_embedding = self.pretrain_entity_embedding, relation_embedding = self.pretrain_relation_embedding)

        else:

            raise ValueError("Model Name <{}> is Wrong".format(self.args.model))

        meta_task_entity = torch.LongTensor(self.meta_task_entity)
        self.model.entity_embedding.weight.data[meta_task_entity] = torch.zeros(len(meta_task_entity), self.embedding_size)

    def train(self):

        checkpoint = torch.load('{}/best_mrr_model.pth'.format(self.exp_name), map_location='cuda:{}'.format(args.gpu))
        self.model.load_state_dict(checkpoint['state_dict'])

        print("Using best epoch: {}, {}".format(checkpoint['epoch'], self.exp_name))

        # Test Code

        eval_types = {
            'normal': True,
            'mc_score': True
        }

        results = {}

        if eval_types['normal']:

            tqdm.write("Results about Normal (Mean) Inference")

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
        
        if eval_types['mc_score']:

            tqdm.write("Results about MC score inference")

            total_results, total_induc_results, total_trans_results = self.mc_score_inference(eval_type='test')

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

        total_task_entity = []
        total_task_entity_embedding = []
        total_train_task_triplets = []
        total_test_task_triplets = []
        total_test_task_triplets_dict = dict()

        for task_entity in tqdm(test_task_pool):

            task_triplets = test_task_dict[task_entity]
            task_triplets = np.array(task_triplets)
            task_heads, task_relations, task_tails = task_triplets.transpose()

            train_task_triplets = task_triplets[:self.args.few]
            test_task_triplets =  task_triplets[self.args.few:]

            if (len(task_triplets)) - self.args.few < 1:
                continue

            # Train (Inductive)
            task_entity_embedding = self.model(task_entity, train_task_triplets, use_cuda = self.use_cuda, is_trans = False)
            total_task_entity.append(task_entity)
            total_task_entity_embedding.append(task_entity_embedding)
            total_train_task_triplets.extend(train_task_triplets)
            total_test_task_triplets.extend(test_task_triplets)
            total_test_task_triplets_dict[task_entity] = torch.LongTensor(test_task_triplets)

        # Train (Transductive)
        total_task_entity = np.array(total_task_entity)
        total_task_entity_embedding = torch.cat(total_task_entity_embedding).view(-1, self.embedding_size)
        total_train_task_triplets = np.array(total_train_task_triplets)
        total_test_task_triplets = torch.LongTensor(total_test_task_triplets)

        task_entity_embeddings, _, _ = self.model(total_task_entity, total_train_task_triplets, use_cuda = self.use_cuda, is_trans = True, total_unseen_entity_embedding = total_task_entity_embedding)

        # Test
        total_task_entity = torch.from_numpy(total_task_entity)

        if self.use_cuda:
            total_task_entity = total_task_entity.cuda()

        my_total_triplets = []
        my_induc_triplets = []
        my_trans_triplets = []

        for task_entity, test_triplets in total_test_task_triplets_dict.items():

            if self.use_cuda:
                device = torch.device('cuda')
                test_triplets = test_triplets.cuda()

            for test_triplet in test_triplets:

                is_trans = self.is_trans(total_task_entity, test_triplet)

                my_total_triplets.append(test_triplet)

                if is_trans:
                    my_trans_triplets.append(test_triplet)
                else:
                    my_induc_triplets.append(test_triplet)
        
        my_total_triplets = torch.stack(my_total_triplets, dim=0)
        y_prob, y = self.model.predict(total_task_entity, task_entity_embeddings, my_total_triplets, target=None, use_cuda=self.use_cuda)
        y_prob = y_prob.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        total_results = utils.metric_report(y, y_prob)
        
        my_induc_triplets = torch.stack(my_induc_triplets, dim=0)
        y_prob, y = self.model.predict(total_task_entity, task_entity_embeddings, my_induc_triplets, target=None, use_cuda=self.use_cuda)
        y_prob = y_prob.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        total_induc_results = utils.metric_report(y, y_prob)
        
        my_trans_triplets = torch.stack(my_trans_triplets, dim=0)
        y_prob, y = self.model.predict(total_task_entity, task_entity_embeddings, my_trans_triplets, target=None, use_cuda=self.use_cuda)
        y_prob = y_prob.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        total_trans_results = utils.metric_report(y, y_prob)
        
        return total_results, total_induc_results, total_trans_results

    def mc_score_inference(self, eval_type='test'):

        self.model.eval()

        if eval_type == 'valid':
            test_task_dict = self.meta_valid_task_entity_to_triplets
            test_task_pool = list(self.meta_valid_task_entity_to_triplets.keys())

        elif eval_type == 'test':
            test_task_dict = self.meta_test_task_entity_to_triplets
            test_task_pool = list(self.meta_test_task_entity_to_triplets.keys())

        else:
            raise ValueError("Eval Type <{}> is Wrong".format(eval_type))

        total_task_entity = []
        total_task_entity_embeddings = []
        total_train_task_triplets = []
        total_test_task_triplets = []
        total_test_task_triplets_dict = dict()

        for task_entity in tqdm(test_task_pool):

            task_triplets = test_task_dict[task_entity]
            task_triplets = np.array(task_triplets)
            task_heads, task_relations, task_tails = task_triplets.transpose()

            train_task_triplets = task_triplets[:self.args.few]
            test_task_triplets =  task_triplets[self.args.few:]

            if (len(task_triplets)) - self.args.few < 1:
                continue

            # Train (Inductive)
            task_entity_embedding = torch.cat([self.model(task_entity, train_task_triplets, use_cuda = self.use_cuda, is_trans = False) for _ in range(self.args.mc_times)]).view(-1, self.embedding_size)

            total_task_entity.append(task_entity)
            total_task_entity_embeddings.append(task_entity_embedding)
            total_train_task_triplets.extend(train_task_triplets)
            total_test_task_triplets.extend(test_task_triplets)
            total_test_task_triplets_dict[task_entity] = torch.LongTensor(test_task_triplets)

        # Train (Transductive)
        total_task_entity = np.array(total_task_entity)
        total_task_entity_embeddings = torch.cat(total_task_entity_embeddings).view(-1, self.args.mc_times, self.embedding_size)
        total_train_task_triplets = np.array(total_train_task_triplets)
        total_test_task_triplets = torch.LongTensor(total_test_task_triplets)

        self.model.train()

        task_entity_embeddings = torch.cat([
            self.model(
                total_task_entity, total_train_task_triplets, use_cuda = self.use_cuda, is_trans = True, total_unseen_entity_embedding = total_task_entity_embeddings[:, i]
            )[0] for i in range(self.args.mc_times)
        ]).view(self.args.mc_times, -1, self.embedding_size)

        # Test
        total_task_entity = torch.from_numpy(total_task_entity)

        if self.use_cuda:
            total_task_entity = total_task_entity.cuda()

        my_total_triplets = []
        my_induc_triplets = []
        my_trans_triplets = []

        for task_entity, test_triplets in total_test_task_triplets_dict.items():

            if self.use_cuda:
                device = torch.device('cuda')
                test_triplets = test_triplets.cuda()

            for test_triplet in test_triplets:

                is_trans = self.is_trans(total_task_entity, test_triplet)

                my_total_triplets.append(test_triplet)

                if is_trans:
                    my_trans_triplets.append(test_triplet)
                else:
                    my_induc_triplets.append(test_triplet)
        
        
        my_total_triplets = torch.stack(my_total_triplets, dim=0)

        for mc_index in range(self.args.mc_times):

            y_prob, y = self.model.predict(total_task_entity, task_entity_embeddings[mc_index], my_total_triplets, target=None, use_cuda=self.use_cuda)

            if mc_index == 0:
                y_prob_mean = y_prob
            else:
                y_prob_mean += y_prob

        y_prob_mean = y_prob_mean / self.args.mc_times

        y_prob = y_prob_mean.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        total_results = utils.metric_report(y, y_prob)
        
        
        my_induc_triplets = torch.stack(my_induc_triplets, dim=0)

        for mc_index in range(self.args.mc_times):

            y_prob, y = self.model.predict(total_task_entity, task_entity_embeddings[mc_index], my_induc_triplets, target=None, use_cuda=self.use_cuda)

            if mc_index == 0:
                y_prob_mean = y_prob
            else:
                y_prob_mean += y_prob

        y_prob_mean = y_prob_mean / self.args.mc_times

        y_prob = y_prob_mean.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        total_induc_results = utils.metric_report(y, y_prob)
        

        my_trans_triplets = torch.stack(my_trans_triplets, dim=0)

        for mc_index in range(self.args.mc_times):

            y_prob, y = self.model.predict(total_task_entity, task_entity_embeddings[mc_index], my_trans_triplets, target=None, use_cuda=self.use_cuda)

            if mc_index == 0:
                y_prob_mean = y_prob
            else:
                y_prob_mean += y_prob

        y_prob_mean = y_prob_mean / self.args.mc_times

        y_prob = y_prob_mean.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        total_trans_results = utils.metric_report(y, y_prob)
        
        return total_results, total_induc_results, total_trans_results

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

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action='store_true')

    parser.add_argument("--exp-name", type=str, default='DeepDDI_Trans')
    parser.add_argument("--data", type=str, default='DeepDDI')

    parser.add_argument("--few", type=int, default=3)
    parser.add_argument("--n-epochs", type=int, default=5000)
    parser.add_argument("--evaluate-every", type=int, default=100)

    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", default=0.0, type=float)

    parser.add_argument("--grad-norm", type=float, default=1.0)
    parser.add_argument("--bases", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--pre-train", action='store_true')
    parser.add_argument("--fine-tune", action='store_true')

    parser.add_argument("--pre-train-model", type=str, default='DDI')
    parser.add_argument("--pre-train-emb-size", type=str, default='100')
    parser.add_argument("--model", type=str, default='TransGEN')

    parser.add_argument("--mc-times", type=int, default=10)

    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    trainer = Trainer(args)

    trainer.train()
    print(args)
