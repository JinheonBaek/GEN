import os
import time
import math
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

        self.best_mrr = 0

        self.use_cuda = args.gpu >= 0 and torch.cuda.is_available()
        if self.use_cuda:
            torch.cuda.set_device(args.gpu)

        self.entity2id, self.relation2id, self.train_triplets, self.valid_triplets, self.test_triplets = utils.load_data('./Dataset/raw_data/{}'.format(args.data))
        self.filtered_triplets, self.meta_train_task_triplets, self.meta_valid_task_triplets, self.meta_test_task_triplets, \
        self.meta_train_task_entity_to_triplets, self.meta_valid_task_entity_to_triplets, self.meta_test_task_entity_to_triplets \
            = utils.load_processed_data('./Dataset/processed_data/{}'.format(args.data))

        self.all_triplets = torch.LongTensor(np.concatenate((
            self.train_triplets, self.valid_triplets, self.test_triplets
        )))

        self.meta_task_entity = np.concatenate((list(self.meta_valid_task_entity_to_triplets.keys()),
                                            list(self.meta_test_task_entity_to_triplets.keys())))

        self.entities_list = np.delete(np.arange(len(self.entity2id)), self.meta_task_entity)

        self.load_pretrain_embedding()
        self.load_model()

        if self.use_cuda:
            self.model.cuda()
            self.all_triplets = self.all_triplets.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.args.lr, weight_decay = self.args.weight_decay)

    def load_pretrain_embedding(self):

        self.embedding_size = int(self.args.pre_train_emb_size)

        if self.args.pre_train:

            pretrain_model_path = './Pretraining/{}'.format(self.args.data)

            entity_file_name = os.path.join(pretrain_model_path, '{}_entity_{}.npy'.format(self.args.pre_train_model, self.embedding_size))
            relation_file_name = os.path.join(pretrain_model_path, '{}_relation_{}.npy'.format(self.args.pre_train_model, self.embedding_size))

            self.pretrain_entity_embedding = torch.Tensor(np.load(entity_file_name))
            self.pretrain_relation_embedding = torch.Tensor(np.load(relation_file_name))

        else:

            self.pretrain_entity_embedding = None
            self.pretrain_relation_embedding = None

    def load_model(self):

        if self.args.model == 'InducGEN':

            self.model = InducGEN(self.embedding_size, self.embedding_size, len(self.entity2id), len(self.relation2id),
                            args = self.args, entity_embedding = self.pretrain_entity_embedding, relation_embedding = self.pretrain_relation_embedding)

        else:

            raise ValueError("Model Name <{}> is Wrong".format(self.args.model))

        meta_task_entity = torch.LongTensor(self.meta_task_entity)
        self.model.entity_embedding.weight.data[meta_task_entity] = torch.zeros(len(meta_task_entity), self.embedding_size)

    # Meta-Learning for Long-Tail Tasks
    def cal_train_few(self, epoch):

        if self.args.model_tail == 'log':

            for i in range(self.args.max_few):

                if epoch < (self.args.n_epochs / (2 ** i)):

                    continue

                return max(min(self.args.max_few, self.args.few + i - 1), self.args.few)

            return self.args.max_few

        else:

            return self.args.few

    def train(self):

        for epoch in trange(0, (self.args.n_epochs + 1), desc = 'Train Epochs', position = 0):

            # Meta-Train
            self.model.train()
            
            train_task_pool = list(self.meta_train_task_entity_to_triplets.keys())
            random.shuffle(train_task_pool)            

            total_loss = 0
            
            train_few = self.cal_train_few(epoch)

            for unseen_entity in train_task_pool[:self.args.num_train_entity]:

                triplets = self.meta_train_task_entity_to_triplets[unseen_entity]
                random.shuffle(triplets)
                triplets = np.array(triplets)
                heads, relations, tails = triplets.transpose()

                train_triplets = triplets[:train_few]
                test_triplets = triplets[train_few:]

                if (len(triplets)) - train_few < 1:
                    continue

                entities_list = self.entities_list
                false_candidates = np.array(list(set(entities_list) - set(np.concatenate((heads, tails)))))
                false_entities = np.random.choice(false_candidates, size=(len(triplets) - train_few) * self.args.negative_sample)

                pos_samples = test_triplets
                neg_samples = np.tile(pos_samples, (self.args.negative_sample, 1))
                neg_samples[neg_samples[:, 0] == unseen_entity, 2] = false_entities[neg_samples[:, 0] == unseen_entity]
                neg_samples[neg_samples[:, 2] == unseen_entity, 0] = false_entities[neg_samples[:, 2] == unseen_entity]

                samples = np.concatenate((pos_samples, neg_samples))
                labels = np.zeros(len(samples), dtype=np.float32)
                labels[:len(pos_samples)] = 1

                samples = torch.LongTensor(samples)
                labels = torch.LongTensor(labels)

                if self.use_cuda:
                    samples = samples.cuda()
                    labels = labels.cuda()

                # Train
                unseen_entity_embedding = self.model(unseen_entity, train_triplets, self.use_cuda)

                # Test
                loss = self.model.score_loss(unseen_entity, unseen_entity_embedding, samples, target=labels, use_cuda=self.use_cuda)

                if total_loss == 0:
                    total_loss = loss
                else:
                    total_loss += loss

            total_loss = total_loss / self.args.num_train_entity

            # Test Update
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_norm)
            self.optimizer.step()

            # Meta-Valid
            if epoch % self.args.evaluate_every == 0:

                tqdm.write("Epochs-{}, Loss-{}".format(epoch, total_loss))

                with torch.no_grad():

                    results = self.eval(eval_type='valid')

                    mrr = results['total_mrr']
                    tqdm.write("Total MRR (filtered): {:.6f}".format(results['total_mrr']))
                    tqdm.write("Total Hits (filtered) @ {}: {:.6f}".format(1, results['total_hits@1']))
                    tqdm.write("Total Hits (filtered) @ {}: {:.6f}".format(3, results['total_hits@3']))
                    tqdm.write("Total Hits (filtered) @ {}: {:.6f}".format(10, results['total_hits@10']))

                if mrr > self.best_mrr:
                    self.best_mrr = mrr
                    torch.save({'state_dict': self.model.state_dict(), 'epoch': epoch},
                            './checkpoints/{}/best_mrr_model.pth'.format(self.exp_name))

        checkpoint = torch.load('./checkpoints/{}/best_mrr_model.pth'.format(self.exp_name))
        self.model.load_state_dict(checkpoint['state_dict'])

        print("Using best epoch: {}, {}".format(checkpoint['epoch'], self.exp_name))

        # Meta-Test
        with torch.no_grad():
            results = self.eval(eval_type='test')

        tqdm.write("Total MRR (filtered): {:.6f}".format(results['total_mrr']))
        tqdm.write("Total Hits (filtered) @ {}: {:.6f}".format(1, results['total_hits@1']))
        tqdm.write("Total Hits (filtered) @ {}: {:.6f}".format(3, results['total_hits@3']))
        tqdm.write("Total Hits (filtered) @ {}: {:.6f}".format(10, results['total_hits@10']))

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

        total_ranks = []
        total_subject_ranks = []
        total_object_ranks = []

        for unseen_entity in test_task_pool:

            triplets = test_task_dict[unseen_entity]
            triplets = np.array(triplets)
            heads, relations, tails = triplets.transpose()

            train_triplets = triplets[:self.args.few]
            test_triplets = triplets[self.args.few:]

            if (len(triplets)) - self.args.few < 1:
                continue

            unseen_entity_embedding = self.model(unseen_entity, train_triplets, self.use_cuda)

            test_triplets = torch.LongTensor(test_triplets)
            if self.use_cuda:
                test_triplets = test_triplets.cuda()

            ranks, ranks_s, ranks_o = utils.calc_induc_mrr(unseen_entity, unseen_entity_embedding, self.model.entity_embedding.weight, self.model.relation_embedding, test_triplets, self.all_triplets, self.use_cuda, score_function=self.args.score_function)

            if len(ranks_s) != 0:
                total_subject_ranks.append(ranks_s)

            if len(ranks_o) != 0:
                total_object_ranks.append(ranks_o)

            total_ranks.append(ranks)

        results = {}
                        
        # Subject
        total_subject_ranks = torch.cat(total_subject_ranks)
        total_subject_ranks += 1

        results['subject_ranks'] = total_subject_ranks
        results['subject_mrr'] = torch.mean(1.0 / total_subject_ranks.float()).item()

        for hit in [1, 3, 10]:
            avg_count = torch.mean((total_subject_ranks <= hit).float())
            results['subject_hits@{}'.format(hit)] = avg_count.item()

        # Object
        total_object_ranks = torch.cat(total_object_ranks)
        total_object_ranks += 1

        results['object_ranks'] = total_object_ranks
        results['object_mrr'] = torch.mean(1.0 / total_object_ranks.float()).item()

        for hit in [1, 3, 10]:
            avg_count = torch.mean((total_object_ranks <= hit).float())
            results['object_hits@{}'.format(hit)] = avg_count.item()

        # Total
        total_ranks = torch.cat(total_ranks)
        total_ranks += 1

        results['total_ranks'] = total_ranks
        results['total_mrr'] = torch.mean(1.0 / total_ranks.float()).item()

        for hit in [1, 3, 10]:
            avg_count = torch.mean((total_ranks <= hit).float())
            results['total_hits@{}'.format(hit)] = avg_count.item()

        return results

    def experiment_name(self, args):

        ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())

        exp_name = str()
        exp_name += "Model={}_".format(args.model)
        exp_name += "Bases={}_".format(args.bases)
        exp_name += "DO={}_".format(args.dropout)
        exp_name += "NS={}_".format(args.negative_sample)
        exp_name += "Margin={}_".format(args.margin)
        exp_name += "Few={}_".format(args.few)
        exp_name += "LR={}_".format(args.lr)
        exp_name += "WD={}_".format(args.weight_decay)
        exp_name += "GN={}_".format(args.grad_norm)
        exp_name += "PT={}_".format(args.pre_train)
        exp_name += "PTM={}_".format(args.pre_train_model)
        exp_name += "PTES={}_".format(args.pre_train_emb_size)
        exp_name += "NE={}_".format(args.num_train_entity)
        exp_name += "FT={}_".format(args.fine_tune)
        exp_name += "SF={}_".format(args.score_function)
        exp_name += "TS={}".format(ts)

        if not args.debug:
            if not(os.path.isdir('./checkpoints/{}'.format(exp_name))):
                os.makedirs(os.path.join('./checkpoints/{}'.format(exp_name)))

            print("Make Directory {} in a Checkpoints Folder".format(exp_name))

        return exp_name

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GEN')
    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action='store_true')

    parser.add_argument("--data", type=str, default='FB15k-237')
    parser.add_argument("--negative-sample", type=int, default=1)

    parser.add_argument("--few", type=int, default=3)
    parser.add_argument("--n-epochs", type=int, default=6000)
    parser.add_argument("--evaluate-every", type=int, default=100)
    
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    
    parser.add_argument("--grad-norm", type=float, default=1.0)
    parser.add_argument("--bases", type=int, default=100)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--pre-train", action='store_true')
    parser.add_argument("--fine-tune", action='store_true')

    parser.add_argument("--pre-train-model", type=str, default='DistMult')
    parser.add_argument("--pre-train-emb-size", type=str, default='100')
    parser.add_argument("--num-train-entity", type=int, default=100)
    parser.add_argument("--model", type=str)
    parser.add_argument("--score-function", type=str, default='DistMult')

    parser.add_argument("--model-tail", type=str, default='None')
    parser.add_argument("--max-few", type=int, default=10)

    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    trainer = Trainer(args)

    trainer.train()
    print(args)
