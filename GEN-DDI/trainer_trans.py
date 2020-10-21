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

        self.best_roc = 0

        self.use_cuda = args.gpu >= 0 and torch.cuda.is_available()
        if self.use_cuda:
            torch.cuda.set_device(args.gpu)

        self.filtered_triplets, self.meta_train_task_triplets, self.meta_valid_task_triplets, self.meta_test_task_triplets, \
        self.meta_train_task_entity_to_triplets, self.meta_valid_task_entity_to_triplets, self.meta_test_task_entity_to_triplets \
            = utils.load_processed_data('./Dataset/processed_data/{}'.format(args.data))

        self.meta_task_entity = np.concatenate((list(self.meta_train_task_entity_to_triplets.keys()),
                                            list(self.meta_valid_task_entity_to_triplets.keys()),
                                            list(self.meta_test_task_entity_to_triplets.keys())))

        self.load_pretrain_embedding()
        self.load_model()

        print(self.model)

        if self.use_cuda:
            self.model.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.args.lr, weight_decay = self.args.weight_decay)

    def load_pretrain_embedding(self):

        self.embedding_size = int(self.args.pre_train_emb_size)

        if self.args.pre_train:
            entity_file_name = './Pretraining/{}/{}_entity.npy'.format(self.args.data, self.args.pre_train_model)
            
            self.pretrain_entity_embedding = torch.Tensor(np.load(entity_file_name))
            self.pretrain_relation_embedding = None

        else:
            self.pretrain_entity_embedding = None
            self.pretrain_relation_embedding = None

    def load_model(self):

        num_entities = None
        num_relations = None

        if self.args.data == 'BIOSNAP-sub':
            num_entities = 637
            num_relations = 200

        elif self.args.data == 'DeepDDI':
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

        for epoch in trange(0, (self.args.n_epochs + 1), desc = 'Train Epochs', position = 0):

            # Meta-Train
            self.model.train()

            train_task_pool = list(self.meta_train_task_entity_to_triplets.keys())
            random.shuffle(train_task_pool)

            total_unseen_entity = []
            total_unseen_entity_embedding = []
            total_train_triplets = []
            total_samples = []

            for unseen_entity in train_task_pool[:self.args.num_train_entity]:

                triplets = self.meta_train_task_entity_to_triplets[unseen_entity]
                random.shuffle(triplets)
                triplets = np.array(triplets)
                heads, relations, tails = triplets.transpose()

                train_triplets = triplets[:self.args.few]
                test_triplets = triplets[self.args.few:]

                if (len(triplets)) - self.args.few < 1:
                    continue

                samples = test_triplets
                samples = torch.LongTensor(samples)

                if self.use_cuda:
                    samples = samples.cuda()

                # Train (Inductive)
                unseen_entity_embedding = self.model(unseen_entity, train_triplets, self.use_cuda, is_trans = False)
                total_unseen_entity.append(unseen_entity)
                total_unseen_entity_embedding.append(unseen_entity_embedding)
                total_train_triplets.extend(train_triplets)
                total_samples.append(samples)

            total_unseen_entity = np.array(total_unseen_entity)
            total_unseen_entity_embedding = torch.cat(total_unseen_entity_embedding).view(-1, self.embedding_size)
            total_train_triplets = np.array(total_train_triplets)

            total_samples = torch.cat(total_samples, dim=0)

            # Train (Transductive)
            unseen_entity_embeddings, _, _ = self.model(total_unseen_entity, total_train_triplets, self.use_cuda, is_trans = True, total_unseen_entity_embedding = total_unseen_entity_embedding)

            loss = self.model.score_loss(total_unseen_entity, unseen_entity_embeddings, total_samples, target=None, use_cuda=self.use_cuda)

            # Test
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_norm)
            self.optimizer.step()

            # Meta-Valid
            if epoch % self.args.evaluate_every == 0:

                tqdm.write("Epochs-{}, Loss-{}".format(epoch, loss))

                with torch.no_grad():

                    results = self.eval(eval_type='valid')

                    roc = results['roc']

                    tqdm.write("Total PR (filtered): {:.6f}".format(results['pr']))
                    tqdm.write("Total ROC (filtered): {:.6f}".format(results['roc']))
                    tqdm.write("Total ACC (filtered): {:.6f}".format(results['acc']))

                if roc > self.best_roc:
                    self.best_roc = roc
                    torch.save({'state_dict': self.model.state_dict(), 'epoch': epoch},
                            './checkpoints/{}/best_mrr_model.pth'.format(self.exp_name))

        checkpoint = torch.load('./checkpoints/{}/best_mrr_model.pth'.format(self.exp_name))
        self.model.load_state_dict(checkpoint['state_dict'])

        print("Using best epoch: {}, {}".format(checkpoint['epoch'], self.exp_name))

        # Meta-Test
        results = self.eval(eval_type='test')

        tqdm.write("Total PR (filtered): {:.6f}".format(results['pr']))
        tqdm.write("Total ROC (filtered): {:.6f}".format(results['roc']))
        tqdm.write("Total ACC (filtered): {:.6f}".format(results['acc']))


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

        total_unseen_entity = []
        total_unseen_entity_embedding = []
        total_train_triplets = []
        total_test_triplets = []
        total_test_triplets_dict = dict()

        for unseen_entity in test_task_pool:

            triplets = test_task_dict[unseen_entity]
            triplets = np.array(triplets)
            heads, relations, tails = triplets.transpose()

            train_triplets = triplets[:self.args.few]
            test_triplets = triplets[self.args.few:]

            if (len(triplets)) - self.args.few < 1:
                continue

            # Train (Inductive)
            unseen_entity_embedding = self.model(unseen_entity, train_triplets, use_cuda = self.use_cuda, is_trans = False)
            total_unseen_entity.append(unseen_entity)
            total_unseen_entity_embedding.append(unseen_entity_embedding)
            total_train_triplets.extend(train_triplets)
            total_test_triplets.extend(test_triplets)
            total_test_triplets_dict[unseen_entity] = torch.LongTensor(test_triplets)

        # Train (Transductive)
        total_unseen_entity = np.array(total_unseen_entity)
        total_unseen_entity_embedding = torch.cat(total_unseen_entity_embedding).view(-1, self.embedding_size)
        total_train_triplets = np.array(total_train_triplets)

        samples = total_test_triplets
        samples = torch.LongTensor(samples)

        if self.use_cuda:
            samples = samples.cuda()

        unseen_entity_embeddings, _, _ = self.model(total_unseen_entity, total_train_triplets, use_cuda = self.use_cuda, is_trans = True, total_unseen_entity_embedding = total_unseen_entity_embedding)
        y_prob, y = self.model.predict(total_unseen_entity, unseen_entity_embeddings, samples, target=None, use_cuda=self.use_cuda)

        y_prob = y_prob.detach().cpu().numpy()
        y = y.detach().cpu().numpy()

        results = utils.metric_report(y, y_prob)

        return results


    def experiment_name(self, args):

        ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())

        exp_name = str()
        exp_name += "Data={}_".format(args.data)
        exp_name += "Model={}_".format(args.model)
        exp_name += "Bases={}_".format(args.bases)
        exp_name += "DO={}_".format(args.dropout)
        exp_name += "Few={}_".format(args.few)
        exp_name += "LR={}_".format(args.lr)
        exp_name += "WD={}_".format(args.weight_decay)
        exp_name += "GN={}_".format(args.grad_norm)
        exp_name += "PT={}_".format(args.pre_train)
        exp_name += "PTM={}_".format(args.pre_train_model)
        exp_name += "PTES={}_".format(args.pre_train_emb_size)
        exp_name += "FT={}_".format(args.fine_tune)
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

    parser.add_argument("--data", type=str, default='DeepDDI')

    parser.add_argument("--few", type=int, default=3)
    parser.add_argument("--n-epochs", type=int, default=6000)
    parser.add_argument("--evaluate-every", type=int, default=100)

    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", default=0.0, type=float)

    parser.add_argument("--grad-norm", type=float, default=1.0)
    parser.add_argument("--bases", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--pre-train", action='store_true')
    parser.add_argument("--fine-tune", action='store_true')

    parser.add_argument("--pre-train-model", type=str, default='MPNN')
    parser.add_argument("--pre-train-emb-size", type=str, default='100')
    parser.add_argument("--num-train-entity", type=int, default=80)
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
