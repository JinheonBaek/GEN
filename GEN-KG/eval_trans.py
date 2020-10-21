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

        self.entity2id, self.relation2id, self.train_triplets, self.valid_triplets, self.test_triplets = utils.load_data('./Dataset/raw_data/{}'.format(args.data))
        self.filtered_triplets, self.meta_train_task_triplets, self.meta_valid_task_triplets, self.meta_test_task_triplets, \
        self.meta_train_task_entity_to_triplets, self.meta_valid_task_entity_to_triplets, self.meta_test_task_entity_to_triplets \
            = utils.load_processed_data('./Dataset/processed_data/{}'.format(args.data))

        self.all_triplets = torch.LongTensor(np.concatenate((
            self.train_triplets, self.valid_triplets, self.test_triplets
        )))

        self.meta_task_entity = np.concatenate((list(self.meta_train_task_entity_to_triplets.keys()),
                                            list(self.meta_valid_task_entity_to_triplets.keys()),
                                            list(self.meta_test_task_entity_to_triplets.keys())))
        
        self.meta_task_triplets = torch.LongTensor(np.concatenate((self.meta_train_task_triplets, self.meta_valid_task_triplets, self.meta_test_task_triplets)))

        self.meta_task_test_entity = torch.LongTensor(np.array(list(self.meta_test_task_entity_to_triplets.keys())))

        self.load_pretrain_embedding(data = args.data, model = args.pre_train_model)
        self.load_model(model = args.model)

        if self.use_cuda:
            self.model.cuda()
            self.all_triplets = self.all_triplets.cuda()
            self.meta_task_triplets = self.meta_task_triplets.cuda()
            self.meta_task_test_entity = self.meta_task_test_entity.cuda()

        self.head_relation_triplets = self.all_triplets[:, :2]
        self.tail_relation_triplets = torch.stack((self.all_triplets[:, 2], self.all_triplets[:, 1])).transpose(0, 1)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.args.lr, weight_decay = self.args.weight_decay)

    def load_pretrain_embedding(self, data, model):

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

    def load_model(self, model):

        if self.args.model == 'TransGEN':

            self.model = TransGEN(self.embedding_size, self.embedding_size, len(self.entity2id), len(self.relation2id),
                            args = self.args, entity_embedding = self.pretrain_entity_embedding, relation_embedding = self.pretrain_relation_embedding)

        else:

            raise ValueError("Model Name <{}> is Wrong".format(self.args.model))

        meta_task_entity = torch.LongTensor(self.meta_task_entity)
        self.model.entity_embedding.weight.data[meta_task_entity] = torch.zeros(len(meta_task_entity), self.embedding_size)
        
    def train(self):
        
        checkpoint = torch.load('{}/best_mrr_model.pth'.format(self.exp_name), map_location='cuda:{}'.format(args.gpu))
        self.model.load_state_dict(checkpoint['state_dict'])

        print("Using best epoch: {}, {}".format(checkpoint['epoch'], self.exp_name))

        eval_types = {
            'normal': True,
            'mc_score_inference': True
        }

        results = {}
        mc_inference_results = {}

        if eval_types['normal']:

            tqdm.write("Results about Normal (Mean) Inference")

            with torch.no_grad():
                total_ranks, total_induc_ranks, total_trans_ranks = self.eval(eval_type='test')

            results['total_mrrs'] = torch.mean(1.0 / total_ranks.float()).item()

            for hit in [1, 3, 10]:
                avg_count = torch.mean((total_ranks <= hit).float())
                results['total_hits@{}s'.format(hit)] = avg_count.item()

            results['total_induc_mrrs'] = torch.mean(1.0 / total_induc_ranks.float()).item()

            for hit in [1, 3, 10]:
                avg_count = torch.mean((total_induc_ranks <= hit).float())
                results['total_induc_hits@{}s'.format(hit)] = avg_count.item()

            results['total_trans_mrrs'] = torch.mean(1.0 / total_trans_ranks.float()).item()

            for hit in [1, 3, 10]:
                avg_count = torch.mean((total_trans_ranks <= hit).float())
                results['total_trans_hits@{}s'.format(hit)] = avg_count.item()

            tqdm.write("Total MRR (filtered): {:.6f}".format(results['total_mrrs']))
            tqdm.write("Total Hits (filtered) @ {}: {:.6f}".format(1, results['total_hits@1s']))
            tqdm.write("Total Hits (filtered) @ {}: {:.6f}".format(3, results['total_hits@3s']))
            tqdm.write("Total Hits (filtered) @ {}: {:.6f}".format(10, results['total_hits@10s']))

            tqdm.write("Total Induc MRR (filtered): {:.6f}".format(results['total_induc_mrrs']))
            tqdm.write("Total Induc Hits (filtered) @ {}: {:.6f}".format(1, results['total_induc_hits@1s']))
            tqdm.write("Total Induc Hits (filtered) @ {}: {:.6f}".format(3, results['total_induc_hits@3s']))
            tqdm.write("Total Induc Hits (filtered) @ {}: {:.6f}".format(10, results['total_induc_hits@10s']))

            tqdm.write("Total Trans MRR (filtered): {:.6f}".format(results['total_trans_mrrs']))
            tqdm.write("Total Trans Hits (filtered) @ {}: {:.6f}".format(1, results['total_trans_hits@1s']))
            tqdm.write("Total Trans Hits (filtered) @ {}: {:.6f}".format(3, results['total_trans_hits@3s']))
            tqdm.write("Total Trans Hits (filtered) @ {}: {:.6f}".format(10, results['total_trans_hits@10s']))


        if eval_types['mc_score_inference']:

            tqdm.write("Results about MC score inference")

            with torch.no_grad():
                total_ranks, total_induc_ranks, total_trans_ranks = self.mc_score_inference(eval_type='test', prob=False)

            mc_inference_results['total_mrrs'] = torch.mean(1.0 / total_ranks.float()).item()

            for hit in [1, 3, 10]:
                avg_count = torch.mean((total_ranks <= hit).float())
                mc_inference_results['total_hits@{}s'.format(hit)] = avg_count.item()

            mc_inference_results['total_induc_mrrs'] = torch.mean(1.0 / total_induc_ranks.float()).item()

            for hit in [1, 3, 10]:
                avg_count = torch.mean((total_induc_ranks <= hit).float())
                mc_inference_results['total_induc_hits@{}s'.format(hit)] = avg_count.item()

            mc_inference_results['total_trans_mrrs'] = torch.mean(1.0 / total_trans_ranks.float()).item()

            for hit in [1, 3, 10]:
                avg_count = torch.mean((total_trans_ranks <= hit).float())
                mc_inference_results['total_trans_hits@{}s'.format(hit)] = avg_count.item()

            tqdm.write("Total MRR (filtered): {:.6f}".format(mc_inference_results['total_mrrs']))
            tqdm.write("Total Hits (filtered) @ {}: {:.6f}".format(1, mc_inference_results['total_hits@1s']))
            tqdm.write("Total Hits (filtered) @ {}: {:.6f}".format(3, mc_inference_results['total_hits@3s']))
            tqdm.write("Total Hits (filtered) @ {}: {:.6f}".format(10, mc_inference_results['total_hits@10s']))

            tqdm.write("Total Induc MRR (filtered): {:.6f}".format(mc_inference_results['total_induc_mrrs']))
            tqdm.write("Total Induc Hits (filtered) @ {}: {:.6f}".format(1, mc_inference_results['total_induc_hits@1s']))
            tqdm.write("Total Induc Hits (filtered) @ {}: {:.6f}".format(3, mc_inference_results['total_induc_hits@3s']))
            tqdm.write("Total Induc Hits (filtered) @ {}: {:.6f}".format(10, mc_inference_results['total_induc_hits@10s']))

            tqdm.write("Total Trans MRR (filtered): {:.6f}".format(mc_inference_results['total_trans_mrrs']))
            tqdm.write("Total Trans Hits (filtered) @ {}: {:.6f}".format(1, mc_inference_results['total_trans_hits@1s']))
            tqdm.write("Total Trans Hits (filtered) @ {}: {:.6f}".format(3, mc_inference_results['total_trans_hits@3s']))
            tqdm.write("Total Trans Hits (filtered) @ {}: {:.6f}".format(10, mc_inference_results['total_trans_hits@10s']))


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
        subject_ranks = []
        object_ranks = []
        total_induc_ranks = []
        subject_induc_ranks = []
        object_induc_ranks = []
        total_trans_ranks = []
        subject_trans_ranks = []
        object_trans_ranks = []

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

        all_entity_embeddings = copy.deepcopy(self.model.entity_embedding.weight).detach()
        all_relation_embeddings = copy.deepcopy(self.model.relation_embedding).detach()

        all_entity_embeddings[total_task_entity] = task_entity_embeddings

        for task_entity, test_triplets in total_test_task_triplets_dict.items():

            if self.use_cuda:
                device = torch.device('cuda')
                test_triplets = test_triplets.cuda()

            for test_triplet in test_triplets:

                rank, (is_trans, is_subject, is_object) = self.calc_rank(task_entity, total_task_entity, all_entity_embeddings[task_entity], all_entity_embeddings, all_relation_embeddings, test_triplet, self.all_triplets, use_cuda = self.use_cuda, score_function = self.args.score_function)

                rank += 1
                rank = rank.cpu()

                total_ranks.append(rank)

                if is_subject:
                
                    subject_ranks.append(rank)
                
                elif is_object:
                    
                    object_ranks.append(rank)

                if is_trans:

                    total_trans_ranks.append(rank)

                    if is_subject:
                        subject_trans_ranks.append(rank)
                    elif is_object:
                        object_trans_ranks.append(rank)

                else:

                    total_induc_ranks.append(rank)

                    if is_subject:
                        subject_induc_ranks.append(rank)
                    elif is_object:
                        object_induc_ranks.append(rank)

        total_ranks = torch.cat(total_ranks)
        total_induc_ranks = torch.cat(total_induc_ranks)
        total_trans_ranks = torch.cat(total_trans_ranks)

        return total_ranks, total_induc_ranks, total_trans_ranks


    def mc_score_inference(self, eval_type='test', prob=False):

        self.model.eval()

        if eval_type == 'valid':
            test_task_dict = self.meta_valid_task_entity_to_triplets
            test_task_pool = list(self.meta_valid_task_entity_to_triplets.keys())

        elif eval_type == 'test':
            test_task_dict = self.meta_test_task_entity_to_triplets
            test_task_pool = list(self.meta_test_task_entity_to_triplets.keys())

        else:
            raise ValueError("Eval Type <{}> is Wrong".format(eval_type))

        
        total_scores = []
        total_induc_scores = []
        total_trans_scores = []

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

        self.model.train()

        # Train (Transductive)
        total_task_entity = np.array(total_task_entity)
        total_task_entity_embeddings = torch.cat(total_task_entity_embeddings).view(-1, self.args.mc_times, self.embedding_size)
        total_train_task_triplets = np.array(total_train_task_triplets)
        total_test_task_triplets = torch.LongTensor(total_test_task_triplets)        

        task_entity_embeddings = torch.cat([
            self.model(
                total_task_entity, total_train_task_triplets, use_cuda = self.use_cuda, is_trans = True, total_unseen_entity_embedding = total_task_entity_embeddings[:, i]
            )[0] for i in range(self.args.mc_times)
        ]).view(self.args.mc_times, -1, self.embedding_size)

        # Test
        total_task_entity = torch.from_numpy(total_task_entity)

        if self.use_cuda:
            total_task_entity = total_task_entity.cuda()

        for mc_index in trange(self.args.mc_times):

            scores = []
            induc_scores = []
            trans_scores = []

            all_entity_embeddings = copy.deepcopy(self.model.entity_embedding.weight).detach()
            all_relation_embeddings = copy.deepcopy(self.model.relation_embedding).detach()

            all_entity_embeddings[total_task_entity] = task_entity_embeddings[mc_index, :]
        
            for task_entity, test_triplets in total_test_task_triplets_dict.items():

                if self.use_cuda:
                    device = torch.device('cuda')
                    test_triplets = test_triplets.cuda()

                for test_triplet in test_triplets:

                    score, (is_trans, is_subject, is_object) = self.calc_score(task_entity, total_task_entity, all_entity_embeddings[task_entity], all_entity_embeddings, all_relation_embeddings, test_triplet, self.all_triplets, use_cuda = self.use_cuda, score_function = self.args.score_function)

                    if prob == True:
                        score = F.softmax(score, dim=1)

                    score = score.to('cpu')
                    scores.append(score)

                    if is_trans:
                        trans_scores.append(score)
                    else:
                        induc_scores.append(score)

            total_scores.append(scores)
            total_induc_scores.append(induc_scores)
            total_trans_scores.append(trans_scores)

        total_scores = list(map(list, zip(*total_scores)))
        total_induc_scores = list(map(list, zip(*total_induc_scores)))
        total_trans_scores = list(map(list, zip(*total_trans_scores)))

        total_ranks = []
        total_induc_ranks = []
        total_trans_ranks = []

        target = torch.tensor(0)

        for i in range(len(total_scores)):

            scores = total_scores[i]
            scores = torch.cat(scores)

            score = torch.mean(scores, dim=0, keepdim=True)

            rank = utils.sort_and_rank(score, target)            
            rank += 1

            total_ranks.append(rank)

        for i in range(len(total_induc_scores)):

            scores = total_induc_scores[i]
            scores = torch.cat(scores)

            score = torch.mean(scores, dim=0, keepdim=True)

            rank = utils.sort_and_rank(score, target)            
            rank += 1

            total_induc_ranks.append(rank)

        for i in range(len(total_trans_scores)):

            scores = total_trans_scores[i]
            scores = torch.cat(scores)

            score = torch.mean(scores, dim=0, keepdim=True)

            rank = utils.sort_and_rank(score, target)            
            rank += 1

            total_trans_ranks.append(rank)

        total_ranks = torch.cat(total_ranks)
        total_induc_ranks = torch.cat(total_induc_ranks)
        total_trans_ranks = torch.cat(total_trans_ranks)

        return total_ranks, total_induc_ranks, total_trans_ranks


    def calc_rank(self, task_entity, total_task_entity, task_entity_embedding, entity_embeddings, relation_embeddings, test_triplet, all_triplets, use_cuda, score_function):

        num_entity = len(entity_embeddings)

        is_trans = False
        is_subject = False
        is_object = False

        if (test_triplet[0] in total_task_entity) and (test_triplet[2] in total_task_entity):

            is_trans = True            

        if (test_triplet[0] == task_entity):

            is_subject = True 

            subject = test_triplet[0]
            relation = test_triplet[1]
            object_ = test_triplet[2]

            subject_relation = torch.LongTensor([subject, relation])
            if use_cuda:
                subject_relation = subject_relation.cuda()

            delete_index = torch.sum(self.head_relation_triplets == subject_relation, dim = 1)
            delete_index = torch.nonzero(delete_index == 2).squeeze()

            if use_cuda:
                device = torch.device('cuda')
                delete_entity_index = all_triplets[delete_index, 2].view(-1).cpu().numpy()
                perturb_entity_index = np.array(list(set(np.arange(num_entity)) - set(delete_entity_index)))
                perturb_entity_index = torch.from_numpy(perturb_entity_index).to(device)
                perturb_entity_index = torch.cat((object_.view(-1), perturb_entity_index))
            else:
                delete_entity_index = all_triplets[delete_index, 2].view(-1).numpy()
                perturb_entity_index = np.array(list(set(np.arange(num_entity)) - set(delete_entity_index)))
                perturb_entity_index = torch.from_numpy(perturb_entity_index)
                perturb_entity_index = torch.cat((object_.view(-1), perturb_entity_index))

            # Score
            if score_function == 'DistMult':

                emb_ar = task_entity_embedding * relation_embeddings[relation]
                emb_ar = emb_ar.view(-1, 1, 1)

                emb_c = entity_embeddings[perturb_entity_index]
                emb_c = emb_c.transpose(0, 1).unsqueeze(1)

                out_prod = torch.bmm(emb_ar, emb_c)
                score = torch.sum(out_prod, dim = 0)
                score = F.softmax(score, dim=1)
                
            elif score_function == 'TransE':

                head_embedding = task_entity_embedding
                relation_embedding = relation_embeddings[relation]
                tail_embeddings = entity_embeddings[perturb_entity_index]

                score = - torch.norm((head_embedding + relation_embedding - tail_embeddings), p = 2, dim = 1)
                score = score.view(1, -1)
                score = F.softmax(score, dim=1)
                
            else:

                raise TypeError

        elif (test_triplet[2] == task_entity):

            is_object = True

            subject = test_triplet[0]
            relation = test_triplet[1]
            object_ = test_triplet[2]

            object_relation = torch.LongTensor([object_, relation])
            if use_cuda:
                object_relation = object_relation.cuda()

            delete_index = torch.sum(self.tail_relation_triplets == object_relation, dim = 1)
            delete_index = torch.nonzero(delete_index == 2).squeeze()

            if use_cuda:
                device = torch.device('cuda')
                delete_entity_index = all_triplets[delete_index, 0].view(-1).cpu().numpy()
                perturb_entity_index = np.array(list(set(np.arange(num_entity)) - set(delete_entity_index)))
                perturb_entity_index = torch.from_numpy(perturb_entity_index).to(device)
                perturb_entity_index = torch.cat((subject.view(-1), perturb_entity_index))
            else:
                delete_entity_index = all_triplets[delete_index, 0].view(-1).numpy()
                perturb_entity_index = np.array(list(set(np.arange(num_entity)) - set(delete_entity_index)))
                perturb_entity_index = torch.from_numpy(perturb_entity_index)
                perturb_entity_index = torch.cat((subject.view(-1), perturb_entity_index))

            # Score
            if score_function == 'DistMult':

                emb_ar = task_entity_embedding * relation_embeddings[relation]
                emb_ar = emb_ar.view(-1, 1, 1)

                emb_c = entity_embeddings[perturb_entity_index]
                emb_c = emb_c.transpose(0, 1).unsqueeze(1)

                out_prod = torch.bmm(emb_ar, emb_c)
                score = torch.sum(out_prod, dim = 0)
                score = F.softmax(score, dim=1)
                
            elif score_function == 'TransE':

                head_embeddings = entity_embeddings[perturb_entity_index]
                relation_embedding = relation_embeddings[relation]
                tail_embedding = task_entity_embedding

                score = head_embeddings + relation_embedding - tail_embedding
                score = - torch.norm(score, p = 2, dim = 1)
                score = score.view(1, -1)
                score = F.softmax(score, dim=1)
                
            else:

                raise TypeError

        if use_cuda:
            target = torch.tensor(0).to(device)
            rank = utils.sort_and_rank(score, target)
        
        else:
            target = torch.tensor(0)
            rank = utils.sort_and_rank(score, target)
            
        return rank, (is_trans, is_subject, is_object)


    def calc_score(self, task_entity, total_task_entity, task_entity_embedding, entity_embeddings, relation_embeddings, test_triplet, all_triplets, use_cuda, score_function):

        num_entity = len(entity_embeddings)

        is_trans = False
        is_subject = False
        is_object = False

        if (test_triplet[0] in total_task_entity) and (test_triplet[2] in total_task_entity):

            is_trans = True
            
        if (test_triplet[0] == task_entity):

            is_subject = True 

            subject = test_triplet[0]
            relation = test_triplet[1]
            object_ = test_triplet[2]

            subject_relation = torch.LongTensor([subject, relation])
            if use_cuda:
                subject_relation = subject_relation.cuda()

            delete_index = torch.sum(self.head_relation_triplets == subject_relation, dim = 1)
            delete_index = torch.nonzero(delete_index == 2).squeeze()

            if use_cuda:
                device = torch.device('cuda')
                delete_entity_index = all_triplets[delete_index, 2].view(-1).cpu().numpy()
                perturb_entity_index = np.array(list(set(np.arange(num_entity)) - set(delete_entity_index)))
                perturb_entity_index = torch.from_numpy(perturb_entity_index).to(device)
                perturb_entity_index = torch.cat((object_.view(-1), perturb_entity_index))
            else:
                delete_entity_index = all_triplets[delete_index, 2].view(-1).numpy()
                perturb_entity_index = np.array(list(set(np.arange(num_entity)) - set(delete_entity_index)))
                perturb_entity_index = torch.from_numpy(perturb_entity_index)
                perturb_entity_index = torch.cat((object_.view(-1), perturb_entity_index))

            # Score
            if score_function == 'DistMult':

                emb_ar = task_entity_embedding * relation_embeddings[relation]
                emb_ar = emb_ar.view(-1, 1, 1)

                emb_c = entity_embeddings[perturb_entity_index]
                emb_c = emb_c.transpose(0, 1).unsqueeze(1)

                out_prod = torch.bmm(emb_ar, emb_c)
                score = torch.sum(out_prod, dim = 0)
                
            elif score_function == 'TransE':

                head_embedding = task_entity_embedding
                relation_embedding = relation_embeddings[relation]
                tail_embeddings = entity_embeddings[perturb_entity_index]

                score = - torch.norm((head_embedding + relation_embedding - tail_embeddings), p = 2, dim = 1)
                score = score.view(1, -1)
                
            else:

                raise TypeError

        elif (test_triplet[2] == task_entity):

            is_object = True

            subject = test_triplet[0]
            relation = test_triplet[1]
            object_ = test_triplet[2]

            object_relation = torch.LongTensor([object_, relation])
            if use_cuda:
                object_relation = object_relation.cuda()

            delete_index = torch.sum(self.tail_relation_triplets == object_relation, dim = 1)
            delete_index = torch.nonzero(delete_index == 2).squeeze()

            if use_cuda:
                device = torch.device('cuda')
                delete_entity_index = all_triplets[delete_index, 0].view(-1).cpu().numpy()
                perturb_entity_index = np.array(list(set(np.arange(num_entity)) - set(delete_entity_index)))
                perturb_entity_index = torch.from_numpy(perturb_entity_index).to(device)
                perturb_entity_index = torch.cat((subject.view(-1), perturb_entity_index))
            else:
                delete_entity_index = all_triplets[delete_index, 0].view(-1).numpy()
                perturb_entity_index = np.array(list(set(np.arange(num_entity)) - set(delete_entity_index)))
                perturb_entity_index = torch.from_numpy(perturb_entity_index)
                perturb_entity_index = torch.cat((subject.view(-1), perturb_entity_index))

            # Score
            if score_function == 'DistMult':

                emb_ar = task_entity_embedding * relation_embeddings[relation]
                emb_ar = emb_ar.view(-1, 1, 1)

                emb_c = entity_embeddings[perturb_entity_index]
                emb_c = emb_c.transpose(0, 1).unsqueeze(1)

                out_prod = torch.bmm(emb_ar, emb_c)
                score = torch.sum(out_prod, dim = 0)
                
            elif score_function == 'TransE':

                head_embeddings = entity_embeddings[perturb_entity_index]
                relation_embedding = relation_embeddings[relation]
                tail_embedding = task_entity_embedding

                score = head_embeddings + relation_embedding - tail_embedding
                score = - torch.norm(score, p = 2, dim = 1)
                score = score.view(1, -1)
                
            else:

                raise TypeError
            
        return score, (is_trans, is_subject, is_object)


    def experiment_name(self, args):

        exp_name = os.path.join('./checkpoints', self.args.exp_name)

        return exp_name


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Meta-KGNN')
    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action='store_true')

    parser.add_argument("--exp-name", type=str, default='FB15k-237_Trans')

    parser.add_argument("--data", type=str, default='FB15k-237')
    parser.add_argument("--negative-sample", type=int, default=1)

    parser.add_argument("--few", type=int, default=3)
    
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
    parser.add_argument("--model", type=str)
    parser.add_argument("--score-function", type=str, default='DistMult')

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