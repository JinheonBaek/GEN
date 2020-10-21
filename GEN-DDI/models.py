import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from layers import GCNConv, GATConv, RGCNConv, GENConv

class InducGEN(nn.Module):

    def __init__(self, entity_embedding_dim, relation_embedding_dim, num_entities, num_relations, args, entity_embedding, relation_embedding):

        super(InducGEN, self).__init__()

        self.args = args

        self.entity_embedding_dim = entity_embedding_dim
        self.relation_embedding_dim = relation_embedding_dim
        self.num_entities = num_entities
        self.num_relations = num_relations

        self.entity_embedding = nn.Embedding(num_entities, entity_embedding_dim)

        if self.args.pre_train:
            self.entity_embedding.weight.data.copy_(entity_embedding.detach())

            if not self.args.fine_tune:
                self.entity_embedding.weight.requires_grad = False

        self.gnn = RGCNConv(self.entity_embedding_dim, self.entity_embedding_dim, self.num_relations * 2, num_bases = self.args.bases, root_weight = False, bias = False)

        self.dropout = nn.Dropout(args.dropout)
        
        self.decoder = nn.Sequential(
            nn.Linear(2*self.entity_embedding_dim, self.entity_embedding_dim),
            nn.ReLU(),
            nn.Linear(self.entity_embedding_dim, self.num_relations),
            nn.Sigmoid())

    def forward(self, unseen_entity, triplets, use_cuda):

        # Pre-process
        src, rel, dst = triplets.transpose()
        uniq_v, edges = np.unique((src, dst), return_inverse=True)
        src, dst = np.reshape(edges, (2, -1))

        unseen_index = np.where(uniq_v == unseen_entity)[0][0]

        src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
        rel = np.concatenate((rel, rel + self.num_relations))

        # Torch
        node_id = torch.LongTensor(uniq_v)
        edge_index = torch.stack((
            torch.LongTensor(src),
            torch.LongTensor(dst)
        ))
        edge_type = torch.LongTensor(rel)

        if use_cuda:
            node_id = node_id.cuda()
            edge_index = edge_index.cuda()
            edge_type = edge_type.cuda()

        x = self.entity_embedding(node_id)

        embeddings = self.gnn(x, edge_index, edge_type, edge_norm = None)
        unseen_entity_embedding = embeddings[unseen_index]
        unseen_entity_embedding = self.dropout(unseen_entity_embedding)

        return unseen_entity_embedding

    def predict(self, unseen_entity, unseen_entity_embedding, triplets, target, use_cuda):

        head_embeddings = self.entity_embedding(triplets[:, 0])
        tail_embeddings = self.entity_embedding(triplets[:, 2])

        head_embeddings[triplets[:, 0] == unseen_entity] = unseen_entity_embedding
        tail_embeddings[triplets[:, 2] == unseen_entity] = unseen_entity_embedding

        prob = self.decoder(torch.cat([head_embeddings, tail_embeddings], dim=1))
        y = triplets[:, 1].unsqueeze(1)
        y_onehot = torch.FloatTensor(triplets.size(0), self.num_relations).to(prob.device)
        y_onehot.zero_()
        y_onehot.scatter_(1, y, 1)

        return prob, y_onehot

    def score_loss(self, unseen_entity, unseen_entity_embedding, triplets, target, use_cuda):

        head_embeddings = self.entity_embedding(triplets[:, 0])
        tail_embeddings = self.entity_embedding(triplets[:, 2])

        head_embeddings[triplets[:, 0] == unseen_entity] = unseen_entity_embedding
        tail_embeddings[triplets[:, 2] == unseen_entity] = unseen_entity_embedding

        prob = self.decoder(torch.cat([head_embeddings, tail_embeddings], dim=1))
        y = triplets[:, 1].unsqueeze(1)
        y_onehot = torch.FloatTensor(triplets.size(0), self.num_relations).to(prob.device)
        y_onehot.zero_()
        y_onehot.scatter_(1, y, 1)

        loss = F.binary_cross_entropy(prob, y_onehot)

        return loss


class TransGEN(nn.Module):

    def __init__(self, entity_embedding_dim, relation_embedding_dim, num_entities, num_relations, args, entity_embedding, relation_embedding):

        super(TransGEN, self).__init__()

        self.args = args

        self.entity_embedding_dim = entity_embedding_dim
        self.relation_embedding_dim = relation_embedding_dim
        self.num_entities = num_entities
        self.num_relations = num_relations

        self.entity_embedding = nn.Embedding(num_entities, entity_embedding_dim)

        if self.args.pre_train:
            self.entity_embedding.weight.data.copy_(entity_embedding.detach())

            if not self.args.fine_tune:
                self.entity_embedding.weight.requires_grad = False

        self.relu = nn.ReLU()

        self.gnn_induc = RGCNConv(self.entity_embedding_dim, self.entity_embedding_dim, self.num_relations * 2, num_bases = self.args.bases, root_weight = False, bias = False)
        self.gnn_trans_mu = RGCNConv(self.entity_embedding_dim, self.entity_embedding_dim, self.num_relations * 2, num_bases = self.args.bases)
        self.gnn_trans_sigma = RGCNConv(self.entity_embedding_dim, self.entity_embedding_dim, self.num_relations * 2, num_bases = self.args.bases)

        self.dropout = nn.Dropout(args.dropout)

        self.decoder = nn.Sequential(
            nn.Linear(2*self.entity_embedding_dim, self.entity_embedding_dim),
            nn.ReLU(),
            nn.Linear(self.entity_embedding_dim, self.num_relations),
            nn.Sigmoid())

    def reparameterize(self, mu, logvar):

        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return mu + eps*std
        else:
            return mu

    def forward(self, unseen_entity, triplets, use_cuda, is_trans = True, total_unseen_entity_embedding = None):

        if is_trans == False:

            # Pre-process
            src, rel, dst = triplets.transpose()
            uniq_v, edges = np.unique((src, dst), return_inverse=True)
            src, dst = np.reshape(edges, (2, -1))

            unseen_index = np.where(uniq_v == unseen_entity)[0][0]

            src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
            rel = np.concatenate((rel, rel + self.num_relations))

            # Torch
            node_id = torch.LongTensor(uniq_v)
            edge_index = torch.stack((
                torch.LongTensor(src),
                torch.LongTensor(dst)
            ))
            edge_type = torch.LongTensor(rel)

            if use_cuda:
                node_id = node_id.cuda()
                edge_index = edge_index.cuda()
                edge_type = edge_type.cuda()

            x = self.entity_embedding(node_id)

            embeddings = self.gnn_induc(x, edge_index, edge_type, edge_norm = None)
            unseen_entity_embedding = embeddings[unseen_index]
            unseen_entity_embedding = self.relu(unseen_entity_embedding)
            unseen_entity_embedding = self.dropout(unseen_entity_embedding)

            return unseen_entity_embedding

        else:

            src, rel, dst = triplets.transpose()
            uniq_v, edges = np.unique((src, dst), return_inverse=True)
            src, dst = np.reshape(edges, (2, -1))

            unseen_index = []
            for entity in unseen_entity:
                unseen_index.append(np.where(uniq_v == entity)[0][0])

            src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
            rel = np.concatenate((rel, rel + self.num_relations))

            # Torch
            node_id = torch.LongTensor(uniq_v)
            edge_index = torch.stack((
                torch.LongTensor(src),
                torch.LongTensor(dst)
            ))
            edge_type = torch.LongTensor(rel)

            if use_cuda:
                node_id = node_id.cuda()
                edge_index = edge_index.cuda()
                edge_type = edge_type.cuda()

            x = self.entity_embedding(node_id)

            # Changed to Pre-calculated Embedding
            x[unseen_index] = total_unseen_entity_embedding

            mu = self.gnn_trans_mu(x, edge_index, edge_type, edge_norm = None)
            logvar = self.gnn_trans_sigma(x, edge_index, edge_type, edge_norm = None)
            embeddings = self.reparameterize(mu, logvar)
            
            unseen_entity_embedding = embeddings[unseen_index]
            unseen_entity_embedding = self.dropout(unseen_entity_embedding)

            return unseen_entity_embedding, mu, logvar

    def predict(self, unseen_entity, unseen_entity_embedding, triplets, target, use_cuda):

        head_embeddings = self.entity_embedding(triplets[:, 0])
        tail_embeddings = self.entity_embedding(triplets[:, 2])

        # Changed to Pre-calculated Embedding
        for index, entity in enumerate(unseen_entity):
            head_embeddings[triplets[:, 0] == entity] = unseen_entity_embedding[index]
            tail_embeddings[triplets[:, 2] == entity] = unseen_entity_embedding[index]

        prob = self.decoder(torch.cat([head_embeddings, tail_embeddings], dim=1))
        y = triplets[:, 1].unsqueeze(1)
        y_onehot = torch.FloatTensor(triplets.size(0), self.num_relations).to(prob.device)
        y_onehot.zero_()
        y_onehot.scatter_(1, y, 1)

        return prob, y_onehot

    def score_loss(self, unseen_entity, unseen_entity_embedding, triplets, target, use_cuda):

        head_embeddings = self.entity_embedding(triplets[:, 0])
        tail_embeddings = self.entity_embedding(triplets[:, 2])

        # Changed to Pre-calculated Embedding
        for index, entity in enumerate(unseen_entity):
            head_embeddings[triplets[:, 0] == entity] = unseen_entity_embedding[index]
            tail_embeddings[triplets[:, 2] == entity] = unseen_entity_embedding[index]

        prob = self.decoder(torch.cat([head_embeddings, tail_embeddings], dim=1))
        y = triplets[:, 1].unsqueeze(1)
        y_onehot = torch.FloatTensor(triplets.size(0), self.num_relations).to(prob.device)
        y_onehot.zero_()
        y_onehot.scatter_(1, y, 1)

        loss = F.binary_cross_entropy(prob, y_onehot)

        return loss
