# Learning to Extrapolate Knowledge:</br> Transductive Few-shot Out-of-Graph Link Prediction

Official Code Repository for the paper "Learning to Extrapolate Knowledge: Transductive Few-shot Out-of-Graph Link Prediction" (NeurIPS 2020) : https://arxiv.org/abs/2006.06648.

## Abstract

<p align="center">
<img width="600" src="https://user-images.githubusercontent.com/26034065/96733515-8046be80-13f4-11eb-8175-fe3e4af30458.png">
</p>

Many practical graph problems, such as knowledge graph construction and drug-drug interaction prediction, require to handle multi-relational graphs. However, handling real-world multi-relational graphs with Graph Neural Networks (GNNs) is often challenging due to their evolving nature, where new entities (nodes) can emerge over time. Moreover, newly emerged entities often have few links, which makes the learning even more difficult. Motivated by this challenge, we introduce a realistic problem of *few-shot out-of-graph link prediction*, where we not only predict the links between the seen and unseen nodes as in a conventional out-of-knowledge link prediction but also between the unseen nodes, with only few edges per node. We tackle this problem with a novel transductive meta-learning framework which we refer to as *Graph Extrapolation Networks (GEN)*. GEN meta-learns both the node embedding network for inductive inference (seen-to-unseen) and the link prediction network for transductive inference (unseen-to-unseen). For transductive link prediction, we further propose a stochastic embedding layer to model uncertainty in the link prediction between unseen entities. We validate our model on multiple benchmark datasets for knowledge graph completion and drug-drug interaction prediction. The results show that our model significantly outperforms relevant baselines for out-of-graph link prediction tasks.

### Contribution of this work

* We tackle a realistic problem setting of **few-shot out-of-graph link prediction**, aiming to perform link prediction not only between seen and unseen entities but also among unseen entities for multi-relational graphs that exhibit long-tail distributions, where each entity has only few triplets.
* To tackle this problem, we propose a **novel meta-learning framework**, Graph Extrapolation Network (GEN), which meta-learns the node embeddings for unseen entities, to obtain low error on link prediction for both seen-to-unseen (inductive) and unseen-to-unseen (transductive) cases.
* We validate GEN for few-shot out-of-graph link prediction tasks on **five benchmark datasets** for **knowledge graph completion** and **drug-drug interaction prediction**, on which it significantly outperforms relevant baselines, even when baseline models are re-trained with the unseen entities.

## Dependencies

* Python 3.7
* PyTorch 1.4
* PyTorch Geometric 1.4.3

## Training

To train models in the paper, run following commands:

* (GEN-KG) FB15k-237 - Inductive (I-GEN)

```python
python trainer_induc.py --data FB15k-237 --gpu -1 --few 3 --pre-train --fine-tune --model InducGEN --pre-train-model DistMult --score-function DistMult --margin 1 --seed 42 --evaluate-every 500 --pre-train-emb-size 100 --negative-sample 32 --model-tail log --max-few 10
```

* (GEN-KG) FB15k-237 - Transductive (T-GEN)

```python
python trainer_trans.py --data FB15k-237 --gpu -1 --few 3 --pre-train --fine-tune --model TransGEN --pre-train-model DistMult --score-function DistMult --margin 1 --seed 42 --evaluate-every 500 --pre-train-emb-size 100 --negative-sample 32 --model-tail log --max-few 10
```

* (GEN-DDI) DeepDDI - Inductive (I-GEN)

```python
python trainer_induc.py --data DeepDDI --pre-train --fine-tune --pre-train-model MPNN --n-epochs 5000 --evaluate-every 100 --model InducGEN --seed 42 --gpu -1 --few 3 --bases 200
```

* (GEN-DDI) DeepDDI - Transductive (T-GEN)

```python
python trainer_trans.py --data DeepDDI --pre-train --fine-tune --pre-train-model MPNN --n-epochs 5000 --evaluate-every 100 --model TransGEN --seed 42 --gpu -1 --few 3 --bases 200
```

## Evaluation

To evaluate models, run following commands:

* (GEN-KG) FB15k-237 - Inductive (I-GEN)

```python
python eval_induc.py --data FB15k-237 --gpu -1 --few 3 --pre-train --fine-tune --model InducGEN --pre-train-model DistMult --score-function DistMult --margin 1 --pre-train-emb-size 100 --negative-sample 32 --exp-name FB15k-237_Induc
```

* (GEN-KG) FB15k-237 - Transductive (T-GEN)

```python
python eval_trans.py --data FB15k-237 --gpu -1 --few 3 --pre-train --fine-tune --model TransGEN --pre-train-model DistMult --score-function DistMult --margin 1 --pre-train-emb-size 100 --negative-sample 32 --exp-name FB15k-237_Trans --mc-times 10
```

* (GEN-DDI) DeepDDI - Inductive (I-GEN)

```python
python eval_induc.py --data DeepDDI --gpu -1 --few 3 --pre-train --fine-tune --model InducGEN --pre-train-model MPNN --exp-name Deep-DDI_Induc --bases 200
```

* (GEN-DDI) DeepDDI - Transductive (T-GEN)

```python
python eval_trans.py --data DeepDDI --gpu -1 --few 3 --pre-train --fine-tune --model TransGEN --pre-train-model MPNN --exp-name Deep-DDI_Trans --bases 200 --mc-times 10
```

## Pre-trained Models

You can see the pre-trained models in a *Pretraining* folder for each task (GEN-KG or GEN-DDI).

## Results

We demonstrate our *Graph Extrapolation Networks* on two types of link prediction task: entity prediction for knowledge graph completion and relation prediction for drug-drug interaction prediction.

Our model achieves the following performances for 3-shot entity prediction on FB15k-237 dataset.
| Model     | MRR   | Hits@1    | Hits@3    | Hits@10   |
| ------    | ----  | ----      | ----      | ---       |
| I-GEN     | .367  | .281      | .407      | .537      |
| T-GEN     | .382  | .289      | .430      | .565      |

Our model achieves the following performances for 3-shot relation prediction on DeepDDI dataset.
| Model     | ROC   | PR    | Acc   |
| ------    | ----  | ----  | ----  |
| I-GEN     | .946  | .681  | .807  |
| T-GEN     | .954  | .708  | .815  |
