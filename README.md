# Learning to Extrapolate Knowledge:</br> Transductive Few-shot Out-of-Graph Link Prediction

Official Code Repository for the paper "Learning to Extrapolate Knowledge: Transductive Few-shot Out-of-Graph Link Prediction" (NeurIPS 2020)

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

You can see the pre-trained models in a *Pretraining* folder on each dataset.

## Results

We demonstrate our *Graph Extrapolation Networks* on two types of link prediction task: entity prediction for knowledge graph completion and relation prediction for drug-drug interaction prediction.

Our model achieves the following performance for 3-shot entity prediction on FB15k-237 dataset.
| Model     | MRR   | Hits@1    | Hits@3    | Hits@10   |
| ------    | ----  | ----      | ----      | ---       |
| I-GEN     | .367  | .281      | .407      | .537      |
| T-GEN     | .382  | .289      | .430      | .565      |

Our model achieves the following performance for 3-shot relation prediction on DeepDDI dataset.
| Model     | ROC   | PR    | Acc   |
| ------    | ----  | ----  | ----  |
| I-GEN     | .946  | .681  | .807  |
| T-GEN     | .954  | .708  | .815  |
