import copy
import math
from collections import defaultdict
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_add

from torchdrug import core, tasks, layers, models, data, metrics, utils
from torchdrug.layers import functional
from torchdrug.core import Registry as R
import numpy as np


@R.register("tasks.PhosphorylationPrediction")
class PhosphorylationPrediction(tasks.InteractionPrediction):
    def __init__(self, model, task, num_mlp_layer=3, graph_construction_model=None, verbose=0, mlp_dropout=0, mlp_batch_norm=False):
        super(PhosphorylationPrediction, self).__init__(model, model2=model, task=task, criterion="bce",
            metric=("acc", "auroc", "auprc", "fpr"), num_mlp_layer=num_mlp_layer, normalization=False,
            num_class=1, graph_construction_model=graph_construction_model, verbose=verbose, 
            mlp_dropout=mlp_dropout, mlp_batch_norm=mlp_batch_norm)

    def preprocess(self, train_set, valid_set, test_set):
        weight = []
        for task, w in self.task.items():
            weight.append(w)

        self.register_buffer("weight", torch.as_tensor(weight, dtype=torch.float))
        print("weight", weight)
        self.num_class = [1]

        hidden_dims = [self.model.output_dim] * (self.num_mlp_layer - 1)
        self.mlp_1 = layers.MLP(self.model.output_dim, hidden_dims + [sum(self.num_class)], dropout=self.mlp_dropout, batch_norm=self.mlp_batch_norm)
    
    def predict(self, batch, all_loss=None, metric=None):
        graph1 = (batch["graph1"], batch["graph1_k"])
        graph2 = batch["graph2"]

        if self.graph_construction_model:
            graph2 = self.graph_construction_model(graph2)

        position = batch["position"]
        plddt = batch["plddt"]
        input = (graph1, position, plddt)

        output = self.model(graph2, input, all_loss=all_loss, metric=metric) 

        pred_1 = self.mlp_1(output["graph_feature"])
        pred = pred_1
        return pred
    

    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred = self.predict(batch, all_loss, metric)

        if all([t not in batch for t in self.task]):
            return all_loss, metric

        target = self.target(batch)
        labeled = ~torch.isnan(target)
        target[~labeled] = 0

        for criterion, weight in self.criterion.items():
            if criterion == "mse":
                if self.normalization:
                    loss = F.mse_loss((pred - self.mean) / self.std, (target - self.mean) / self.std, reduction="none")
                else:
                    loss = F.mse_loss(pred, target, reduction="none")
            elif criterion == "bce":
                loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
                loss = functional.masked_mean(loss, labeled, dim=0)
            elif criterion == "ce":
                loss = F.cross_entropy(pred, target.long().squeeze(-1), reduction="none").unsqueeze(-1)
            else:
                raise ValueError("Unknown criterion `%s`" % criterion)

            name = tasks._get_criterion_name(criterion)
            if self.verbose > 0:
                for t, l in zip(self.task, loss):
                    metric["%s [%s]" % (name, t)] = l
            loss = (loss * self.weight).sum() / self.weight.sum()
            metric[name] = loss
            all_loss += loss * weight
        
        for _metric in self.metric:
            if _metric == "acc":
                score = []
                num_class = 0
                for i, cur_num_class in enumerate(self.num_class):
                    _pred = torch.sigmoid(pred)
                    _pred = (_pred > 0.5).float()
                    _pred = torch.squeeze(_pred)
                    _target = target[:, i]
                    _labeled = labeled[:, i]

                    result = torch.eq(_pred[_labeled], _target[_labeled].float())
                    _accuracy = result.sum().float()
                    _score = _accuracy / len(_pred[_labeled])
                    score.append(_score)
                    num_class += cur_num_class
                score = torch.stack(score)
            elif _metric == "auroc":
                score = []
                for _pred, _target, _labeled in zip(pred.t(), target.long().t(), labeled.t()):
                    _score = metrics.area_under_roc(_pred[_labeled], _target[_labeled])
                    score.append(_score)
                score = torch.stack(score)
            elif _metric == "auprc":
                score = []
                for _pred, _target, _labeled in zip(pred.t(), target.long().t(), labeled.t()):
                    _score = metrics.area_under_prc(_pred[_labeled], _target[_labeled])
                    score.append(_score)
                score = torch.stack(score)
            elif _metric == "fpr":
                score = []
                for _pred, _target, _labeled in zip(pred.t(), target.long().t(), labeled.t()):
                    _pred = torch.sigmoid(_pred[_labeled])
                    _target = _target[_labeled]
                    _fpr = (_pred[_target == 0] > 0.5).sum().float() / (_target == 0).sum().float()
                    score.append(_fpr)
                score = torch.stack(score)
            else:
                raise ValueError("Unknown metric `%s`" % _metric)

            name = tasks._get_metric_name(_metric)
            for t, s in zip(self.task, score):
                metric["%s [%s]" % (name, t)] = s

        return all_loss, metric

    def target(self, batch):
        target = torch.stack([batch[t].float() for t in self.task], dim=-1)
        labeled = batch.get("labeled", torch.ones(len(target), dtype=torch.bool, device=target.device)) 
        target[~labeled] = math.nan
        return target

    def evaluate(self, pred, target):
        labeled = ~torch.isnan(target)

        metric = {}

        for _metric in self.metric:
            if _metric == "mae":
                score = F.l1_loss(pred, target, reduction="none")
                score = functional.masked_mean(score, labeled, dim=0)
            elif _metric == "rmse":
                score = F.mse_loss(pred, target, reduction="none")
                score = functional.masked_mean(score, labeled, dim=0).sqrt()
            elif _metric == "acc":
                score = []
                num_class = 0
                for i, cur_num_class in enumerate(self.num_class):
                    _pred = torch.sigmoid(pred)
                    _pred = (_pred > 0.5).float()
                    _pred = torch.squeeze(_pred)
                    _target = target[:, i]
                    _labeled = labeled[:, i]

                    result = torch.eq(_pred[_labeled], _target[_labeled].float())
                    _accuracy = result.sum().float()
                    _score = _accuracy / len(_pred[_labeled])
                    score.append(_score)
                    num_class += cur_num_class
                score = torch.stack(score)
            elif _metric == "mcc":
                score = []
                num_class = 0
                for i, cur_num_class in enumerate(self.num_class):
                    _pred = pred[:, num_class:num_class + cur_num_class]
                    _target = target[:, i]
                    _labeled = labeled[:, i]
                    _score = metrics.matthews_corrcoef(_pred[_labeled], _target[_labeled].long())
                    score.append(_score)
                    num_class += cur_num_class
                score = torch.stack(score)
            elif _metric == "auroc":
                score = []
                for _pred, _target, _labeled in zip(pred.t(), target.long().t(), labeled.t()):
                    _score = metrics.area_under_roc(_pred[_labeled], _target[_labeled])
                    score.append(_score)
                score = torch.stack(score)
            elif _metric == "auprc":
                score = []
                for _pred, _target, _labeled in zip(pred.t(), target.long().t(), labeled.t()):
                    _score = metrics.area_under_prc(_pred[_labeled], _target[_labeled])
                    score.append(_score)
                score = torch.stack(score)
            elif _metric == "fpr":
                score = []
                for _pred, _target, _labeled in zip(pred.t(), target.long().t(), labeled.t()):
                    _pred = torch.sigmoid(_pred[_labeled])
                    _target = _target[_labeled]
                    _fpr = (_pred[_target == 0] > 0.5).sum().float() / (_target == 0).sum().float()
                    score.append(_fpr)
                score = torch.stack(score)
            elif _metric == "r2":
                score = []
                for _pred, _target, _labeled in zip(pred.t(), target.t(), labeled.t()):
                    _score = metrics.r2(_pred[_labeled], _target[_labeled])
                    score.append(_score)
                score = torch.stack(score)
            elif _metric == "spearmanr":
                score = []
                for _pred, _target, _labeled in zip(pred.t(), target.t(), labeled.t()):
                    _score = metrics.spearmanr(_pred[_labeled], _target[_labeled])
                    score.append(_score)
                score = torch.stack(score)
            elif _metric == "pearsonr":
                score = []
                for _pred, _target, _labeled in zip(pred.t(), target.t(), labeled.t()):
                    _score = metrics.pearsonr(_pred[_labeled], _target[_labeled])
                    score.append(_score)
                score = torch.stack(score)
            else:
                raise ValueError("Unknown metric `%s`" % _metric)

            name = tasks._get_metric_name(_metric)
            for t, s in zip(self.task, score):
                metric["%s [%s]" % (name, t)] = s
        # add loss
        metric["loss"] = F.binary_cross_entropy_with_logits(pred, target, reduction="none").mean()
        return metric


class GLU(nn.Module):
    def __init__(self,hidden_size):
        super().__init__()
        self.linear_proj = nn.Linear(hidden_size,hidden_size,bias=False)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.act1 = nn.GELU()
        self.act2 = nn.functional.silu
        self.dense_h_to_4h = nn.Linear(hidden_size,hidden_size*4,bias=False)
        self.gate_proj = nn.Linear(hidden_size,hidden_size*4,bias=False)
        self.dense_4h_to_h = nn.Linear(hidden_size*4,hidden_size,bias=False)

    def forward(self,x):
        x = self.linear_proj(x)
        x = self.act1(self.norm1(x))
        x = self.act2(self.gate_proj(x))*self.dense_h_to_4h(x)
        x = self.dense_4h_to_h(x) # x.shape = (batch_size,seq_len,hidden_size)
        return x

class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout):
        super(MultiLayerPerceptron, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        # add dropout
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(self.dropout)
            input_dim = hidden_dim
        self.layers.append(nn.Linear(input_dim,input_dim))
        self.layers.append(nn.ReLU())
        self.layers.append(self.dropout)
        self.layers.append(nn.Linear(input_dim, output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        return x