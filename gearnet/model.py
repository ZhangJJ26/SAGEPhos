from collections.abc import Sequence

import os
import os.path as osp
import esm
import warnings
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from torch_scatter import scatter_add

from torchdrug import core, layers, utils, data, models
from torchdrug.core import Registry as R
from torchdrug.layers import functional

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import random

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        self.scaling = self.head_dim ** -0.5

        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Project input tensors
        Q = self.query_proj(query)
        K = self.key_proj(key)
        V = self.value_proj(value)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scaling

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        attention_probs = F.softmax(attention_scores, dim=-1)

        context = torch.matmul(attention_probs, V)

        # Concatenate heads and put through final linear layer
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        output = self.out_proj(context)

        return output

# normalize
def dic_normalize(dic):
    # print(dic)
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    # print(max_value)
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic

residue_symbol2id = {"G": 0, "A": 1, "S": 2, "P": 3, "V": 4, "T": 5, "C": 6, "I": 7, "L": 8, "N": 9,
                     "D": 10, "Q": 11, "K": 12, "E": 13, "M": 14, "H": 15, "F": 16, "R": 17, "Y": 18, "W": 19, "X": 20}
id2residue_symbol = {v: k for k, v in residue_symbol2id.items()}

pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
pro_res_aromatic_table = ['F', 'W', 'Y']
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
pro_res_acidic_charged_table = ['D', 'E']
pro_res_basic_charged_table = ['H', 'K', 'R']

res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}

res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}

res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}

res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}

res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}

res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49}

res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63}

res_weight_table = dic_normalize(res_weight_table)
res_pka_table = dic_normalize(res_pka_table)
res_pkb_table = dic_normalize(res_pkb_table)
res_pkx_table = dic_normalize(res_pkx_table)
res_pl_table = dic_normalize(res_pl_table)
res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)


def residue_features(residue):
    res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 
                     1 if residue in pro_res_aromatic_table else 0,
                     1 if residue in pro_res_acidic_charged_table else 0,
                     1 if residue in pro_res_basic_charged_table else 0]

    # res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue], res_pkx_table[residue],
    #                  res_pl_table[residue], res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]
    # print(np.array(res_property1 + res_property2).shape)
    # return np.array(res_property1 + res_property2)
    return np.array(res_property1)


@R.register("models.FusionNetwork_two_fusion")
class FusionNetwork_two_fusion(nn.Module, core.Configurable):

    def __init__(self, sequence_model, structure_model, fusion="series", cross_dim=None):
        super(FusionNetwork_two_fusion, self).__init__()
        self.sequence_model = sequence_model
        self.structure_model = structure_model
        self.fusion = fusion
        self.a = torch.nn.Parameter(torch.tensor([1.0, 1.0, 1.0, 1.0]))
        self._a = torch.nn.Parameter(torch.tensor([1.0, 1.0, 1.0, 1.0]))
        
        seq_expand_dim = sequence_model.output_dim + structure_model.output_dim
        self.batch_norm_esm = torch.nn.Sequential(torch.nn.Linear(sequence_model.output_dim, seq_expand_dim))
        stru_expand_dim = structure_model.output_dim
        self.batch_norm_stru = torch.nn.Sequential(torch.nn.BatchNorm1d(stru_expand_dim),
                                                   torch.nn.Linear(stru_expand_dim, structure_model.output_dim))

        self.node_linear = torch.nn.Linear(sequence_model.output_dim * 2, sequence_model.output_dim)

        embed_dim = sequence_model.output_dim+ structure_model.output_dim
        self.gated_feature_composer = torch.nn.Sequential(torch.nn.BatchNorm1d(embed_dim), torch.nn.ReLU(),
            torch.nn.Linear(embed_dim, sequence_model.output_dim))
        self.res_info_composer = torch.nn.Sequential(torch.nn.BatchNorm1d(embed_dim), torch.nn.ReLU(),
            torch.nn.Linear(embed_dim, embed_dim), torch.nn.ReLU(),
            torch.nn.Linear(embed_dim, sequence_model.output_dim))         
        self.phos_site_emb = nn.Embedding(1, embed_dim)
        
        _embed_dim = sequence_model.output_dim * 2 + 4
        self._gated_feature_composer = torch.nn.Sequential(torch.nn.BatchNorm1d(_embed_dim), torch.nn.ReLU(),
            torch.nn.Linear(_embed_dim, sequence_model.output_dim))
        self._res_info_composer = torch.nn.Sequential(torch.nn.BatchNorm1d(_embed_dim), torch.nn.ReLU(),
            torch.nn.Linear(_embed_dim, _embed_dim), torch.nn.ReLU(),
            torch.nn.Linear(_embed_dim, sequence_model.output_dim))
        self._phos_site_emb = nn.Embedding(1, _embed_dim)
        
        if fusion in ["series", "parallel"]:
            self.output_dim = sequence_model.output_dim
        elif fusion == "cross":
            self.seq_linear = nn.Linear(sequence_model.output_dim, cross_dim)
            self.struct_linear = nn.Linear(structure_model.output_dim, cross_dim)
            self.attn = layers.SelfAttentionBlock(cross_dim, num_heads=8, dropout=0.1)
            self.output_dim = cross_dim * 2
        else:
            raise ValueError("Not support fusion scheme %s" % fusion)

    def forward(self, graph, input, all_loss=None, metric=None):
        k = 11
        # Sequence model
        graph1 = input[0]
        kin_graph = graph1[1]
        sub_graph = graph1[0]
        position = input[1]
        plddt = input[2]
        kin_output = self.sequence_model(kin_graph, position, all_loss, metric) 
        node_output1_kin = kin_output.get("node_feature", kin_output.get("residue_feature"))[:, 1:, :]
        node_output1_kin = torch.mean(node_output1_kin, dim=1)
        # expand
        node_output1_kin = node_output1_kin.unsqueeze(1).expand(-1, 11, -1)
        
        sub_output = self.sequence_model(sub_graph, position, all_loss, metric) 
        node_output1_sub = sub_output.get("node_feature", sub_output.get("residue_feature"))[:, 1:k+1, :]
        # add phy_che property
        raw_seq_id = sub_graph.residue_type
        input2graph = sub_graph.residue2graph
        unique_classes = input2graph.unique() # tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        result = {}
        for cls in unique_classes:
            class_mask = (input2graph == cls)
            class_data = raw_seq_id[class_mask]
            select_data = class_data[:k]
            result[cls.item()] = select_data
        raw_seq_id = torch.stack([result[i] for i in range(len(result))])
        
        result = {}
        for j in range(len(raw_seq_id)):
            pro_property = np.zeros((k, 4))
            # pro_property = np.zeros((k, 11))
            for i in range(k):
                id = raw_seq_id[j, i].int().item()
                raw_seq = id2residue_symbol[id]
                pro_property[i,] = residue_features(raw_seq)
            pro_property = torch.tensor(pro_property).float().to(graph.device)
            result[j] = pro_property
        raw_seq = torch.stack([result[i] for i in range(len(result))])
        
        node_output1 = node_output1_sub
        
        input = graph.node_feature.float()
        # Structure model
        if self.fusion == "series":
            input = node_output1
        output2 = self.structure_model(graph, input, position, all_loss, metric)
        node_output2 = output2.get("node_feature", output2.get("node_feature"))
        
        # Fusion
        if self.fusion in ["series", "parallel"]:
            node_feature = node_output1
            gf1 = node_output1
            result = {}
            for i in range(len(position)):
                posi = position[i]
                if posi-k//2 < 0:
                    padding_l = torch.zeros((k//2-posi, node_output2.shape[2])).to(graph.device)
                    one_gf2 = torch.cat([padding_l, node_output2[i, :posi+k//2+1, :]], dim=0)
                elif posi+k//2+1 > node_output2[i].shape[0]:
                    padding_r = torch.zeros((posi+k//2+1-node_output2[i].shape[0], node_output2.shape[2])).to(graph.device)
                    one_gf2 = torch.cat([node_output2[i, posi-k//2:, :], padding_r], dim=0)
                else:
                    one_gf2 = node_output2[i, posi-k//2:posi+k//2+1, :]
                result[i] = one_gf2
            gf2 = torch.stack([result[i] for i in range(len(position))])

            graph_feature = torch.cat([gf1, gf2], dim=-1)
            graph_feature[:, k//2, :] += self.phos_site_emb.weight[0][None, :]
            batch_size, sequence_length, num_features = graph_feature.size()
            graph_feature = graph_feature.view(batch_size * sequence_length, num_features)
            f1 = self.gated_feature_composer(graph_feature)
            f1 = f1.view(batch_size, sequence_length, -1)
            f2 = self.res_info_composer(graph_feature)
            f2 = f2.view(batch_size, sequence_length, -1)
            graph_feature = F.sigmoid(f1) * gf1 * self.a[0] + f2 * self.a[1]
            
            _gf1 = graph_feature
            _gf2 = node_output1_kin
            _graph_feature = torch.cat([_gf1, _gf2, raw_seq], dim=-1)
            batch_size, sequence_length, num_features = _graph_feature.size()
            _graph_feature = _graph_feature.view(batch_size * sequence_length, num_features)
            _f1 = self._gated_feature_composer(_graph_feature)
            _f1 = _f1.view(batch_size, sequence_length, -1)
            _f2 = self._res_info_composer(_graph_feature)
            _f2 = _f2.view(batch_size, sequence_length, -1)
            _graph_feature = F.sigmoid(_f1) * _gf1 * self._a[0] + _f2 * self._a[1]
            
            graph_feature = _graph_feature.sum(1)
            
        else:
            seq_output = self.seq_linear(node_output1)
            struct_output = self.struct_linear(node_output2)
            attn_input, sizes = functional._extend(seq_output, graph.num_residues, struct_output, graph.num_residues)
            attn_input, mask = functional.variadic_to_padded(attn_input, sizes)
            attn_output = self.attn(attn_input, mask)
            node_feature = functional.padded_to_variadic(attn_output, sizes)
            seq_index = torch.arange(graph.num_residue, dtype=torch.long, device=graph.device)
            num_cum_residues = torch.cat([torch.zeros((1,), dtype=torch.long, device=graph.device), graph.num_cum_residues])
            seq_index += num_cum_residues[graph.residue2graph]
            struct_index = torch.arange(graph.num_residue, dtype=torch.long, device=graph.device)
            struct_index += graph.num_cum_residues[graph.residue2graph]
            node_feature = torch.cat([node_feature[seq_index], node_feature[struct_index]], dim=-1)
            graph_feature = scatter_add(node_feature, graph.residue2graph, dim=0, dim_size=graph.batch_size)

        return {
            "graph_feature": graph_feature,
            "node_feature": node_feature
        }


@R.register("models.ESM_modified")
class ESM_modified(nn.Module, core.Configurable):
    """
    The protein language model, Evolutionary Scale Modeling (ESM) proposed in
    `Biological Structure and Function Emerge from Scaling Unsupervised Learning to 250 Million Protein Sequences`_.

    .. _Biological Structure and Function Emerge from Scaling Unsupervised Learning to 250 Million Protein Sequences:
        https://www.biorxiv.org/content/10.1101/622803v1.full.pdf

    Parameters:
        path (str): path to store ESM model weights
        model (str, optional): model name. Available model names are ``ESM-1b``, ``ESM-1v`` and ``ESM-1b-regression``.
        readout (str, optional): readout function. Available functions are ``sum`` and ``mean``.
    """

    url = {
        "ESM-1b": "https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S.pt",
        "ESM-1v": "https://dl.fbaipublicfiles.com/fair-esm/models/esm1v_t33_650M_UR90S_1.pt",
        "ESM-1b-regression":
            "https://dl.fbaipublicfiles.com/fair-esm/regression/esm1b_t33_650M_UR50S-contact-regression.pt",
        "ESM-2-8M": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t6_8M_UR50D.pt",
        "ESM-2-35M": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t12_35M_UR50D.pt",
        "ESM-2-150M": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t30_150M_UR50D.pt",
        "ESM-2-650M": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt",
        "ESM-2-3B": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt",
        "ESM-2-15B": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t48_15B_UR50D.pt",
    }

    md5 = {
        "ESM-1b": "ba8914bc3358cae2254ebc8874ee67f6",
        "ESM-1v": "1f04c2d2636b02b544ecb5fbbef8fefd",
        "ESM-1b-regression": "e7fe626dfd516fb6824bd1d30192bdb1",
        "ESM-2-8M": "8039fc9cee7f71cd2633b13b5a38ff50",
        "ESM-2-35M": "a894ddb31522e511e1273abb23b5f974",
        "ESM-2-150M": "229fcf8f9f3d4d442215662ca001b906",
        "ESM-2-650M": "ba6d997e29db07a2ad9dca20e024b102",
        "ESM-2-3B": "d37a0d0dbe7431e48a72072b9180b16b",
        "ESM-2-15B": "af61a9c0b792ae50e244cde443b7f4ac",
    }

    output_dim = {
        "ESM-1b": 1280,
        "ESM-1v": 1280,
        "ESM-2-8M": 320,
        "ESM-2-35M": 480,
        "ESM-2-150M": 640,
        "ESM-2-650M": 1280,
        "ESM-2-3B": 2560,
        "ESM-2-15B": 5120,
    }

    num_layer = {
        "ESM-1b": 33,
        "ESM-1v": 33,
        "ESM-2-8M": 6,
        "ESM-2-35M": 12,
        "ESM-2-150M": 30,
        "ESM-2-650M": 33,
        "ESM-2-3B": 36,
        "ESM-2-15B": 48,
    }

    max_input_length = 1024 - 2

    def __init__(self, path, model="ESM-1b", readout="mean"):
        super(ESM_modified, self).__init__()
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        _model, alphabet = self.load_weight(path, model)
        self.alphabet = alphabet
        mapping = self.construct_mapping(alphabet)
        self.output_dim = self.output_dim[model]
        self.model = _model
        self.alphabet = alphabet
        self.repr_layer = self.num_layer[model]
        self.register_buffer("mapping", mapping)

        if readout == "sum":
            self.readout = layers.SumReadout("residue")
        elif readout == "mean":
            self.readout = layers.MeanReadout("residue")
        elif readout == "site":
            self.readout = "site"
        else:
            raise ValueError("Unknown readout `%s`" % readout)

    def load_weight(self, path, model):
        if model not in self.url:
            raise ValueError("Unknown model `%s`" % model)
        model_file = utils.download(self.url[model], path, md5=self.md5[model])
        model_data = torch.load(model_file, map_location="cpu")
        if model != "ESM-1v" and not model.startswith("ESM-2"):
            regression_model = "%s-regression" % model
            regression_file = utils.download(self.url[regression_model], path, md5=self.md5[regression_model])
            regression_data = torch.load(regression_file, map_location="cpu")
        else:
            regression_data = None
        model_name = os.path.basename(self.url[model]) # 
        return esm.pretrained.load_model_and_alphabet_core(model_name, model_data, regression_data)

    def construct_mapping(self, alphabet):
        mapping = [-1] * max(len(data.Protein.id2residue_symbol), len(self.alphabet))
        for i, token in data.Protein.id2residue_symbol.items():
            mapping[i] = alphabet.get_idx(token) # get_idx返回token在alphabet中的索引
        mapping = torch.tensor(mapping)
        return mapping

    def forward(self, graph, input, all_loss=None, metric=None):
        """
        Compute the residue representations and the graph representation(s).

        Parameters:
            graph (Protein): :math:`n` protein(s)
            input (Tensor): input positions
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict

        Returns:
            dict with ``residue_feature`` and ``graph_feature`` fields:
                residue representations of shape :math:`(|V_{res}|, d)`, graph representations of shape :math:`(n, d)`
        """
        positions = input
        input = graph.residue_type
        input = self.mapping[input]
        input[input == -1] = graph.residue_type[input == -1]
        size = graph.num_residues
        if (size > self.max_input_length).any():
            warnings.warn("ESM can only encode proteins within %d residues. Truncate the input to fit into ESM."
                          % self.max_input_length)
            starts = size.cumsum(0) - size
            size = size.clamp(max=self.max_input_length)
            ends = starts + size
            mask = functional.multi_slice_mask(starts, ends, graph.num_residue)
            input = input[mask]
            graph = graph.subresidue(mask)
        size_ext = size
        if self.alphabet.prepend_bos:
            bos = torch.ones(graph.batch_size, dtype=torch.long, device=self.device) * self.alphabet.cls_idx
            input, size_ext = functional._extend(bos, torch.ones_like(size_ext), input, size_ext)
        if self.alphabet.append_eos:
            eos = torch.ones(graph.batch_size, dtype=torch.long, device=self.device) * self.alphabet.eos_idx
            input, size_ext = functional._extend(input, size_ext, eos, torch.ones_like(size_ext))
        input = functional.variadic_to_padded(input, size_ext, value=self.alphabet.padding_idx)[0] # padding_idx=0

        output = self.model(input, repr_layers=[self.repr_layer])
        residue_feature = output["representations"][self.repr_layer]
        residue_feature = functional.padded_to_variadic(residue_feature, size_ext)
        starts = size_ext.cumsum(0) - size_ext
        if self.alphabet.prepend_bos:
            starts = starts + 1
        ends = starts + size
        mask = functional.multi_slice_mask(starts, ends, len(residue_feature))
        residue_feature = residue_feature[mask]
        if self.readout == "site":
            input2graph = graph.residue2graph
            unique_classes = input2graph.unique() # tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            result = {}
            result_2 = {}

            for cls in unique_classes:
                class_mask = (input2graph == cls)
                class_data = residue_feature[class_mask]
                padding_class_data = torch.cat([class_data, torch.zeros((1024-class_data.shape[0], class_data.shape[1])).to(graph.device)], 0)
                result_2[cls.item()] = padding_class_data

            residue_feature = torch.stack([result_2[cls.item()] for cls in unique_classes])
            graph_feature = residue_feature.mean(1)
        else:
            graph_feature = self.readout(graph, residue_feature)

        return {
            "graph_feature": graph_feature,
            "residue_feature": residue_feature,
        }


        
@R.register("models.GCN_modified")
class GraphConvolutionalNetwork_m(nn.Module, core.Configurable):
    """
    Graph Convolutional Network proposed in `Semi-Supervised Classification with Graph Convolutional Networks`_.

    .. _Semi-Supervised Classification with Graph Convolutional Networks:
        https://arxiv.org/pdf/1609.02907.pdf

    Parameters:
        input_dim (int): input dimension
        hidden_dims (list of int): hidden dimensions
        edge_input_dim (int, optional): dimension of edge features
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        concat_hidden (bool, optional): concat hidden representations from all layers as output
        readout (str, optional): readout function. Available functions are ``sum`` and ``mean``.
    """

    def __init__(self, input_dim, hidden_dims, edge_input_dim=None, short_cut=False, batch_norm=False,
                 activation="relu", concat_hidden=False, readout="sum"):
        super(GraphConvolutionalNetwork_m, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.output_dim = sum(hidden_dims) if concat_hidden else hidden_dims[-1]
        self.dims = [input_dim] + list(hidden_dims)
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(layers.GraphConv(self.dims[i], self.dims[i + 1], edge_input_dim, batch_norm, activation))

        if readout == "sum":
            self.readout = layers.SumReadout()
        elif readout == "mean":
            self.readout = layers.MeanReadout()
        elif readout == "site":
            self.readout = "site"
        else:
            raise ValueError("Unknown readout `%s`" % readout)

    def forward(self, graph, input, positions, all_loss=None, metric=None):
        """
        Compute the node representations and the graph representation(s).

        Parameters:
            graph (Graph): :math:`n` graph(s)
            input (Tensor): input node representations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict

        Returns:
            dict with ``node_feature`` and ``graph_feature`` fields:
                node representations of shape :math:`(|V|, d)`, graph representations of shape :math:`(n, d)`
        """
        hiddens = []
        layer_input = input

        for layer in self.layers:
            hidden = layer(graph, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]
        # graph_feature = self.readout(graph, node_feature)
        if self.readout == "site":
            atom2residue = graph.atom2residue
            residue_feature = torch.zeros((graph.num_residue, node_feature.shape[1])).to(graph.device)
            for i in range(graph.num_residue):
                mask = (atom2residue == i)
                residue_feature[i] = node_feature[mask].mean(0)
            input2graph = graph.residue2graph
            unique_classes = input2graph.unique()
            result = {}
            result_2 = {}
            for cls in unique_classes:
                class_mask = (input2graph == cls)
                class_data = residue_feature[class_mask]
                
                padding_class_data = torch.cat([class_data, torch.zeros((1024-class_data.shape[0], class_data.shape[1])).to(graph.device)], 0)
                result_2[cls.item()] = padding_class_data
                position = positions[cls.item()]
                selected_data = class_data[position, :]
                result[cls.item()] = selected_data
            graph_feature = torch.stack([result[cls.item()] for cls in unique_classes])
            node_feature = torch.stack([result_2[cls.item()] for cls in unique_classes])
        else:
            graph_feature = self.readout(graph, node_feature)

        return {
            "graph_feature": graph_feature,
            "node_feature": node_feature
        }


@R.register("models.RGCN_modified")
class RelationalGraphConvolutionalNetwork_m(nn.Module, core.Configurable):
    """
    Relational Graph Convolutional Network proposed in `Modeling Relational Data with Graph Convolutional Networks?`_.

    .. _Modeling Relational Data with Graph Convolutional Networks?:
        https://arxiv.org/pdf/1703.06103.pdf

    Parameters:
        input_dim (int): input dimension
        hidden_dims (list of int): hidden dimensions
        num_relation (int): number of relations
        edge_input_dim (int, optional): dimension of edge features
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        concat_hidden (bool, optional): concat hidden representations from all layers as output
        readout (str, optional): readout function. Available functions are ``sum`` and ``mean``.
    """

    def __init__(self, input_dim, hidden_dims, num_relation, edge_input_dim=None, short_cut=False, batch_norm=False,
                 activation="relu", concat_hidden=False, readout="sum"):
        super(RelationalGraphConvolutionalNetwork_m, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.output_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1)
        self.dims = [input_dim] + list(hidden_dims)
        self.num_relation = num_relation
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(layers.RelationalGraphConv(self.dims[i], self.dims[i + 1], num_relation, edge_input_dim,
                                                          batch_norm, activation))

        if readout == "sum":
            self.readout = layers.SumReadout()
        elif readout == "mean":
            self.readout = layers.MeanReadout()
        elif readout == "site":
            self.readout = "site"
        else:
            raise ValueError("Unknown readout `%s`" % readout)

    def forward(self, graph, input, positions, all_loss=None, metric=None, flag=True):
        """
        Compute the node representations and the graph representation(s).

        Require the graph(s) to have the same number of relations as this module.

        Parameters:
            graph (Graph): :math:`n` graph(s)
            input (Tensor): input node representations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict

        Returns:
            dict with ``node_feature`` and ``graph_feature`` fields:
                node representations of shape :math:`(|V|, d)`, graph representations of shape :math:`(n, d)`
        """
        hiddens = []
        layer_input = input
        
        print("graph.num_node", graph.num_node)
        print("graph.num_residue", graph.num_residue)

        for layer in self.layers:
            hidden = layer(graph, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]

        if self.readout == "site":
            atom2residue = graph.atom2residue
            residue_feature = torch.zeros((graph.num_residue, node_feature.shape[1])).to(graph.device)
            for i in range(graph.num_residue):
                mask = (atom2residue == i)
                residue_feature[i] = node_feature[mask].mean(0)
            input2graph = graph.residue2graph
            unique_classes = input2graph.unique()
            result = {}
            result_2 = {}
            for cls in unique_classes:
                class_mask = (input2graph == cls)
                class_data = residue_feature[class_mask]
                
                padding_class_data = torch.cat([class_data, torch.zeros((1024-class_data.shape[0], class_data.shape[1])).to(graph.device)], 0)
                result_2[cls.item()] = padding_class_data
                if flag:
                    position = positions[cls.item()]
                    selected_data = class_data[position, :]
                    result[cls.item()] = selected_data
            if flag:
                graph_feature = torch.stack([result[cls.item()] for cls in unique_classes])
            else:
                graph_feature = torch.zeros((graph.num_residue, node_feature.shape[1])).to(graph.device)
            node_feature = torch.stack([result_2[cls.item()] for cls in unique_classes])
        else:
            graph_feature = self.readout(graph, node_feature)

        return {
            "graph_feature": graph_feature,
            "node_feature": node_feature
        }

        
@R.register("models.GIN_modified")
class GraphIsomorphismNetwork_m(nn.Module, core.Configurable):
    """
    Graph Ismorphism Network proposed in `How Powerful are Graph Neural Networks?`_

    .. _How Powerful are Graph Neural Networks?:
        https://arxiv.org/pdf/1810.00826.pdf

    Parameters:
        input_dim (int): input dimension
        hidden_dims (list of int): hidden dimensions
        edge_input_dim (int, optional): dimension of edge features
        num_mlp_layer (int, optional): number of MLP layers
        eps (int, optional): initial epsilon
        learn_eps (bool, optional): learn epsilon or not
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        concat_hidden (bool, optional): concat hidden representations from all layers as output
        readout (str, optional): readout function. Available functions are ``sum`` and ``mean``.
    """

    def __init__(self, input_dim, hidden_dims, edge_input_dim=None, num_mlp_layer=2, eps=0, learn_eps=False,
                 short_cut=False, batch_norm=False, activation="relu", concat_hidden=False,
                 readout="sum"):
        super(GraphIsomorphismNetwork_m, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.output_dim = sum(hidden_dims) if concat_hidden else hidden_dims[-1]
        self.dims = [input_dim] + list(hidden_dims)
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            layer_hidden_dims = [self.dims[i + 1]] * (num_mlp_layer - 1)
            self.layers.append(layers.GraphIsomorphismConv(self.dims[i], self.dims[i + 1], edge_input_dim,
                                                           layer_hidden_dims, eps, learn_eps, batch_norm, activation))

        if readout == "sum":
            self.readout = layers.SumReadout()
        elif readout == "mean":
            self.readout = layers.MeanReadout()
        elif readout == "site":
            self.readout = "site"
        else:
            raise ValueError("Unknown readout `%s`" % readout)

    def forward(self, graph, input, positions, all_loss=None, metric=None):
        """
        Compute the node representations and the graph representation(s).

        Parameters:
            graph (Graph): :math:`n` graph(s)
            input (Tensor): input node representations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict

        Returns:
            dict with ``node_feature`` and ``graph_feature`` fields:
                node representations of shape :math:`(|V|, d)`, graph representations of shape :math:`(n, d)`
        """
        hiddens = []
        layer_input = input

        for layer in self.layers:
            hidden = layer(graph, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]
        
        if self.readout == "site":
            atom2residue = graph.atom2residue
            residue_feature = torch.zeros((graph.num_residue, node_feature.shape[1])).to(graph.device)
            for i in range(graph.num_residue):
                mask = (atom2residue == i)
                residue_feature[i] = node_feature[mask].mean(0)
            input2graph = graph.residue2graph
            unique_classes = input2graph.unique()
            result = {}
            result_2 = {}
            for cls in unique_classes:
                class_mask = (input2graph == cls)
                class_data = residue_feature[class_mask]
                
                padding_class_data = torch.cat([class_data, torch.zeros((1024-class_data.shape[0], class_data.shape[1])).to(graph.device)], 0)
                result_2[cls.item()] = padding_class_data
                position = positions[cls.item()]
                selected_data = class_data[position, :]
                result[cls.item()] = selected_data
            graph_feature = torch.stack([result[cls.item()] for cls in unique_classes])
            node_feature = torch.stack([result_2[cls.item()] for cls in unique_classes])
        else:
            graph_feature = self.readout(graph, node_feature)

        return {
            "graph_feature": graph_feature,
            "node_feature": node_feature
        }
        
@R.register("models.GAT_modified")
class GraphAttentionNetwork_m(nn.Module, core.Configurable):
    """
    Graph Attention Network proposed in `Graph Attention Networks`_.

    .. _Graph Attention Networks:
        https://arxiv.org/pdf/1710.10903.pdf

    Parameters:
        input_dim (int): input dimension
        hidden_dims (list of int): hidden dimensions
        edge_input_dim (int, optional): dimension of edge features
        num_head (int, optional): number of attention heads
        negative_slope (float, optional): negative slope of leaky relu activation
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        concat_hidden (bool, optional): concat hidden representations from all layers as output
        readout (str, optional): readout function. Available functions are ``sum`` and ``mean``.
    """

    def __init__(self, input_dim, hidden_dims, edge_input_dim=None, num_head=1, negative_slope=0.2, short_cut=False,
                 batch_norm=False, activation="relu", concat_hidden=False, readout="sum"):
        super(GraphAttentionNetwork_m, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.output_dim = sum(hidden_dims) if concat_hidden else hidden_dims[-1]
        self.dims = [input_dim] + list(hidden_dims)
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(layers.GraphAttentionConv(self.dims[i], self.dims[i + 1], edge_input_dim, num_head,
                                                         negative_slope, batch_norm, activation))

        if readout == "sum":
            self.readout = layers.SumReadout()
        elif readout == "mean":
            self.readout = layers.MeanReadout()
        elif readout == "site":
            self.readout = "site"
        else:
            raise ValueError("Unknown readout `%s`" % readout)

    def forward(self, graph, input, positions, all_loss=None, metric=None):
        """
        Compute the node representations and the graph representation(s).

        Parameters:
            graph (Graph): :math:`n` graph(s)
            input (Tensor): input node representations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict

        Returns:
            dict with ``node_feature`` and ``graph_feature`` fields:
                node representations of shape :math:`(|V|, d)`, graph representations of shape :math:`(n, d)`
        """
        hiddens = []
        layer_input = input

        for layer in self.layers:
            hidden = layer(graph, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]
        
        if self.readout == "site":
            atom2residue = graph.atom2residue
            residue_feature = torch.zeros((graph.num_residue, node_feature.shape[1])).to(graph.device)
            for i in range(graph.num_residue):
                mask = (atom2residue == i)
                residue_feature[i] = node_feature[mask].mean(0)
            input2graph = graph.residue2graph
            unique_classes = input2graph.unique()
            result = {}
            result_2 = {}
            for cls in unique_classes:
                class_mask = (input2graph == cls)
                class_data = residue_feature[class_mask]
                
                padding_class_data = torch.cat([class_data, torch.zeros((1024-class_data.shape[0], class_data.shape[1])).to(graph.device)], 0)
                result_2[cls.item()] = padding_class_data
                position = positions[cls.item()]
                selected_data = class_data[position, :]
                result[cls.item()] = selected_data
            graph_feature = torch.stack([result[cls.item()] for cls in unique_classes])
            node_feature = torch.stack([result_2[cls.item()] for cls in unique_classes])
        else:
            graph_feature = self.readout(graph, node_feature)

        return {
            "graph_feature": graph_feature,
            "node_feature": node_feature
        }