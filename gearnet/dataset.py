import os
import math
import h5py
import glob
import random
import pickle
import logging
import warnings
from tqdm import tqdm
import subprocess

import numpy as np
import pandas as pd
import scipy
from sklearn.preprocessing import normalize

import torch
from torch.utils import data as torch_data
from torch.utils.data import IterableDataset

from torchdrug import data, utils, core, datasets
from torchdrug.layers import functional
from torchdrug.core import Registry as R
from torchdrug.utils import comm

from atom3d import datasets as da
from atom3d.datasets import LMDBDataset

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

logger = logging.getLogger(__name__)


class Atom3DDataset:

    def protein_from_data_frame(self, df, atom_feature=None, bond_feature=None, 
                                residue_feature="default", mol_feature=None):  
        assert bond_feature is None
        assert mol_feature is None
        atom_feature = data.Protein._standarize_option(atom_feature)
        bond_feature = data.Protein._standarize_option(bond_feature)
        mol_feature = data.Protein._standarize_option(mol_feature)
        residue_feature = data.Protein._standarize_option(residue_feature)
        
        atom2residue = []
        atom_type = []
        residue_type = []
        atom_name = []
        is_hetero_atom = []
        residue_number = []
        occupancy = []
        b_factor = []
        insertion_code = []
        chain_id = []
        node_position = []
        _residue_feature = []
        _atom_feature = []
        last_residue = None
        for i, atom in df.iterrows():
            atom_type.append(data.feature.atom_vocab.get(atom['element'], 0))
            type = atom['resname']
            number = atom['residue']
            code = atom['insertion_code']
            canonical_residue = (number, code, type)
            if canonical_residue != last_residue:
                last_residue = canonical_residue
                if type not in data.Protein.residue2id:
                    warnings.warn("Unknown residue `%s`. Treat as glycine" % type)
                    type = "GLY"
                residue_type.append(data.Protein.residue2id[type])
                residue_number.append(number)
                insertion_code.append(data.Protein.alphabet2id.get(code, 0))
                chain_id.append(data.Protein.alphabet2id.get(atom['chain'], 0))
                feature = []
                for name in residue_feature:
                    if name == "default":
                        feature = data.feature.onehot(type, data.feature.residue_vocab, allow_unknown=True)
                    else:
                        raise ValueError('Feature %s not included' % name)
                _residue_feature.append(feature)
            name = atom['name']
            if name not in data.Protein.atom_name2id:
                name = "UNK"
            atom_name.append(data.Protein.atom_name2id[name])
            is_hetero_atom.append(atom['hetero'] != ' ')
            occupancy.append(atom['occupancy'])
            b_factor.append(atom['bfactor'])
            node_position.append([atom['x'], atom['y'], atom['z']])
            atom2residue.append(len(residue_type) - 1)
            feature = []
            for name in atom_feature:
                if name == "residue_symbol":
                    feature += \
                        data.feature.onehot(atom['element'], data.feature.atom_vocab, allow_unknown=True) + \
                        data.feature.onehot(type, data.feature.residue_vocab, allow_unknown=True)
                else:
                    raise ValueError('Feature %s not included' % name)
            _atom_feature.append(feature)
        
        atom_type = torch.tensor(atom_type)
        residue_type = torch.tensor(residue_type)
        atom_name = torch.tensor(atom_name)
        is_hetero_atom = torch.tensor(is_hetero_atom)
        occupancy = torch.tensor(occupancy)
        b_factor = torch.tensor(b_factor)
        atom2residue = torch.tensor(atom2residue)
        residue_number = torch.tensor(residue_number)
        insertion_code = torch.tensor(insertion_code)
        chain_id = torch.tensor(chain_id)
        node_position = torch.tensor(node_position)
        if len(residue_feature) > 0:
            _residue_feature = torch.tensor(_residue_feature)
        else:
            _residue_feature = None
        if len(atom_feature) > 0:
            _atom_feature = torch.tensor(_atom_feature)
        else:
            _atom_feature = None

        return data.Protein(edge_list=None, num_node=len(atom_type), atom_type=atom_type, bond_type=[], 
                    residue_type=residue_type, atom_name=atom_name, atom2residue=atom2residue, 
                    is_hetero_atom=is_hetero_atom, occupancy=occupancy, b_factor=b_factor,
                    residue_number=residue_number, insertion_code=insertion_code, chain_id=chain_id, 
                    node_position=node_position, atom_feature=_atom_feature, residue_feature=_residue_feature)

    @torch.no_grad()
    def construct_graph(self, data_list, model=None, batch_size=1, gpus=None, verbose=True):
        protein_list = []
        if gpus is None:
            device = torch.device("cpu")
        else:
            device = torch.device(gpus[comm.get_rank() % len(gpus)])
        model = model.to(device)
        t = range(0, len(data_list), batch_size)
        if verbose:
            t = tqdm(t, desc="Constructing graphs for training")
        for start in t:
            end = start + batch_size
            batch = data_list[start:end]
            proteins = data.Protein.pack(batch).to(device)
            if gpus and hasattr(proteins, "residue_feature"):
                with proteins.residue():
                    proteins.residue_feature = proteins.residue_feature.to_dense()
            proteins = model(proteins).cpu()
            for protein in proteins:
                if gpus and hasattr(protein, "residue_feature"):
                    with protein.residue():
                        protein.residue_feature = protein.residue_feature.to_sparse()
                protein_list.append(protein)
        return protein_list

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = torch_data.Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        return splits


@R.register("datasets.PhosKinSubDataset")
class PhosKinSubDataset(data.ProteinDataset, Atom3DDataset):
    processed_file = "phos.pkl.gz"
    pdb_file_lists = []

    def __init__(self, path, protein_path, transform=None, lazy=False, verbose=1, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        self.splits = ["train", "valid", "test"]
        self.lazy = lazy
        self.transform = transform
        self.kwargs = kwargs
        self.data = []
        self.sequences = []
        self.pdb_files = []
        self.protein_path = protein_path

        csv_files = [os.path.join(path, "___new_%s_zj.csv" % split)
                     for split in self.splits]

        pkl_file = os.path.join(path, self.processed_file)

        if os.path.exists(pkl_file):
            print("has pkl files")
            self.load_pickle(pkl_file, transform=transform, lazy=False, verbose=verbose, **kwargs)
        else:
            self.transform = transform
            self.kwargs = kwargs

            for csv_file in csv_files:
                if "train" in csv_file:
                    split = "train"
                elif "valid" in csv_file:
                    split = "valid"
                elif "test" in csv_file:
                    split = "test" 
                self.load_csv(path, csv_file, pkl_file, lazy, verbose, split, **kwargs)

        splits = [os.path.basename(os.path.dirname(pdb_file)) for pdb_file in self.pdb_files]
        self.num_samples = [splits.count("train"), splits.count("valid"), splits.count("test")]
    
    def load_csv(self, path, csv_file, pkl_file, lazy, verbose=1, split="train", **kwargs):
        # read csv files
        dataset = pd.read_csv(csv_file)
        kin_id = dataset['kin_id']
        sub_id = dataset['sub_id']
        substrate = dataset['substrate']
        labels = dataset['label']
        kin_seq = dataset['kin_seq']
        sub_seq = dataset['sub_seq']
        position = dataset['position']
        short_seq = dataset['11_mer']
        plddt = dataset['plddt']
        
        datasets = []
        if verbose:
            dataset = tqdm(dataset, "Constructing pdbs from data frames")
        for i in range(len(dataset)):
            datasets.append([kin_id[i], sub_id[i], substrate[i], labels[i], kin_seq[i], short_seq[i], position[i], plddt[i]])

        for i, pdb_file_list in enumerate(datasets):
            if not lazy or i == 0:
                kin_id = pdb_file_list[0]
                sub_id = pdb_file_list[1]
                pdb_file_list[2] = os.path.basename(pdb_file_list[2])
                pdb_file_list[2] = os.path.join(self.protein_path, pdb_file_list[2])
                short_seq = pdb_file_list[5]

                father_path = os.path.dirname(self.protein_path)
                kin_seq_pickle = os.path.join(father_path, "sequence", kin_id+"_seq.pkl")
                sub_seq_pickle = os.path.join(father_path, "sequence", short_seq+"_seq.pkl")
                sub_pickle = os.path.join(father_path, "protein", "%s_sub.pkl" % (sub_id))

                if not os.path.exists(kin_seq_pickle):
                    sequence = pdb_file_list[4]
                    protein = data.Protein._residue_from_sequence(sequence)
                    with open(kin_seq_pickle, 'wb') as fout:
                        pickle.dump(protein, fout)
                        
                if not os.path.exists(sub_seq_pickle):
                    sequence = pdb_file_list[5]
                    protein = data.Protein._residue_from_sequence(sequence)
                    with open(sub_seq_pickle, 'wb') as fout:
                        pickle.dump(protein, fout)
                        
                if not os.path.exists(sub_pickle):
                    try:
                        mol_2 = Chem.MolFromPDBFile(pdb_file_list[2])
                        if not mol_2:
                            logger.debug("Can't construct molecule from pdb file `%s`. Ignore this sample." % pdb_file_list)
                            continue
                        protein_2 = data.Protein.from_molecule(mol_2, **kwargs)
                        if not protein_2:
                            logger.debug("Can't construct protein from pdb file `%s`. Ignore this sample." % pdb_file_list)
                            continue
                        with open(sub_pickle, 'wb') as fout:
                            pickle.dump(protein_2, fout)
                    except Exception as e:
                        print(f"Error occurred while reading file {sub_pickle}: {e}")
                        continue
    
                self.data.append((kin_id, sub_id, torch.tensor(pdb_file_list[3]), pdb_file_list[2], torch.tensor(pdb_file_list[6]), pdb_file_list[5], torch.tensor(pdb_file_list[7])))
                self.sequences.append(('s', 's'))
                self.pdb_files.append(os.path.join(path, split, str(i)))
        
        self.save_pickle(pkl_file, verbose=verbose)

    def split(self, keys=None):
        keys = keys or self.splits
        offset = 0
        splits = []
        for split_name, num_sample in zip(self.splits, self.num_samples):
            if split_name in keys:
                split = torch_data.Subset(self, range(offset, offset + num_sample))
                splits.append(split)
            offset += num_sample
        return splits

    def get_item(self, index):
        kin_id = self.data[index][0]
        sub_id = self.data[index][1]
        label = self.data[index][2]
        substrate = self.data[index][3]
        position = self.data[index][4]
        short_seq = self.data[index][5]
        plddt = self.data[index][6]
        
        father_path = os.path.dirname(self.protein_path)
        kin_seq_pickle = os.path.join(father_path, "sequence", kin_id+"_seq.pkl")
        sub_seq_pickle = os.path.join(father_path, "sequence", short_seq+"_seq.pkl")
        sub_pickle = os.path.join(father_path, "protein", "%s_sub.pkl" % (sub_id))

        try:
            with open(sub_pickle, 'rb') as fin:
                stru = pickle.load(fin)
        except Exception as e:
            print(f"Error occurred while reading file {sub_pickle}: {e}")
            stru = data.Protein.from_pdb(substrate, **self.kwargs)
        try:
            with open(kin_seq_pickle, 'rb') as fin:
                kin_seq_g = pickle.load(fin)
        except Exception as e:
            print(f"Error occurred while reading file {kin_seq_pickle}: {e}")
            kin_seq_g = data.Protein._residue_from_sequence("-")
        try:
            with open(sub_seq_pickle, 'rb') as fin:
                sub_seq_g = pickle.load(fin)
        except Exception as e:
            print(f"Error occurred while reading file {sub_seq_pickle}: {e}")
            sub_seq_g = data.Protein._residue_from_sequence(short_seq)
            
        if hasattr(stru, "residue_feature"):
            with stru.residue():
                stru.residue_feature = stru.residue_feature.to_dense()
            with kin_seq_g.residue():
                kin_seq_g.residue_feature = kin_seq_g.residue_feature.to_dense()
            with sub_seq_g.residue():
                sub_seq_g.residue_feature = sub_seq_g.residue_feature.to_dense()
        
        item = {"graph2": stru, "graph1": sub_seq_g, "graph1_k": kin_seq_g, "position": torch.tensor(position), "plddt": torch.tensor(plddt)}
        item["label"] = label
 
        if self.transform:
            item = self.transform(item)
        return item

    @property
    def tasks(self):
        """List of tasks."""
        return ["label"]

    def __repr__(self):
        lines = [
            "#sample: %d" % len(self),
            "#task: label",
        ]
        return "%s(\n  %s\n)" % (self.__class__.__name__, "\n  ".join(lines))