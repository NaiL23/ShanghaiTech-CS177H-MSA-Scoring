import os
import re
from Bio import SeqIO

from torch.utils.data import Dataset
from subsampling import subsampling_msa_data

import torch
import numpy as np
import random

def process_sequence(sequence_raw) :
    return re.sub(r'[a-z]', '', sequence_raw)
    
def read_msa(filename) :
    return [(record.description, process_sequence(str(record.seq))) for record in SeqIO.parse(filename, "fasta")]

def read_embedding(msa_name, padding_size, embedding_path):
    embedding = torch.load(embedding_path / (msa_name + ".pt"), map_location = torch.device('cpu'))
    rest = padding_size - len(embedding)
    embedding = np.pad(embedding, ((0, rest), (0, 0)), 'constant')
    return torch.Tensor(embedding)

class EmbeddingScoreDataset(Dataset):

    def __init__(self, embedding_path, root = "./dataset", is_train = True, padding_size = 584):
        self.embedding_path = embedding_path
        self.padding_size = padding_size

        target_file = root / ("train_set.txt" if is_train else "test_set.txt")
        
        with open(target_file) as f:
            msa_scores = f.readlines()
        
        self.msa_name_list = []
        self.msa_score_list = []
        for msa_score in msa_scores :
            msa_name, score = msa_score.split(' ')
            self.msa_name_list.append(msa_name)
            self.msa_score_list.append(eval(score) / 100.0)

    def __len__(self):
        return len(self.msa_name_list)

    def __getitem__(self, idx):
        msa_name, msa_score = self.msa_name_list[idx], self.msa_score_list[idx]
        msa_embedding = read_embedding(msa_name, self.padding_size, self.embedding_path)
        return {
            'name' : msa_name,
            'score' : msa_score,
            'embedding' : msa_embedding
        }

def generate_pair_dataset(root = './dataset', is_train = True, num_pair_per_msa = 2) :
    if is_train :
        with open(root / 'train_set.txt') as f :
            msa_lines = f.readlines()

        msa_data = []
        for msa_line in msa_lines :
            msa_name, msa_score = msa_line.split()
            msa_data.append({
                'name' : msa_name,
                'score' : eval(msa_score)
            })
        msa_class_list = list(set([msa['name'].split('_')[0] for msa in msa_data]))

        msa_data_by_class = {msa_class : [] for msa_class in msa_class_list }
        for msa in msa_data:
            msa_class = msa['name'].split('_')[0]
            msa_data_by_class[msa_class].append(msa)
        
        with open(root / 'train_set_pair.txt', 'w') as f :
            for msa_class in msa_class_list:
                class_msas = msa_data_by_class[msa_class]
                for msa1 in class_msas :
                    msa1_name, msa1_score = msa1['name'], msa1['score']
                    msa2_candidates = random.sample([msa for msa in class_msas if msa['name'] != msa1_name], num_pair_per_msa)
                    for msa2 in msa2_candidates:
                        msa2_name, msa2_score = msa2['name'], msa2['score']
                        f.write(' '.join([msa1_name, str(msa1_score), msa2_name, str(msa2_score)]) + '\n')
    else :
        assert num_pair_per_msa == 2
        with open(root / 'test_set.txt') as f :
            msa_lines = f.readlines()
        with open(root / 'test_set_pair.txt', 'w') as f :
            for i in range(len(msa_lines) // 2) :
                msa1_name, msa1_score = msa_lines[2 * i].split()
                msa2_name, msa2_score = msa_lines[2 * i + 1].split()
                f.write(' '.join([msa1_name, msa1_score, msa2_name, msa2_score]) + '\n')

class PairDataset(Dataset):
    def __init__(self, embedding_path, root = "./dataset", is_train = True, padding_size = 584):
        self.embedding_path = embedding_path
        self.padding_size = padding_size
        
        target_file = root / ("train_set_pair.txt" if is_train else "test_set_pair.txt")
        if not os.path.exists(target_file) :
            generate_pair_dataset(root, is_train, num_pair_per_msa = 5 if is_train else 2)
        
        with open(target_file) as f:
            pair_lines = f.readlines()
        
        self.msa_name_list = []
        self.msa_score_list = []
        for pair_line in pair_lines:
            msa1_name, msa1_score, msa2_name, msa2_score = pair_line.split()
            self.msa_name_list.append((msa1_name, msa2_name))
            self.msa_score_list.append((eval(msa1_score) / 100.0, eval(msa2_score) / 100.0))

    def __len__(self):
        return len(self.msa_name_list)

    def __getitem__(self, idx):
        (msa1_name, msa2_name), (msa1_score, msa2_score) = self.msa_name_list[idx], self.msa_score_list[idx]
        msa_embedding1 = read_embedding(msa1_name, self.padding_size, self.embedding_path)
        msa_embedding2 = read_embedding(msa2_name, self.padding_size, self.embedding_path)
        return {
            'name1' : msa1_name,
            'name2' : msa2_name,
            'score1' : msa1_score,
            'score2' : msa2_score,
            'embedding1' : msa_embedding1,
            'embedding2' : msa_embedding2
        }


    
# ---------------------------------------------------------------------------- #
#                                  Deprecated                                  #
# ---------------------------------------------------------------------------- #

def read_msa_data(root = "./dataset", is_train = True, is_filtered = False) :
    path = root / (("train" if is_train else "test") + ("_filtered" if is_filtered else ""))
    target_file = root / ("train_set.txt" if is_train else "test_set.txt")
    
    msa_name_list = sorted([filename[:-4] for filename in os.listdir(path) if filename.endswith(".a3m")])
    msa_data = { msa_name : {'msa': read_msa(path / (msa_name+".a3m")), 'name' : msa_name } for msa_name in msa_name_list }
    
    with open(target_file) as f:
        msa_scores = f.readlines()
    for msa_score in msa_scores :
        msa_name, score = msa_score.split(' ')
        msa_data[msa_name]['score'] = eval(score)
    
    return msa_data, msa_name_list

class MSAScoreDataset(Dataset):
    def __init__(self, root = "./dataset", is_train = True):
        self.msa_data, self.msa_name_list = read_msa_data(root, is_train, is_filtered = True)
        self.msa_data = subsampling_msa_data(self.msa_data)

    def __len__(self):
        return len(self.msa_data)

    def __getitem__(self, idx):
        return self.msa_data[self.msa_name_list[idx]]

    def summary(self, head = 10, last = 10) :
        print(f"Dataset has { self.__len__() } MSAs:")
        for msa_name in self.msa_name_list[:head] :
            msa = self.msa_data[msa_name]
            print(f"> MSA {msa_name} has { len(msa['msa']) } sequences of length { len(msa['msa'][0][1]) }, scored {msa['score']:.3f}")
        print("......")
        for msa_name in self.msa_name_list[-last:] :
            msa = self.msa_data[msa_name]
            print(f"> MSA {msa_name} has { len(msa['msa']) } sequences of length { len(msa['msa'][0][1]) }, scored {msa['score']:.3f}")