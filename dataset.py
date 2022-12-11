import os
import re
from Bio import SeqIO

from torch.utils.data import Dataset
from subsampling import subsampling_msa_data

import torch
import numpy as np

def process_sequence(sequence_raw) :
    return re.sub(r'[a-z]', '', sequence_raw)
    
def read_msa(filename) :
    return [(record.description, process_sequence(str(record.seq))) for record in SeqIO.parse(filename, "fasta")]

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

def GetEmbeddings(name, padding, EMBDEDDINGS_PATH):
    data = torch.load(EMBDEDDINGS_PATH /  (name + ".pt"), map_location=torch.device('cpu'))
    rest = padding - len(data)
    data = np.pad(data, ((0, rest), (0, 0)), 'constant')
    return torch.Tensor(data).cuda()

def GetAllEmbeddings(names, padding, EMBDEDDINGS_PATH):
    dic = {}
    for name in names:
        dic[name] = GetEmbeddings(name, padding, EMBDEDDINGS_PATH)

def GetRawEmbeddings(name, padding, EMBDEDDINGS_PATH):
    data = torch.load(EMBDEDDINGS_PATH /  (name + ".pt"))
    return data

def read_msa_data_(root = "./dataset", is_train = True, is_filtered = False) :
    path = root / (("train" if is_train else "test") + ("_filtered" if is_filtered else ""))
    target_file = root / ("train_set.txt" if is_train else "test_set.txt")
    
    msa_name_list = sorted([filename[:-4] for filename in os.listdir(path) if filename.endswith(".a3m")])
    msa_data = { msa_name : 0 for msa_name in msa_name_list }

    with open(target_file) as f:
        msa_scores = f.readlines()
    for msa_score in msa_scores :
        msa_name, score = msa_score.split(' ')
        msa_data[msa_name] = eval(score)
        
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

class EScoreDataset(Dataset):

    def __init__(self, embedding_path, root = "./dataset", is_train = True):
        self.msa_score, self.msa_name_list = read_msa_data_(root, is_train, is_filtered = True)
        self.path = embedding_path

    def __len__(self):
        return len(self.msa_name_list)

    def __getitem__(self, name):
        return GetEmbeddings(name, 584, self.path)

    def summary(self):
        print(self.msa_name_list[0])
        print(self.msa_score[self.msa_name_list[0]])