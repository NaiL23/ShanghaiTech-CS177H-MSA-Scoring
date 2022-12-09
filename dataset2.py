import torch
import numpy as np

def GetEmbeddings(name, padding, EMBDEDDINGS_PATH):
    data = torch.load(EMBDEDDINGS_PATH / (name + ".pt"))
    rest = padding - len(data)
    data = np.pad(data, ((0, rest), (0, 0)), 'constant')
    return data

def GetAllEmbeddings(names, padding, EMBDEDDINGS_PATH):
    dic = {}
    for name in names:
        dic[name] = GetEmbeddings(name, padding, EMBDEDDINGS_PATH)

    