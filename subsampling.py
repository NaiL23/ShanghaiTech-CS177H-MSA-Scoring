import os
import shutil
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

def hh_filter(filename, input_path, output_path, diff = 256, verbose = 1) :
    os.system(f"hhfilter -i {input_path / filename} -o {output_path / filename} -diff {diff} -v {verbose}")

def filter_msa_data(msa_data, root = './dataset', target_depth = 256, is_train = True) :
    input_path = root / ("train" if is_train else "test")
    output_path = root / ("train_filtered" if is_train else "test_filtered")
    if not os.path.exists(output_path) : os.makedirs(output_path)
    for msa_name, msa in tqdm(msa_data.items()) :
        filename = msa_name + '.a3m'
        if len(msa['msa']) > target_depth :
            hh_filter(filename, input_path, output_path)
        else :
            shutil.copyfile(input_path / filename, output_path / filename)

def greedy_subsampling(msa, max_depth = 256):
    if len(msa) <= max_depth: return msa
    array = np.array([list(seq) for _, seq in msa], dtype=np.bytes_).view(np.uint8)
    all_indices = np.arange(len(msa))
    indices = [0]
    pairwise_distances = np.zeros((0, len(msa)))
    for _ in range(max_depth - 1):
        dist = cdist(array[indices[-1:]], array, "hamming")
        pairwise_distances = np.concatenate([pairwise_distances, dist])
        shifted_distance = np.delete(pairwise_distances, indices, axis=1).mean(0)
        shifted_index = np.argmax(shifted_distance)
        index = np.delete(all_indices, indices)[shifted_index]
        indices.append(index)
    indices = sorted(indices)
    return [msa[idx] for idx in indices]

def subsampling_msa_data(msa_data, max_depth = 256) :
    for msa in tqdm(msa_data.values()):
        if len(msa['msa']) > max_depth :
            msa['msa'] = greedy_subsampling(msa['msa'], max_depth)
    return msa_data

if __name__ == "__main__":
    from dataset import read_msa_data
    from pathlib import Path
    msa_data, _ = read_msa_data(Path() / "dataset" / "train_toy")

    depth = len(msa_data["T1024-D1_cov50_fm"]["msa"])
    print(f"before subsampling: {depth}")
    msa_data = subsampling_msa_data(msa_data)
    depth = len(msa_data["T1024-D1_cov50_fm"]["msa"])
    print(f"after subsampling: { depth }")