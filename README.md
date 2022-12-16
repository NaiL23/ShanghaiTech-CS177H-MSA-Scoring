# ShanghaiTech-CS177H-MSA-Scoring

Author: Lian Yihang, Yang Xuchu & Teng Yue



## Content

- `explore.ipynb` pre-processing with HH-Filter; explore dataset & models
- `extract_embedding.ipynb` extract MSA embeddings using MSA Transformer
- `subsampling.py` subsampling utilities
- `dataset.py` dataset utilities (pointwise + pairwise)
- `mlp.ipynb` MLP models & training
- `CNN.ipynb` CNN models & training
- `dataset/` MSA data & dataset split file
- `embeddings/` extracted MSA embeddings
- `model/` training checkpoints
- `report/` presentations



## Reference 

- https://github.com/facebookresearch/esm MSA Transformer = ESM-MSA-1b

- https://github.com/axelmarmet/protein_protein_interaction MSA Transformer Example Jupyter Notebooks

  - https://github.com/axelmarmet/protein_protein_interaction/blob/main/src/msa_to_embeddings/MSA%20to%20embeddings.ipynb MSA Embeddings

  - https://github.com/axelmarmet/protein_protein_interaction/blob/main/src/ppi_pred/dataset/embedding_seq_dataset.py Embedding Feature Map Dataset

  - https://github.com/axelmarmet/protein_protein_interaction/blob/main/src/ppi_pred/dataset/pooling_baseline_dataset.py Embedding Vector Dataset
