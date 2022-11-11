import torch
import torch.nn.functional as F
from torch import nn

import esm

class MSAPredictor(nn.Module):
    def __init__(self, msa_transformer_path):
        super(MSAPredictor, self).__init__()
        
        if msa_transformer_path:
            self.encoder, msa_alphabet = esm.pretrained.load_model_and_alphabet_local(msa_transformer_path)
        else :
            self.encoder, msa_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
        
        self.encoder = self.encoder.eval()
        self.batch_converter = msa_alphabet.get_batch_converter()

        # Freeze parameters of MSATransformer
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Regressor module (to be tested)
        # self.conv1 = nn.Conv2d(1, 6, 5)
        # self.pool = nn.MaxPool2d(3, 3)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(25232, 2048)
        # self.fc2 = nn.Linear(2048, 512)
        # self.fc3 = nn.Linear(512, 1)
        self.fc1 = nn.Linear(768, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, x):
        seq_length = [list(x[i, 0]).index(1) if 1 in x[i, 0] else x.size(2) for i in range(x.size(0))]
        
        self.encoder.eval()
        embeddings = []
        with torch.no_grad():
            for i in range(x.size(0)) :
                # Remove sequence paddings
                xi = x[i:i+1, :, :seq_length[i]]
                # 1 x NUM_SEQ x 1+SEQ_LEN

                # xx = xx[:, :list(x[i, :, 0]).index(1), :] if 1 in x[i, :, 0] else xx

                xi = self.encoder(xi, repr_layers=[12])["representations"][12][:, 0, 1:, :]
                # 1 x SEQ_LEN x 768

                xi = torch.mean(xi, dim = 1)
                # 1 x 768
                embeddings.append(xi)
                
        x = torch.vstack(embeddings)
        # BATHCH_SIZE x 768

        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))

        # x = torch.flatten(x, 1)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        # x = torch.sigmoid(self.fc3(x))
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

class MSAPredictorBOS(MSAPredictor):
    def __init__(self, msa_transformer_path):
        super(MSAPredictorBOS, self).__init__(msa_transformer_path)
        
    def forward(self, x):
        seq_length = [list(x[i, 0]).index(1) if 1 in x[i, 0] else x.size(2) for i in range(x.size(0))]
        
        self.encoder.eval()
        embeddings = []
        with torch.no_grad():
            for i in range(x.size(0)) :
                # Remove sequence paddings
                xi = x[i:i+1, :, :seq_length[i]]
                # 1 x NUM_SEQ x 1+SEQ_LEN

                xi = self.encoder(xi, repr_layers=[12])["representations"][12][:, 0, 0, :]
                # 1 x 768
                
                embeddings.append(xi)
                
        x = torch.vstack(embeddings)
        # BATHCH_SIZE x 768
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x