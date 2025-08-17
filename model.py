import torch
import torch.nn as nn

class LSTMBase(nn.Module):
    """
    Input: x_coords [B, T, n_feats_coords]
    Output: logits [B,1]
    """
    def __init__(self,n_feats: int,hidden: int = 64,num_layers: int = 1,dropout: float = 0.0): #hidden = 32-256, num_layers 1-3, dropout 0.0-0.1
        super(LSTMBase, self).__init__()
        self.lstm = nn.LSTM(n_feats,hidden,num_layers=num_layers,batch_first=True,dropout=dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x_coords: torch.Tensor):
        # x_coords: [B, T, n_feats_coords]
        _, (h_n, _) = self.lstm(x_coords)
        h_last = h_n[-1]        # [B, hidden]
        return self.fc(h_last)  # [B,1] logits
