import os
import glob
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset
from pyunicorn.timeseries.surrogates import Surrogates

def load_sequences(root_dir):
    """"
    Lê todos os CSVs em 'correto' e 'incorreto'
    Retorna X_coords [N, seq_len, n_feats], y [N].
    """
    dirs = [(os.path.join(root_dir, 'correto'), 1),(os.path.join(root_dir, 'incorreto'), 0)]
    sequences, labels = [], []
    for folder, lbl in dirs:
        for path in glob.glob(os.path.join(folder, '*.csv')):
            df = pd.read_csv(path) [300:] # Pula os primeiros 300 frames
            seq = df.values.astype(np.float32)
            sequences.append(seq)
            labels.append(lbl)
    X_coords = np.stack(sequences)
    y = np.array(labels)
    return X_coords, y

def calcula_surrogate(X_coords):
    """
    Aplica Surrogate em X_coords
    Retorna:
    arrays concatenados:([originais + surrogados], [labels duplicados]).
    """
    N, T, F = X_coords.shape
    X_sur = np.zeros_like(X_coords, dtype=np.float32)
    for i in range(N):
        # gerar um surrogate para cada feature f
        for f in range(F):
            original_serie = X_coords[i, :, f]            # shape (T,)
            lote = original_serie.reshape(1, T)           # shape (1, T)
            sur_gen = Surrogates(original_data=lote)     # método AAFT
            lote_sur = sur_gen.AAFT_surrogates()          # retorna array shape (1, T)
            X_sur[i, :, f] = lote_sur[0]  
            
    return X_sur
    
def calcula_smote(X_coords, y):
    """
    Aplica SMOTE em X_coords  
    Retorna:
      X_res: np.ndarray [N_res, seq_len, n_feats]
      y_res: np.ndarray [N_res]
    """
    N, T, F = X_coords.shape
    coords_flat  = X_coords.reshape(N, -1)
    sm = SMOTE(random_state=42, k_neighbors=3)
    coords_res, y_res = sm.fit_resample(coords_flat, y) #aplica o smote
    N_res = coords_res.shape[0]
    X_res = coords_res.reshape(N_res, T, F)
    return X_res, y_res

def create_windows(X_coords,y,window_size):
    """
    Divide cada sequência de X_coords em janelas não sobrepostas de tamanho window_size
    e retorna novos arrays X_windows, y_windows no mesmo formato (mas ampliado).
    """
    N, T, F = X_coords.shape
    n_win = T // window_size # calcula quantas janelas inteiras cabem em cada sequência original.
    # corta o excesso para termos múltiplos exatos de window_size
    T_trim = n_win * window_size
    X_trim = X_coords[:, :T_trim, :]                        # (N, n_win*window_size, F)
    # primeiro reshape para (N, n_win, window_size, F)
    X_reshaped = X_trim.reshape(N, n_win, window_size, F)
    # então para (N*n_win, window_size, F)
    X_windows = X_reshaped.reshape(-1, window_size, F)
    # repete os rótulos para cada janela
    y_windows = np.repeat(y, n_win)
    return X_windows, y_windows
            
class coordsDataset(Dataset):
    """
    Dataset usando apenas coordenadas: retorna x_coords, y.
    """
    def __init__(self, X_coords, y):
        #X_coords shape [N, T, F] N = número de vídeos, T = seq_len, F = número de features por frame)
        self.X_coords = torch.tensor(X_coords, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_coords[idx], self.y[idx]
