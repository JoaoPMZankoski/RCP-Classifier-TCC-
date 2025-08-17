import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import DataLoader
from dataset import load_sequences, calcula_smote, calcula_surrogate,create_windows, coordsDataset
from model   import LSTMBase

ROOT_DIR    = 'mediapipekp_centered'   # ou 'yolokp'
TEST_SIZE   = 0.10                     # 10% para teste final
N_SPLITS    = 10                        # 10-Fold CV fixo
EPOCHS_OPT  = 20                       # épocas de treino dentro de cada fold
WINDOW_SIZE = 256

device = 'cuda' if torch.cuda.is_available() else 'cpu'

X_coords, y = load_sequences(ROOT_DIR)
X_coords, y = calcula_smote(X_coords, y)
X_sur = calcula_surrogate(X_coords)
X_coords = np.concatenate([X_coords, X_sur], axis=0)
y = np.concatenate([y, y], axis=0)
X_coords, y = create_windows(X_coords, y, WINDOW_SIZE)
print("janelado:", X_coords.shape)

# test de 10%
idx = np.arange(len(y))
trainval_idx, test_idx = train_test_split(idx, test_size=TEST_SIZE, stratify=y, random_state=42)
X_trainval = X_coords[trainval_idx]   # 90% dos dados
y_trainval = y[trainval_idx]
X_test = X_coords[test_idx]       # 10% do total
y_test = y[test_idx]

# Normalização
_, seq_len, n_feats = X_trainval.shape
scaler = StandardScaler().fit(X_trainval.reshape(-1, n_feats))
X_trainval = scaler.transform(X_trainval.reshape(-1, n_feats)).reshape(-1, seq_len, n_feats)
X_test     = scaler.transform(X_test.reshape(-1, n_feats)).reshape(-1, seq_len, n_feats)

def estudo(trial):
    hidden_size = trial.suggest_categorical("hidden_size", [32,64,128,256])
    num_layers  = trial.suggest_int("num_layers", 1, 3)
    dropout     = trial.suggest_float("dropout", 0.0, 0.3, step=0.1)
    lr          = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size  = trial.suggest_categorical("batch_size", [16,32,64])

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    f1_scores = []

    for tr_idx, va_idx in skf.split(X_trainval, y_trainval):
        X_tr, y_tr = X_trainval[tr_idx], y_trainval[tr_idx]
        X_va, y_va = X_trainval[va_idx],   y_trainval[va_idx]

        ds_tr = coordsDataset(X_tr, y_tr)
        ds_va = coordsDataset(X_va, y_va)
        train_dl = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
        val_dl   = DataLoader(ds_va, batch_size=batch_size, shuffle=False)

        model = LSTMBase(n_feats=n_feats,
                        hidden=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Treino dentro do fold
        for _ in range(EPOCHS_OPT):
            model.train()
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

        # Validação
        model.eval()
        preds_list, trues_list = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(device)
                probs = torch.sigmoid(model(xb)).cpu().numpy().flatten()
                preds_list.extend((probs > 0.5).astype(int).tolist())
                trues_list.extend(yb.numpy().flatten().tolist())

        f1 = f1_score(trues_list, preds_list, zero_division=0)
        f1_scores.append(f1)
        trial.report(f1, len(f1_scores))
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return np.mean(f1_scores)

# Executa o estudo
study = optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=42),pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2))
study.optimize(estudo, n_trials=250)

print("RESUMO DA BUSCA")
print(f"Melhor F1 média ({N_SPLITS}-Fold CV): {study.best_value:.4f}")
print("Melhores hiperparâmetros:")
for key, val in study.best_params.items():
    print(f" {key}: {val}")

# Treino final com os melhores parâmetros
best = study.best_params
ds_trainval = coordsDataset(X_trainval, y_trainval)
ds_test     = coordsDataset(X_test, y_test)
trainval_dl = DataLoader(ds_trainval, batch_size=best["batch_size"], shuffle=True)
test_dl     = DataLoader(ds_test,     batch_size=best["batch_size"], shuffle=False)

final_model = LSTMBase(n_feats=n_feats,hidden=best["hidden_size"],num_layers=best["num_layers"],dropout=best["dropout"]).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(final_model.parameters(), lr=best["lr"])
EPOCHS_FINAL = 60

for epoch in range(1, EPOCHS_FINAL+1):
    final_model.train()
    for xb, yb in trainval_dl:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(final_model(xb), yb)
        loss.backward()
        optimizer.step()
    if epoch % 5 == 0 or epoch == EPOCHS_FINAL:
        print(f"[Final] Ep {epoch}/{EPOCHS_FINAL} concluída")

# Avaliação final
final_model.eval()
all_preds, all_trues = [], []
with torch.no_grad():
    for xb, yb in test_dl:
        xb = xb.to(device)
        probs = torch.sigmoid(final_model(xb)).cpu().numpy().flatten()
        all_preds.extend((probs > 0.5).astype(int).tolist())
        all_trues.extend(yb.numpy().flatten().tolist())

print(f"\nTest F1 (final): {f1_score(all_trues, all_preds):.4f}")
print("Matriz de Confusão (teste 10%):")
print(confusion_matrix(all_trues, all_preds, labels=[0, 1]))
print("\nClassification Report:")
print(classification_report(all_trues, all_preds, zero_division=0))