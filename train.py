import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from dataset import (load_sequences,calcula_smote,calcula_surrogate,create_windows,coordsDataset)
from model import LSTMBase

#Configurações
ROOT_DIR    = "yolokp_centered" #yolokp
#K-FOLD
TEST_SIZE   = 0.10 
N_SPLITS = 10 
HIDDEN_SIZE = 128
NUM_LAYERS = 3
DROPOUT = 0.1
BATCH_SIZE  = 16 #8-16-32-64
LR          = 0.0001875401322749408
EPOCHS      = 60 #10-100
WINDOW_SIZE = 512  # 128 ou 512

device = 'cuda' if torch.cuda.is_available() else 'cpu'

X_coords, y = load_sequences(ROOT_DIR) #Carrega sequências
X_coords, y = calcula_smote(X_coords, y) #SMOTE
X_sur = calcula_surrogate(X_coords) #Surrogate(Aumento de dados)
X_coords = np.concatenate([X_coords, X_sur], axis=0) # originais + sinteticas:
y = np.concatenate([y, y], axis=0)
X_coords, y = create_windows(X_coords, y, WINDOW_SIZE)
print("janelado:", X_coords.shape)
#Separe 20% para TESTE antes de normalizar
idx = np.arange(len(y))
trainval_idx, test_idx = train_test_split(idx,test_size=TEST_SIZE,stratify=y,random_state=42)
X_trainval = X_coords[trainval_idx]   # 80% dos dados
y_trainval = y[trainval_idx]
X_test     = X_coords[test_idx]       # 20% do total
y_test     = y[test_idx]

# NORMALIZAÇÃO
_, seq_len, n_feats = X_trainval.shape
scaler = StandardScaler().fit(X_trainval.reshape(-1, n_feats))
X_trainval = scaler.transform(X_trainval.reshape(-1, n_feats)).reshape(-1, seq_len, n_feats)
X_test     = scaler.transform(X_test.reshape(-1, n_feats)).reshape(-1, seq_len, n_feats)

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# Variáveis para acumular métricas
precisions = []
recalls    = []
f1s        = []
cms        = []   # lista de matrizes de confusão

fold = 1
for train_idx, val_idx in skf.split(X_coords, y):
    print(f"\nFold {fold} / {N_SPLITS}")
    # Separa dados de treino e validação para este fold
    X_tr, y_tr = X_coords[train_idx], y[train_idx]
    X_va, y_va = X_coords[val_idx],   y[val_idx]
    # Cria Datasets e DataLoaders
    ds_tr = coordsDataset(X_tr, y_tr)
    ds_va = coordsDataset(X_va, y_va)
    train_dl = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False)

    #Modelo, loss e otimizador
    model  = LSTMBase(n_feats=n_feats,hidden=HIDDEN_SIZE,num_layers=NUM_LAYERS,dropout=DROPOUT).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    #Loop de treino
    for epoch in range(1, EPOCHS+1):
        model.train()
        for xr,yb in train_dl:
            xr, yb = xr.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xr)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
    # Validação
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for xb, yb in val_dl:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)
            trues = yb.numpy().flatten().astype(int)
            all_preds.extend(preds.tolist())
            all_trues.extend(trues.tolist())
            
    # Calcula métricas para este fold       
    cm   = confusion_matrix(all_trues, all_preds, labels=[0,1])
    prec = precision_score(all_trues, all_preds, zero_division=0)
    rec  = recall_score(all_trues, all_preds, zero_division=0)
    f1   = f1_score(all_trues, all_preds, zero_division=0)
    print("Matriz de Confusão:")
    print(cm)
    print(f"Precision: {prec:.4f}   Recall: {rec:.4f}   F1‐Score: {f1:.4f}")
    # Armazena métricas
    precisions.append(prec)
    recalls.append(rec)
    f1s.append(f1)
    cms.append(cm)
    fold += 1 #proximo fold

mean_prec = np.mean(precisions) #media
std_prec  = np.std(precisions) #erro
mean_rec  = np.mean(recalls)
std_rec   = np.std(recalls)
mean_f1   = np.mean(f1s)
std_f1    = np.std(f1s)
cm_sum    = sum(cms)
print(f"\n{N_SPLITS}-Fold CV: MÉDIAS")
print(f"Precisão média: {mean_prec:.4f}  ±  {std_prec:.4f}")
print(f"Recall    média: {mean_rec:.4f}  ±  {std_rec:.4f}")
print(f"F1-Score  média: {mean_f1:.4f}  ±  {std_f1:.4f}")
print("\nMatriz de Confusão (soma dos folds):")
print(cm_sum)

# Treino final em todos os 90% e avaliação no 10% de teste
print("\nTreino final:")
ds_trainval = coordsDataset(X_trainval, y_trainval)
ds_test = coordsDataset(X_test, y_test)
trainval_dl = DataLoader(ds_trainval, batch_size=BATCH_SIZE, shuffle=True)
test_dl = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

# modelo final
final_model = LSTMBase(n_feats=n_feats,hidden=HIDDEN_SIZE,num_layers=NUM_LAYERS,dropout=DROPOUT).to(device)
optimizer = optim.Adam(final_model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()

for epoch in range(1, EPOCHS + 1):
    final_model.train()
    running_loss = 0.0
    for xr, yb in trainval_dl:
        xr, yb = xr.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = final_model(xr)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * yb.size(0)
    if epoch % 5 == 0 or epoch == EPOCHS:
        print(f"[Final] Ep {epoch}/{EPOCHS}  Loss: {running_loss/len(ds_trainval):.4f}")

# Avaliação final no conjunto de teste
final_model.eval()
test_preds, test_trues = [], []
with torch.no_grad():
    for xb, yb in test_dl:
        xb = xb.to(device)
        logits = final_model(xb)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        preds = (probs > 0.5).astype(int)
        trues = yb.numpy().flatten().astype(int)
        test_preds.extend(preds.tolist())
        test_trues.extend(trues.tolist())

cm_test = confusion_matrix(test_trues, test_preds, labels=[0, 1])
print("\nMatriz de Confusão (teste 10%):")
print(cm_test)
print("\nClassification Report (teste 10%):")
print(classification_report(test_trues, test_preds, zero_division=0))