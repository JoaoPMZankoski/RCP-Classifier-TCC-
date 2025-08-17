import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataset import (load_sequences,create_windows,calcula_surrogate,calcula_smote)

features = [
    ('ombro esquerdo', 0, 1),
    ('ombro direito',  2, 3),
    ('cotovelo esquerdo', 4, 5),
    ('cotovelo direito',  6, 7),
    ('pulso esquerdo',   8, 9),
    ('pulso direito',   10,11),
    ('quadril esquerdo',12,13),
    ('quadril direito', 14,15),
]

path = "mediapipekp_centered"
X_coords, y = load_sequences(path)
#X_coords, y = calcula_smote(X_coords, y)
#X_sur = calcula_surrogate(X_coords)
#X_coords = np.concatenate([X_coords, X_sur], axis=0) # originais + sinteticas:
#y = np.concatenate([y, y], axis=0)
X_coords, y = create_windows(X_coords, y, 128)
print("janelado:", X_coords.shape)
data = X_coords[0] #X_smote X_sur #video errado 31
"""
Plota subplots para cada feature:
    - x & y de cada no mesmo gráfico
"""
total = len(features)

fig, axes = plt.subplots(3, 4, figsize=(12, 12), sharex=True)
axes = axes.flatten()

# features
for i, (name, xi, yi) in enumerate(features):
    ax = axes[i]
    #ax.plot(data[:, xi], label=f'x {name}')
    ax.plot(data[:, yi], label=f'y {name}')
    ax.set_title(f'{name} (Y)')
    ax.legend(fontsize='x-small', loc='best')
    ax.grid(True)

#Remove eixos extras
for k in range(total, len(axes)):
    fig.delaxes(axes[k])

plt.tight_layout()
plt.show()


    
    
   

