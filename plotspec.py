import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.signal import stft
from dataset import load_sequences, calcula_smote
import seaborn as sns

DATA_PATH = "yolokp_centered"       # Pasta com subpastas “correto/” e “incorreto/” contendo CSVs
FPS       = 30.0               # Frames por segundo do vídeo original
NPERSEG   = 128               # Tamanho da janela para STFT (em número de amostras)
NOVERLAP  = 64

FEATURES = [
    ("ombro esquerdo (y)",   1),
    #("ombro direito (y)",    3),
    ("cotovelo esquerdo (y)", 5),
    #("cotovelo direito (y)",  7),
    ("pulso esquerdo (y)",    9),
    #("pulso direito (y)",    11),
]

def freq_mean(seq: np.ndarray,fs: float = FPS,nperseg: int = NPERSEG,noverlap: int = NOVERLAP):
    f_bins, t_bins, Zxx = stft(seq,fs=fs,nperseg=nperseg,noverlap=noverlap,boundary=None,padded=False)
    mag = np.abs(Zxx) # matriz de magnitude do stft
    mag_no_dc = mag.copy()
    mag_no_dc[0, :] = 0.0 #zera o DC
    # Índice de pico para cada janela
    idx_peaks_hz = np.argmax(mag_no_dc, axis=0)   # shape = (n_janelas,)
    f_peaks_hz = f_bins[idx_peaks_hz]             # frequências em Hz
    return t_bins, f_peaks_hz, f_bins, mag

def plot_specs(video_index: int = 0):
    start_frame = 1500      # índice do primeiro frame a plotar
    end_frame   = 1900      # índice do último frame a plotar
    X_coords, y = load_sequences(DATA_PATH)
    data = X_coords[video_index]
    T, n_feats = data.shape
    n_features = len(FEATURES)
    fig, axes = plt.subplots(n_features, 2, figsize=(12, 4 * n_features))
    # Listas para coletar as métricas de todas as features
    for i, (label, idx) in enumerate(FEATURES):
        # Extrai coordenada Y do keypoint
        seq = data[:, idx].astype(float)
        seq = seq - seq.mean() # remove DC
        # Calcula média dos picos
        t_bins, f_peaks_hz, f_bins, mag = freq_mean(seq)
        f_peaks_cpm = f_peaks_hz * 60.0               # converte para CPM
        mean_peak = np.mean(f_peaks_cpm)
        std_peak  = np.std(f_peaks_cpm)
        ax_spec = axes[i, 0]
        f_bins_cpm = f_bins * 60.0
        #plot spectrograma
        ax_spec.pcolormesh(t_bins,f_bins_cpm,mag,shading='gouraud',cmap='inferno', )
        ax_spec.set_title(f"{label}( Media ≃ {mean_peak:.1f} ± {std_peak:.1f} CPM)")
        ax_spec.set_ylabel("Frequência (CPM)", fontsize = 8)
        ax_spec.set_xlabel("Tempo (s)", fontsize = 8)
        ax_spec.tick_params(axis='both', labelsize=8)
        ax_spec.set_ylim(0, f_bins_cpm.max())
        ax_spec.grid(False)
        # plot serie temporal
        ax_time = axes[i, 1]
        times_seq = np.arange(start_frame, end_frame) / FPS  # vetor de tempo em segundos para cada frame
        ax_time.plot(times_seq, seq[start_frame:end_frame], color="green", lw=0.8)
        ax_time.set_title(f"{label}(Y)")
        ax_time.set_xlabel("Tempo (s)", fontsize = 8)
        ax_time.set_xlim(start_frame / FPS, end_frame / FPS)
        ax_time.tick_params(axis='both', labelsize=8)
        ax_time.grid(True)
        t_bins, f_peaks_cpm, f_bins, mag = freq_mean(seq)
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Analisa o primeiro víd
    # eo (índice 0). Ajuste o índice conforme necessidade.
    plot_specs(video_index=1)
