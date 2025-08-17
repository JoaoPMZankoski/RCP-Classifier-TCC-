
import os
import glob
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

#Configurações de extração
RESIZE_W, RESIZE_H = 640, 360
SEQ_LEN = 3600 #2min 30fps
# Inicializa MediaPipe Pose
POSE = mp.solutions.pose.Pose(static_image_mode=False)
# Índices de keypoints: ombro L/R, cotovelo L/R, pulso L/R, quadril L/R
PTS  = [11, 12, 13, 14, 15, 16, 23, 24]

def extract_sequence(path):
    """
    Processa um vídeo e retorna array float32 de shape [3600, 16]
    para cada frame detectado, (x,y)
    """
    cap    = cv2.VideoCapture(path)
    frames = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # redimensiona e converte para RGB
        img = cv2.resize(frame, (RESIZE_W, RESIZE_H)) #640x360
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # detecta pose
        res = POSE.process(rgb)
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            coords = []
            # extrai (x,y)
            for p in PTS:
                coords.extend([lm[p].x, lm[p].y])
            frames.append(coords)
    cap.release()
    return np.array(frames, dtype=np.float32) # Retorna array de float (x1, y1...xn, yn)

def video_to_sequence(path, seq_len):
    """
    lê/escreve CSV ao lado do vídeo
    """
    # extrai classe e nome do vídeo
    cls = os.path.basename(os.path.dirname(path))   # 'correto' ou 'incorreto'
    video_name = os.path.splitext(os.path.basename(path))[0]
    out_dir   = os.path.join("mediapipekp", cls) 
    os.makedirs(out_dir, exist_ok=True)     # cria diretório de destino: mediapipekp/
    out_csv   = os.path.join(out_dir, f"{video_name}.csv")

    seq = extract_sequence(path)
    #Ajusta comprimento
    n = seq.shape[0]
    if n >= seq_len:
        seq = seq[:seq_len]
    #Salva
    cols = [f"feat_{i}" for i in range(seq.shape[1])]
    pd.DataFrame(seq, columns=cols).to_csv(out_csv, index=False)
    return seq

def main():
    path = "videos"
    for cls in ["correto", "incorreto"]:
        for video_path in glob.glob(os.path.join(path, cls, "*.mkv")):
            video_to_sequence(video_path, SEQ_LEN)
            print(f"Extração do {os.path.basename(video_path)} concluida")

if __name__ == "__main__":
    main()
