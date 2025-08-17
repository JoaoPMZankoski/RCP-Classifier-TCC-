import os
import glob
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

RESIZE_W, RESIZE_H = 640, 360
SEQ_LEN            = 3600   # número fixo de frames por vídeo
MODEL_WEIGHTS      = 'yolov8n-pose.pt'
KP_IDX = [5, 6, 7, 8, 9, 10, 11, 12]

# Critérios para filtrar detecções “ruins”
MIN_CONFIDENCE_PER_KP = 0.3   # cada keypoint deve ter conf ≥ 0.3
MIN_KPS_NEEDED        = 3     # no mínimo 3 keypoints confiáveis no frame
MIN_HEIGHT_PX         = 20    # altura mínima (em px na imagem redimensionada) para aceitar

JUMP_THRESHOLD_PX = 50.0

# Carrega o modelo YOLOv8-pose
model = YOLO(MODEL_WEIGHTS)
model.to('cpu')


def process_video(video_path, seq_len: int = SEQ_LEN):
    """
    Extrai keypoints do vídeo `video_path` e salva um CSV sem pulsos anômalos.
    """
    cls = os.path.basename(os.path.dirname(video_path))  # “correto” ou “incorreto”
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    out_dir = os.path.join("yolokp", cls)
    os.makedirs(out_dir, exist_ok=True)
    out_csv    = os.path.join(out_dir, f"{video_name}.csv")
    avgvis_csv = os.path.join(out_dir, f"{video_name}_vis.csv")

    n_kp    = len(KP_IDX)
    vis_sum = np.zeros(n_kp, dtype=np.float32)
    valid   = 0
    coords  = []  # aqui guardamos, para cada frame, um vetor de shape (2*n_kp,)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERRO: não pôde abrir {video_path}")
        return

    last_valid_coords = None  # vetores do último frame “aceito” (2*n_kp)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Redimensiona e converte para RGB
        frame_resized = cv2.resize(frame, (RESIZE_W, RESIZE_H))
        img = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        results = model(img)
        res  = results[0]

        candidatos = []

        # Para cada detecção “(esqueleto)”
        for kps_tensor in res.keypoints:
            coords_xy = kps_tensor.xy.cpu().numpy()[0]   # shape = (17, 2)
            confs     = kps_tensor.conf.cpu().numpy()[0] # shape = (17,)

            # Seleciona só os KP_IDX
            sel_coords = coords_xy[KP_IDX, :]   # shape = (n_kp, 2)
            sel_confs  = confs[KP_IDX]          # shape = (n_kp,)

            # Checa quantos keypoints têm confiança ≥ MIN_CONFIDENCE_PER_KP
            mask_conf_ok = sel_confs >= MIN_CONFIDENCE_PER_KP
            if mask_conf_ok.sum() < MIN_KPS_NEEDED:
                continue  # descartar, pois muito poucos pontos confiáveis

            # Calcula altura apenas sobre pontos confiáveis
            coords_ok = sel_coords[mask_conf_ok, :]
            height_ok = float(coords_ok[:, 1].max() - coords_ok[:, 1].min())
            if height_ok < MIN_HEIGHT_PX:
                continue  # descartar, pois muito “achatado”

            # Este candidato passou nos filtros; armazena
            candidatos.append((height_ok, sel_coords, sel_confs))

        # Se existir ao menos um candidato, escolhe o de maior altura
        if candidatos:
            candidatos.sort(key=lambda x: x[0], reverse=True)
            _, chosen_coords, chosen_confs = candidatos[0]
            chosen_flat = chosen_coords.reshape(-1).astype(np.float32)  # vetor (2*n_kp,)

            # Se já tivermos um last_valid_coords, checar salto anômalo
            if last_valid_coords is not None:
                diff = np.abs(chosen_flat - last_valid_coords)
                if np.any(diff > JUMP_THRESHOLD_PX):
                    # salto > JUMP_THRESHOLD_PX em pelo menos um canal →  
                    # NÃO aceitamos esse candidato, fazemos forward-fill:
                    coords.append(last_valid_coords.copy())
                    continue
                else:
                    # salto razoável: aceita a nova detecção
                    coords.append(chosen_flat)
                    last_valid_coords = chosen_flat.copy()
                    vis_sum += chosen_confs
                    valid += 1
            else:
                # ainda não havia frame válido antes: aceitamos a primeira detecção
                coords.append(chosen_flat)
                last_valid_coords = chosen_flat.copy()
                vis_sum += chosen_confs
                valid += 1

        else:
            # NÃO há nenhum candidato válido neste frame
            if last_valid_coords is not None:
                # Forward-fill do último frame válido
                coords.append(last_valid_coords.copy())
            else:
                # Nem sequer tivemos um frame válido antes: jogamos um único vetor zero
                coords.append(np.zeros(n_kp * 2, dtype=np.float32))
                # last_valid_coords permanece None até o primeiro candidato bom

    cap.release()

    # Converte lista de T vetores (2*n_kp) em array (T, 2*n_kp)
    seq = np.array(coords, dtype=np.float32)
    N = seq.shape[0]
    print(f"{video_name}: frames lidos = {N}")

    # Ajusta para comprimento fixo: seq_len
    if N >= seq_len:
        seq = seq[:seq_len]
    else:
        # Se já achamos ao menos um frame válido, fazemos forward-fill do último válido
        if last_valid_coords is not None:
            pad = np.tile(last_valid_coords.reshape(1, -1), (seq_len - N, 1))
        else:
            # Caso muito raro: nunca houve frame válido em TODO o vídeo
            pad = np.zeros((seq_len - N, n_kp * 2), dtype=np.float32)
        seq = np.vstack([seq, pad])

    # (Segurança extra) Preenche eventuais “linhas zero” com forward-fill
    def interpolate_zeros_per_frame(arr: np.ndarray) -> np.ndarray:
        arr_filled = arr.copy()
        zero_rows = np.all(arr == 0, axis=1)
        valid_rows = np.where(~zero_rows)[0]
        if valid_rows.size == 0:
            return arr_filled
        for i in np.where(zero_rows)[0]:
            prev_idxs = valid_rows[valid_rows < i]
            next_idxs = valid_rows[valid_rows > i]
            if prev_idxs.size > 0:
                arr_filled[i, :] = arr_filled[prev_idxs[-1], :]
            elif next_idxs.size > 0:
                arr_filled[i, :] = arr_filled[next_idxs[0], :]
        return arr_filled

    seq = interpolate_zeros_per_frame(seq)

    # Gera cabeçalhos “kp5_x, kp5_y, kp6_x, kp6_y, …, kp12_y”
    cols = []
    for k in KP_IDX:
        cols += [f"kp{k}_x", f"kp{k}_y"]

    # Salva CSV final
    pd.DataFrame(seq, columns=cols).to_csv(out_csv, index=False)

    #Salva CSV de visibilidade média 
    avg_vis = vis_sum / (valid if valid > 0 else 1)
    vis_cols = [f"kp{k}_v" for k in KP_IDX]
    pd.DataFrame([avg_vis], columns=vis_cols).to_csv(avgvis_csv, index=False)

    return seq, avg_vis


def main():
    path = "videos"
    for cls in ["correto", "incorreto"]:
        pattern = os.path.join(path, cls, "*.mkv")
        for video_path in glob.glob(pattern):
            process_video(video_path)
    print("Extração via YOLO concluída.")


if __name__ == "__main__":
    main()
