import os
import cv2
import numpy as np
from ultralytics import YOLO

def pixelate_face_in_frame(input_video_path,output_image_path,model_weights,target_frame_index,pixelation_scale,min_confidence):

    # 1) Carrega o modelo YOLOv8‐Face
    model = YOLO(model_weights)
    model.to('cpu')  # força o CPU; se tiver GPU e quiser usar, comente esta linha

    # 2) Abre o vídeo com OpenCV
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise IOError(f"Não foi possível abrir o vídeo: {input_video_path}")

    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_idx = 0
    saved_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx == target_frame_index:
            # Detecta rostos apenas neste frame
            results = model(frame)[0]  # primeiro objeto de detecção
            boxes   = results.boxes

            # Se não houver detecções de face, mantemos o frame inteiramente inalterado
            if boxes is not None and len(boxes) > 0:
                coords_xyxy = boxes.xyxy.cpu().numpy()   # array Nx4: [x1, y1, x2, y2]
                confidences = boxes.conf.cpu().numpy()   # array N com confiança de cada bbox
                classes     = boxes.cls.cpu().numpy().astype(int)  # array N (espera‐se classe “0” = face)

                # Para cada caixa, se for classe “face” e confiança ≥ min_confidence, pixeliza
                for (x1, y1, x2, y2), conf, cls in zip(coords_xyxy, confidences, classes):
                    if cls == 0 and conf >= min_confidence:
                        # Converte coordenadas para inteiros e limites válidos
                        x1i = max(0, int(x1))
                        y1i = max(0, int(y1))
                        x2i = min(width, int(x2))
                        y2i = min(height, int(y2))
                        roi = frame[y1i:y2i, x1i:x2i]
                        if roi.size == 0:
                            continue
                        # Pixelização: redimensiona para bem pequeno e volta ao tamanho original
                        h_roi, w_roi = roi.shape[:2]
                        small_w = max(1, int(w_roi * pixelation_scale))
                        small_h = max(1, int(h_roi * pixelation_scale))
                        # Reduz
                        temp = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
                        # Aumenta de volta com INTER_NEAREST para obter a “grade” pixelada
                        pixelated = cv2.resize(temp, (w_roi, h_roi), interpolation=cv2.INTER_NEAREST)
                        # Substitui o rosto original pela versão pixelizada
                        frame[y1i:y2i, x1i:x2i] = pixelated
            # Guarda o frame final (anonimizado) e interrompe o loop
            saved_frame = frame.copy()
            break

        frame_idx += 1

    cap.release()
    cv2.imwrite(output_image_path, saved_frame)
    print(f"Frame {target_frame_index} (com rosto pixelado) salvo em:\n  {output_image_path}")


if __name__ == "__main__":
    video_input = "videos/correto/01-certo.mkv"
    frame_output = "frame_face_anonimizado.png"
    target_idx = 100
    pixelate_face_in_frame(input_video_path = video_input,output_image_path= frame_output,model_weights= 'yolov8n-face-lindevs.pt',target_frame_index = target_idx,pixelation_scale= 0.05,min_confidence= 0.6)