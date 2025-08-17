import os
import pandas as pd

# Pasta base onde estão os CSVs de visibilidade gerados pelo processo anterior
path = "yolokp/vis" 

# Índices COCO dos keypoints que usamos
KP_IDX     = [5, 6, 7, 8, 9, 10, 11, 12]
KP_NAMES   = {
    5:  "Ombro esquerdo",
    6:  "Ombro direito",
    7:  "Cotovelo esquerdo",
    8:  "Cotovelo direito",
    9:  "Pulso esquerdo",
    10: "Pulso direito",
    11: "Quadril esquerdo",
    12: "Quadril direito",
}

def print_vis(video_filename):
    """
    Procura o arquivo _vis.csv.
    Se encontrado, lê a linha de visibilidades médias e imprime por keypoint.
    Exemplo de uso:
        print_visibilities_for_video("01-certo")
    """
    # Remover extensão, caso o usuário a tenha passado
    base_name = os.path.splitext(video_filename)[0]

    for cls in ["correto", "incorreto"]:
        vis_csv = os.path.join(path, cls, f"{base_name}_vis.csv")
        if os.path.exists(vis_csv):
            df = pd.read_csv(vis_csv)
            print(f"\nVisibilidades médias para '{base_name}' (classe: {cls}):")
            row = df.iloc[0]
            for idx in KP_IDX:
                col_name = f"kp{idx}_v"
                print(f"  {KP_NAMES[idx]} (kp{idx}): {row[col_name]:.4f}")
            break


if __name__ == "__main__":
    # Para imprimir as visibilidades do vídeo 01-certo.mkv:
    print_vis("01-certo")
    print_vis("03-certo")
    print_vis("05-certo")
    print_vis("08-certo")
    print_vis("09-certo")
    print_vis("11-certo")
    print_vis("13-certo")
    print_vis("14-certo")
    print_vis("15-certo")
    print_vis("17-certo")
    print_vis("21-certo")
    print_vis("23-certo")
    # Ou, para '25-errado.mkv':
    print_vis("25-errado")
    print_vis("27-errado")
    print_vis("29-errado")
    print_vis("31-errado")
