import os
import numpy as np
import pandas as pd

INPUT_ROOT = "yolokp"  
OUTPUT_ROOT = "yolokp_centered"

def process_all_csvs(input_root: str, output_root: str):
    """
    Percorre recursivamente `input_root`, procura todos os arquivos *.csv,
    apenas subtrai a média
    e salva a versão “centralizada” em output_root
    """
    os.makedirs(output_root, exist_ok=True)

    for dirpath, dirnames,filenames in os.walk(input_root):
        rel_path = os.path.relpath(dirpath, input_root)
        # Garante que o mesmo subdiretório exista em output_root
        target_dir = os.path.join(output_root, rel_path)
        os.makedirs(target_dir, exist_ok=True)

        for fname in filenames:

            src_path = os.path.join(dirpath, fname)
            dst_path = os.path.join(target_dir, fname)
            # Carrega o CSV original 
            df = pd.read_csv(src_path)
            seq = df.values.astype(np.float32)  # shape = (seq_len, n_feats)
            # subtrai a média de cada coluna (feature)
            mean_cols = seq.mean(axis=0, keepdims=True)  # shape = (1, n_feats)
            seq_processed = seq - mean_cols               # broadcast
            # Transforma de volta em DataFrame (mantendo os mesmos nomes de coluna)
            df_processed = pd.DataFrame(seq_processed, columns=df.columns)
            # Salva o novo CSV “centrado” em output_root/rel_path/fname
            df_processed.to_csv(dst_path, index=False)
            print(f"Processado: {src_path} → {dst_path}")
    print(f"Todos os arquivos foram processados. Novos CSVs em: {output_root}")

if __name__ == "__main__":

    process_all_csvs(INPUT_ROOT, OUTPUT_ROOT)
