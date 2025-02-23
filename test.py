import argparse
import os
import torch
import pandas as pd
from modules.models.Transformer.ChordNet import ChordNet
from modules.utils.device import get_device

def load_model(checkpoint_path, device):
    model = ChordNet(n_freq=12, n_classes=274, n_group=3,
                     f_layer=2, f_head=4,
                     t_layer=2, t_head=4,
                     d_layer=2, d_head=4,
                     dropout=0.2)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model

def run_inference(model, input_tensor):
    with torch.no_grad():
        predictions, _ = model(input_tensor, inference=True)
    return predictions

def load_chroma_from_csv(csv_path, device):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    df = pd.read_csv(csv_path)
    chroma_columns = df.columns[2:]
    # Loading chroma process: extracting full time series from CSV
    chroma_array = df[chroma_columns].astype(float).values  # shape: [T, 12]
    seq_len = 4
    sequences = []
    for i in range(len(chroma_array) - seq_len + 1):
        seq = chroma_array[i:i+seq_len]
        seq_tensor = torch.tensor(seq, dtype=torch.float32, device=device)
        sequences.append(seq_tensor)
    if len(sequences) == 0:
        seq_tensor = torch.tensor(chroma_array[0], dtype=torch.float32, device=device).unsqueeze(0).repeat(seq_len,1)
        sequences.append(seq_tensor)
    batch = torch.stack(sequences, dim=0)  # [B, seq_len, 12]
    batch = batch.unsqueeze(1).repeat(1, 2, 1, 1)  # [B, 2, seq_len, 12]
    return batch

def main():
    parser = argparse.ArgumentParser(description="Predict chord classes using ChordNet")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("csv_file", type=str, help="Path to CSV file with chroma features")
    parser.add_argument("--output", type=str, help="Output CSV file for predictions")
    args = parser.parse_args()
    device = get_device()
    model = load_model(args.checkpoint, device)
    input_tensor = load_chroma_from_csv(args.csv_file, device)
    predictions = run_inference(model, input_tensor)
    if args.output:
        if os.path.isdir(args.output):
            output_path = os.path.join(args.output, "predictions.csv")
        else:
            output_path = args.output
        predictions_np = predictions.cpu().numpy()
        df = pd.DataFrame(predictions_np)
        df.to_csv(output_path, index=False)
        print("Predictions saved to:", output_path)
    else:
        print("Predictions:", predictions)

if __name__ == "__main__":
    main()