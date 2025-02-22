import argparse
import torch
import pandas as pd
import os  # added: import os
from modules.models.Transformer.ChordNet import ChordNet
from modules.utils.device import get_device

def load_model(checkpoint_path, device):
    model = ChordNet(n_freq=12, n_group=4, f_head=1, t_head=1, d_head=1)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def run_inference(model, input_tensor):
    with torch.no_grad():
        predictions, _ = model(input_tensor, inference=True)
    return predictions

def load_chroma_from_csv(csv_path, device):
    if not os.path.exists(csv_path):  # added: check if file exists
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    # Assumes CSV columns: Filename, Time (seconds), Chroma_A, Chroma_A#, ... Chroma_G#
    df = pd.read_csv(csv_path)
    chroma_columns = df.columns[2:] # adjust if necessary
    inputs = []
    for _, row in df.iterrows():
        # Convert the chroma values to a list of floats
        chroma_data = row[chroma_columns].astype(float).tolist()
        # Create tensor of shape [1, 12]
        chroma_tensor = torch.tensor(chroma_data, dtype=torch.float32, device=device).unsqueeze(0)
        # Expand dims: duplicate channel and add time dimension to get shape [2, 1, 12]
        chroma_tensor = chroma_tensor.unsqueeze(0).repeat(2, 1, 1)
        inputs.append(chroma_tensor)
    # Stack to form batch: [batch_size, 2, 1, 12]
    return torch.stack(inputs, dim=0)

def main():
    parser = argparse.ArgumentParser(description="Predict chord classes for every instance in a CSV file using ChordNet")
    parser.add_argument("checkpoint", type=str, help="Path to the model checkpoint (e.g. checkpoints/model_epoch_5.pth)")
    parser.add_argument("csv_file", type=str, help="Path to the CSV file containing chroma features (e.g. mhwgo_nnls_chromagram.csv)")
    parser.add_argument("--output", type=str, help="Path to the output CSV file for predictions")  # added: optional output argument
    args = parser.parse_args()
    device = get_device()  # added: get device
    model = load_model(args.checkpoint, device)  # added: load model
    input_tensor = load_chroma_from_csv(args.csv_file, device)  # added: load chroma input
    predictions = run_inference(model, input_tensor)  # added: run inference
    
    if args.output:
        import os  # Ensure os is imported
        # If output path is a directory, append default filename
        if os.path.isdir(args.output):
            output_path = os.path.join(args.output, "predictions.csv")
        else:
            output_path = args.output
        predictions_np = predictions.cpu().numpy()  # convert predictions to numpy array
        df = pd.DataFrame(predictions_np)  # create DataFrame from predictions
        df.to_csv(output_path, index=False)  # write DataFrame to CSV
        print("Predictions saved to:", output_path)  # output saved message
    else:
        print("Predictions:", predictions)  # added: output predictions

if __name__ == "__main__":
    main()