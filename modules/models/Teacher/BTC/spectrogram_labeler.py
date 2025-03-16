import os
import argparse
import numpy as np
import torch
import librosa
from utils.hparams import HParams
from btc_model import BTC_model
from utils.mir_eval_modules import audio_file_to_features, idx2chord, idx2voca_chord, get_audio_paths
from utils import logger

def main():
    parser = argparse.ArgumentParser(description="Process audio from data/fma_small into labels and spectrograms.")
    parser.add_argument('--voca', default=True, type=lambda x: (str(x).lower()=='true'))
    parser.add_argument('--audio_dir', type=str, default='./data/fma_small')
    parser.add_argument('--save_dir', type=str, default='./data/synth')
    args = parser.parse_args()
    
    logger.logging_verbosity(1)
    config = HParams.load("run_config.yaml")
    if args.voca:
        config.feature['large_voca'] = True
        config.model['num_chords'] = 170
        model_file = './test/btc_model_large_voca.pt'
        idx_to_chord = idx2voca_chord()
        logger.info("label type: large voca")
    else:
        config.model['num_chords'] = 25
        model_file = './test/btc_model.pt'
        idx_to_chord = idx2chord
        logger.info("label type: Major and minor")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BTC_model(config=config.model).to(device)
    if os.path.isfile(model_file):
        checkpoint = torch.load(model_file, map_location=device)
        mean = checkpoint['mean']
        std = checkpoint['std']
        model.load_state_dict(checkpoint['model'])
        logger.info("restore model")
    else:
        mean = 0.0
        std = 1.0
    
    model.eval()
    
    # Get project root directory (up 4 levels from the current script)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../../.."))
    
    # Create base save directories for spectrograms and labels in the root directory
    base_spec_save_dir = os.path.join(project_root, args.save_dir, "spectrograms")
    base_lab_save_dir = os.path.join(project_root, args.save_dir, "labels")
    os.makedirs(base_spec_save_dir, exist_ok=True)
    os.makedirs(base_lab_save_dir, exist_ok=True)
    
    # Get all audio files
    audio_paths = get_audio_paths(args.audio_dir)
    logger.info(f"Found {len(audio_paths)} audio files.")
    
    # Create sub-folders for spectrograms and labels based on audio file IDs
    for i in range(1000):  # Create folders for 000-999
        folder_name = f"{i:03d}"
        os.makedirs(os.path.join(base_spec_save_dir, folder_name), exist_ok=True)
        os.makedirs(os.path.join(base_lab_save_dir, folder_name), exist_ok=True)
    
    # Process each audio file
    processed_count = 0
    for audio_path in audio_paths:
        logger.info(f"Processing: {audio_path}")
        try:
            # Extract feature, time per hop, song length
            feature, feature_per_second, song_length_second = audio_file_to_features(audio_path, config)
        except Exception as e:
            logger.info(f"Skipping file {audio_path}: {e}")
            continue
        
        # Determine subfolder and filename
        file_id = os.path.basename(audio_path).split('.')[0]
        try:
            folder_name = f"{int(file_id)//1000:03d}"  # Group by thousands (000, 001, etc.)
        except ValueError:
            # If file_id can't be converted to int, use a default folder
            folder_name = "000"
        
        spec_save_dir = os.path.join(base_spec_save_dir, folder_name)
        lab_save_dir = os.path.join(base_lab_save_dir, folder_name)
        
        # Save spectrogram (transposed) as a numpy file
        spec = feature.T  # CQT spectrogram with shape [time, frequency]
        base = os.path.splitext(os.path.basename(audio_path))[0]
        npy_save_path = os.path.join(spec_save_dir, base + "_spec.npy")
        np.save(npy_save_path, spec)
        
        logger.info(f"Saved spectrogram to: {npy_save_path}")
        logger.info(f"Spectrogram shape: {spec.shape}")
        
        # Normalize and pad the feature for model inference
        spec_norm = (spec - mean) / std
        n_timestep = config.model['timestep']
        num_pad = n_timestep - (spec_norm.shape[0] % n_timestep)
        spec_norm = np.pad(spec_norm, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
        num_instance = spec_norm.shape[0] // n_timestep
        
        time_unit = feature_per_second
        start_time = 0.0
        lines = []
        spec_tensor = torch.tensor(spec_norm, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            model.eval()
            for t in range(num_instance):
                chunk = spec_tensor[:, n_timestep*t:n_timestep*(t+1), :]
                self_attn_out, _ = model.self_attn_layers(chunk)
                pred, _ = model.output_layer(self_attn_out)
                pred = pred.squeeze()
                for i in range(n_timestep):
                    if t==0 and i==0:
                        prev_chord = pred[i].item()
                        continue
                    if pred[i].item() != prev_chord:
                        lines.append('%.3f %.3f %s\n' % (start_time, time_unit*(n_timestep*t+i), idx_to_chord[prev_chord]))
                        start_time = time_unit*(n_timestep*t+i)
                        prev_chord = pred[i].item()
                    if t==num_instance-1 and i+num_pad==n_timestep:
                        if start_time != time_unit*(n_timestep*t+i):
                            lines.append('%.3f %.3f %s\n' % (start_time, time_unit*(n_timestep*t+i), idx_to_chord[prev_chord]))
                        break
        
        lab_save_path = os.path.join(lab_save_dir, base + ".lab")
        with open(lab_save_path, "w") as f:
            f.writelines(lines)
        logger.info(f"Saved label file to: {lab_save_path}")
        
        processed_count += 1
    
    logger.info(f"Processing complete. Processed {processed_count} files.")
    
if __name__ == "__main__":
    main()
