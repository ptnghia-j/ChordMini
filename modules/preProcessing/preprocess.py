import os
import argparse
import logging
import csv
from typing import Optional

import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import nnls
import zipfile
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def get_device() -> torch.device:
    """
    Returns the appropriate torch.device by considering CUDA, MPS (for Apple Silicon), or CPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("Using CUDA device.")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using MPS device (Apple Silicon).")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU device.")
    return device


class AudioProcessor:
    def __init__(
        self,
        file_path: str,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 128,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initialize the audio processor with processing parameters.
        
        Args:
            file_path (str): Path to the audio file.
            sample_rate (int): Desired sample rate for audio.
            n_fft (int): FFT window size.
            hop_length (int): Hop length for STFT.
            n_mels (int): Number of mel bins.
            device (torch.device, optional): Device to run processing on.
                Defaults to auto-detection (CUDA, MPS, or CPU).
        """
        self.file_path = file_path
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.device = device if device is not None else get_device()

    def load_audio(self) -> torch.Tensor:
        """
        Load audio from file, handle multi-channel (using first channel) and resample if necessary.

        Returns:
            torch.Tensor: Audio waveform of shape [1, time] on the specified device.
        
        Raises:
            FileNotFoundError: If the audio file does not exist.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Audio file not found: {self.file_path}")
        
        waveform, sr = torchaudio.load(self.file_path)
        logging.info(f"Loaded audio with sample rate {sr} and shape {waveform.shape}")

        # Use only the first channel if multiple channels exist.
        if waveform.size(0) > 1:
            waveform = waveform[0].unsqueeze(0)
            logging.info("Multiple channels detected. Using the first channel only.")

        # Resample if source sample rate doesn't match desired sample rate
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)
            logging.info(f"Resampled audio from {sr} Hz to {self.sample_rate} Hz.")

        return waveform.to(self.device)

    def compute_stft(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Compute the Short-Time Fourier Transform (STFT) of the waveform.

        Args:
            waveform (torch.Tensor): Input audio waveform of shape [1, time].

        Returns:
            torch.Tensor: Complex STFT with shape [freq_bins, time] for the first channel.
        
        Raises:
            Exception: If STFT computation fails.
        """
        try:
            stft = torch.stft(
                waveform,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                return_complex=True
            )
            # Squeeze channel dimension if present
            stft = stft.squeeze(0)
            logging.info(f"Computed STFT with shape {stft.shape}")
            return stft
        except Exception as e:
            logging.error(f"Error computing STFT: {e}")
            raise

    def extract_chroma(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Approximate chroma extraction using a mel spectrogram approximation.
        For MPS devices, temporarily move data to CPU for this transform.
        
        Args:
            waveform (torch.Tensor): Input audio waveform of shape [1, time].

        Returns:
            torch.Tensor: Chroma matrix of shape [12, time_frames] (on CPU).
        
        Raises:
            Exception: If chroma extraction fails.
        """
        try:
            # If on MPS, use CPU for torchaudio transforms since MPS support is incomplete.
            if self.device.type == "mps":
                logging.info("Using CPU for mel spectrogram computation due to MPS limitations.")
                waveform_for_transform = waveform.cpu()
                transform_device = torch.device("cpu")
            else:
                waveform_for_transform = waveform
                transform_device = self.device

            window_fn = lambda win_length: torch.ones(win_length, device=transform_device)
            melspec_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                window_fn=window_fn
            )
            mel_spec = melspec_transform(waveform_for_transform)  # shape: [1, n_mels, time_frames]
            logging.info(f"Mel spectrogram computed with shape {mel_spec.shape}")

            # Convert amplitude to decibels for visualization
            mel_spec_db = torchaudio.transforms.AmplitudeToDB(top_db=80)(mel_spec)
            
            # Approximate chroma by grouping mel bins into 12 pitch classes
            n_chroma = 12
            time_frames = mel_spec_db.shape[2]
            chroma = torch.zeros(n_chroma, time_frames)
            bins_per_chroma = self.n_mels // n_chroma

            # Extract the first (and only) channel of the mel spectrogram
            mel_db = mel_spec_db[0]  # shape: [n_mels, time_frames]
            for i in range(n_chroma):
                start = i * bins_per_chroma
                end = self.n_mels if i == n_chroma - 1 else start + bins_per_chroma
                chroma[i, :] = mel_db[start:end, :].mean(dim=0)
            logging.info(f"Extracted chroma features with shape {chroma.shape}")
            return chroma  # returning on CPU
        except Exception as e:
            logging.error(f"Error extracting chroma: {e}")
            raise

    def visualize_chroma(self, chroma: torch.Tensor, output_path: Optional[str] = None) -> None:
        """
        Visualize chroma features using matplotlib. If output_path is provided,
        the visualization is saved to that file.

        Args:
            chroma (torch.Tensor): Chroma matrix of shape [12, time_frames].
            output_path (str, optional): File path to save the visualization image.
        
        Raises:
            Exception: If visualization fails.
        """
        try:
            plt.figure(figsize=(10, 4))
            plt.imshow(chroma.cpu(), aspect="auto", origin="lower", cmap="viridis")
            plt.title("Chroma Visualization")
            plt.xlabel("Time Frames")
            plt.ylabel("Chroma Bins")
            plt.colorbar(format="%+2.0f dB")
            plt.tight_layout()
            if output_path:
                plt.savefig(output_path)
                logging.info(f"Chroma visualization saved to {output_path}")
            plt.show()
            plt.close()  # Explicitly close the figure to release resources.
        except Exception as e:
            logging.error(f"Error during chroma visualization: {e}")
            raise


def save_labels_to_csv(file_name: str, chroma: torch.Tensor, sample_rate: int, hop_length: int, csv_output: str) -> None:
    """
    Save the processed audio labels (chroma features with timestamps) to a CSV file.
    Each row corresponds to a time frame.
    
    The CSV columns are:
      Filename, Time (seconds), Chroma_A, Chroma_A#, Chroma_B, Chroma_C, Chroma_C#, 
      Chroma_D, Chroma_D#, Chroma_E, Chroma_F, Chroma_F#, Chroma_G, Chroma_G#
    
    Args:
        file_name (str): The audio file name.
        chroma (torch.Tensor): Chroma matrix of shape [12, time_frames].
        sample_rate (int): The sample rate used for processing.
        hop_length (int): Hop length used for the STFT/Mel spectrogram.
        csv_output (str): Path to save the CSV file.
    """
    time_frames = chroma.shape[1]
    header = [
        "Filename", "Time (seconds)", "Chroma_A", "Chroma_A#", "Chroma_B", "Chroma_C",
        "Chroma_C#", "Chroma_D", "Chroma_D#", "Chroma_E", "Chroma_F", "Chroma_F#",
        "Chroma_G", "Chroma_G#"
    ]
    
    with open(csv_output, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        # Loop over each time frame and write chroma values along with corresponding timestamp.
        for i in range(time_frames):
            timestamp = (i * hop_length) / sample_rate
            # Extract chroma values for this column from the tensor and convert to a list of rounded floats.
            chroma_values = [round(float(chroma[chroma_idx, i]), 4) for chroma_idx in range(12)]
            row = [file_name, f"{timestamp:.5f}"] + chroma_values
            writer.writerow(row)
    logging.info(f"Chroma labels saved to CSV: {csv_output}")


def compute_chromagram_nnls(waveform: torch.Tensor, sample_rate: int, window_size: int = 8192, step_size: int = 4410, n_classes: int = 10) -> np.ndarray:
    """
    Compute a chromagram using NNLS approximate transcription.
    
    Args:
        waveform (torch.Tensor): Audio waveform of shape [1, time].
        sample_rate (int): Sample rate.
        window_size (int): FFT window size (default: 8192 samples).
        step_size (int): Hop size (default: 4410 samples; ~10 Hz resolution).
        n_classes (int): Number of classes/chroma bins (default: 10).
    
    Returns:
        np.ndarray: Chromagram with shape [n_classes, num_frames].
    
    Note: For demonstration, random non-negative templates serve as basis functions.
          In a production system, template matrices should be derived from reference data.
    """
    # Convert waveform to a 1D numpy array
    audio = waveform.squeeze().cpu().numpy()
    num_samples = len(audio)
    
    # Segment audio
    frames = []
    for start in range(0, num_samples - window_size + 1, step_size):
        frame = audio[start:start + window_size]
        frames.append(frame)
    frames = np.stack(frames, axis=0)  # shape: [num_frames, window_size]
    
    # Compute magnitude spectrum for each frame using FFT along last axis.
    fft_magnitudes = np.abs(np.fft.rfft(frames, axis=1))  # shape: [num_frames, freq_bins]
    
    num_frames, freq_bins = fft_magnitudes.shape
    # For NNLS, prepare a basis matrix (templates) of shape [n_classes, freq_bins]
    # Here we use random non-negative templates for demonstration.
    np.random.seed(0)
    templates = np.abs(np.random.randn(n_classes, freq_bins))
    
    # For each frame, solve NNLS for activations.
    chromagram = np.zeros((n_classes, num_frames))
    for i in range(num_frames):
        # Solve: templates^T * x ≈ fft_magnitudes[i]
        coeffs, _ = nnls(templates.T, fft_magnitudes[i])
        chromagram[:, i] = coeffs
    return chromagram

def save_chromagram_to_csv(chromagram: np.ndarray, file_name: str, csv_output: str, sample_rate: int, hop_length: int) -> None:
    """
    Save a chromagram (shape: [n_classes, num_frames]) as a CSV file.
    
    The CSV will have one row per time frame with the following columns:
      Filename, Time (seconds), Chroma_A, Chroma_A#, Chroma_B, Chroma_C, Chroma_C#, 
      Chroma_D, Chroma_D#, Chroma_E, Chroma_F, Chroma_F#, Chroma_G, Chroma_G#
      
    Args:
        chromagram (np.ndarray): Chromagram of shape [n_classes, num_frames].
        file_name (str): Audio filename (e.g., "mhwgo.mp3").
        csv_output (str): Path to save the CSV file.
        sample_rate (int): Sample rate used for processing.
        hop_length (int): Hop length used in processing.
    """
    # Transpose chromagram so that rows correspond to time frames
    chroma_t = chromagram.T  # shape: [num_frames, n_classes]
    num_frames, n_classes = chroma_t.shape
    # Header: first column is filename, second is timestamp, then one column per chroma (expecting n_classes==12)
    header = ["Filename", "Time (seconds)"] + [f"Chroma_{i+1}" for i in range(n_classes)]
    
    with open(csv_output, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for i in range(num_frames):
            timestamp = (i * hop_length) / sample_rate
            row = [file_name, f"{timestamp:.5f}"] + [f"{chroma_t[i, j]:.4f}" for j in range(n_classes)]
            writer.writerow(row)
    logging.info(f"Chromagram for {file_name} saved to CSV: {csv_output}")

def compute_nnls_chroma(audio: torch.Tensor, sample_rate: int, n_fft: int = 4096, 
                          hop_length: int = 2048, n_log_bins: int = 256, s_method: str = 'LS') -> np.ndarray:
    """
    Compute NNLS chroma features following the algorithm described:
      1. Resample to 11025 Hz if needed.
      2. Compute the DFT (magnitude spectrum) with given n_fft and hop_length.
      3. Map the magnitude spectrum to a log-frequency spectrogram with n_log_bins using linear interpolation.
      4. Create a note dictionary E of idealised tone profiles over 84 tones (7 octaves) using 
         linearly-spaced s parameters (if s_method == 'LS') otherwise constant s.
      5. For each frame, solve the NNLS problem Y ≈ E x (with x ≥ 0).
      6. Map the 84-tone NNLS transcription to a 12-bin chromagram by summing activations per semitone.
      7. Normalize each chroma vector by dividing by its maximum value.
    
    Returns:
        np.ndarray: Chromagram of shape [12, num_frames].
    """
    # Step 1: Resample audio if needed.
    target_sr = 11025
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
        audio = resampler(audio)
        sample_rate = target_sr

    # Step 2: Compute STFT
    stft = torch.stft(audio, n_fft=n_fft, hop_length=hop_length, return_complex=True)
    magnitude = torch.abs(stft).squeeze(0)  # shape: [freq_bins, num_frames]
    magnitude = magnitude.cpu().numpy()

    # Step 3: Map linear frequency to log-frequency bins.
    num_freq_bins = magnitude.shape[0]
    freqs = np.linspace(0, sample_rate/2, num_freq_bins)
    # Define target log-frequency bins from A0 (27.5 Hz) to ~3322 Hz
    target_log_bins = np.linspace(np.log2(27.5), np.log2(3322), n_log_bins)
    log_freqs = np.log2(freqs + 1e-6)
    Y = np.zeros((n_log_bins, magnitude.shape[1]))
    # For each frame, interpolate the magnitude spectrum to the log-frequency bins.
    for t in range(magnitude.shape[1]):
        Y[:, t] = np.interp(target_log_bins, log_freqs, magnitude[:, t])

    # Step 4: (Optional pre-processing: here we use the original representation.)
    Y_processed = Y

    # Step 5: Create note dictionary E.
    num_tones = 84  # seven octaves, 12 semitones per octave
    E = np.zeros((n_log_bins, num_tones))
    if s_method == 'LS':
        s_values = np.linspace(0.9, 0.6, num_tones)
    else:
        s_values = np.full(num_tones, 0.9)
    # Compute fundamental frequencies for the 84 tones from A0 (27.5 Hz) upward.
    fundamentals = 27.5 * (2 ** (np.arange(num_tones) / 12.0))
    log_fundamentals = np.log2(fundamentals)
    for i in range(num_tones):
        # Here we create a Gaussian-like profile centered at the tone's log-frequency.
        E[:, i] = np.exp(-0.5 * ((target_log_bins - log_fundamentals[i])**2) / (0.1**2))
        E[:, i] /= E[:, i].sum() + 1e-6

    # Step 6: Solve NNLS for each frame to get activation matrix X of shape [84, num_frames].
    num_frames = Y_processed.shape[1]
    X = np.zeros((num_tones, num_frames))
    for t in range(num_frames):
        X[:, t], _ = nnls(E, Y_processed[:, t])
    
    # Step 7: Map the 84-tone transcription to a 12-bin chromagram by summing over octaves.
    chroma = np.zeros((12, num_frames))
    for i in range(num_tones):
        chroma[i % 12, :] += X[i, :]
    
    # Step 8: Normalize each chroma vector.
    chroma = chroma / (np.max(chroma, axis=0, keepdims=True) + 1e-6)
    return chroma

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for audio processing.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Audio to Chroma Vector Processing using PyTorch Audio"
    )
    parser.add_argument("audio_path", type=str, help="Path to the input audio file")
    parser.add_argument("--sample_rate", type=int, default=22050, help="Target sample rate (default: 22050)")
    parser.add_argument("--n_fft", type=int, default=1024, help="FFT window size (default: 1024)")
    parser.add_argument("--hop_length", type=int, default=256, help="Hop length (default: 256)")
    parser.add_argument("--n_mels", type=int, default=128, help="Number of mel bins (default: 128)")
    parser.add_argument("--output_image", type=str, default=None, help="Path to save the chroma visualization image (optional)")
    parser.add_argument("--output_csv", type=str, default=None, help="Path to save the CSV file with audio labels (optional)")
    parser.add_argument("--output_zip", type=str, default="chromagram_features.zip", help="Output zip filename for CSVs")
    return parser.parse_args()


def main() -> None:
    """
    Main execution: load audio, compute STFT, extract chroma, visualize features, and optionally save labels to CSV.
    """
    args = parse_args()

    processor = AudioProcessor(
        file_path=args.audio_path,
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels
    )

    try:
        waveform = processor.load_audio()
    except Exception as e:
        logging.error(f"Failed to load audio: {e}")
        return

    try:
        stft = processor.compute_stft(waveform)
        logging.info(f"STFT computed with shape: {stft.shape}")
    except Exception as e:
        logging.error(f"Failed to compute STFT: {e}")
        return

    try:
        chroma = processor.extract_chroma(waveform)
        logging.info(f"Chroma extracted with shape: {chroma.shape}")
    except Exception as e:
        logging.error(f"Failed to extract chroma features: {e}")
        return

    # Save labels to CSV if requested.
    if args.output_csv:
        try:
            file_name = os.path.basename(args.audio_path)
            save_labels_to_csv(file_name, chroma, args.sample_rate, args.hop_length, args.output_csv)
        except Exception as e:
            logging.error(f"Failed to save labels to CSV: {e}")

    try:
        processor.visualize_chroma(chroma, output_path=args.output_image)
    except Exception as e:
        logging.error(f"Failed to visualize chroma features: {e}")

    # Compute chromagram with NNLS parameters.
    chroma_nnls = compute_chromagram_nnls(waveform, sample_rate=processor.sample_rate, window_size=8192, step_size=4410, n_classes=10)
    logging.info(f"Computed chromagram shape: {chroma_nnls.shape}")
    file_name = os.path.basename(args.audio_path)
    # Instead of zipping, save the chromagram directly as a CSV.
    save_chromagram_to_csv(chroma_nnls, file_name, args.output_zip, processor.sample_rate, processor.hop_length)

    # Compute NNLS-based chroma features using the proposed algorithm.
    nnls_chroma = compute_nnls_chroma(waveform, sample_rate=processor.sample_rate, 
                                      n_fft=4096, hop_length=2048, n_log_bins=256, s_method='LS')
    logging.info(f"Computed NNLS chroma shape: {nnls_chroma.shape}")
    file_name = os.path.basename(args.audio_path)
    # Save the NNLS chroma directly as a CSV (using our CSV-saving function).
    save_chromagram_to_csv(nnls_chroma, file_name, args.output_zip, processor.sample_rate, 2048)
    logging.info("NNLS chroma processing complete.")

if __name__ == "__main__":
    main()