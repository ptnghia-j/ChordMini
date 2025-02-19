import os
import argparse
import logging
import csv
from typing import Optional

import torch
import torchaudio
import matplotlib.pyplot as plt

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
            # Obtain only the file name component for CSV output.
            file_name = os.path.basename(args.audio_path)
            save_labels_to_csv(file_name, chroma, args.sample_rate, args.hop_length, args.output_csv)
        except Exception as e:
            logging.error(f"Failed to save labels to CSV: {e}")

    try:
        processor.visualize_chroma(chroma, output_path=args.output_image)
    except Exception as e:
        logging.error(f"Failed to visualize chroma features: {e}")


if __name__ == "__main__":
    main()