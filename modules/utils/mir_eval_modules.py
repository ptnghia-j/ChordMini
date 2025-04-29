import soundfile as sf
if not hasattr(sf, 'SoundFileRuntimeError'):
    sf.SoundFileRuntimeError = RuntimeError

import numpy as np
import librosa
import mir_eval
import torch
import os
import audioread
import torch
from tqdm import tqdm
from modules.utils.chords import idx2voca_chord

def audio_file_to_features(audio_file, config):
    import os
    if not os.path.isfile(audio_file):
        raise FileNotFoundError(f"Audio file not found: {audio_file}")
    # Debug info: file exists, log its size
    file_size = os.path.getsize(audio_file)
    # print("DEBUG: Audio file found:", audio_file, "Size:", file_size, "bytes")
    # If file is unusually small, it might be corrupt – add debug and skip
    if file_size < 5000:
        # print("DEBUG: File size is unusually small, possibly corrupt:", audio_file)
        raise RuntimeError(f"Audio file '{audio_file}' is too small and may be corrupt.")
    try:
        original_wav, sr = librosa.load(audio_file, sr=config.mp3['song_hz'], mono=True)
        # print("DEBUG: Successfully loaded audio file:", audio_file)
        # print("DEBUG: Sample rate:", sr, "Signal length:", len(original_wav))
    except Exception as e:
        # print("DEBUG: Exception occurred while loading audio file:", audio_file)
        # print("DEBUG: Exception details:", e)
        # print("DEBUG: Available audioread backends:", audioread.available_backends())
        raise RuntimeError(f"Failed to load audio file '{audio_file}': {e}")

    # Get FFT size from config or use default
    n_fft = config.feature.get('n_fft', 512)
    hop_length = config.feature.get('hop_length', 512)

    currunt_sec_hz = 0
    feature = None  # initialize feature
    while len(original_wav) > currunt_sec_hz + config.mp3['song_hz'] * config.mp3['inst_len']:
        start_idx = int(currunt_sec_hz)
        end_idx = int(currunt_sec_hz + config.mp3['song_hz'] * config.mp3['inst_len'])
        tmp = librosa.cqt(original_wav[start_idx:end_idx],
                          sr=sr,
                          n_bins=config.feature['n_bins'],
                          bins_per_octave=config.feature['bins_per_octave'],
                          hop_length=config.feature['hop_length'])
        if feature is None:
            feature = tmp
        else:
            feature = np.concatenate((feature, tmp), axis=1)
        currunt_sec_hz = end_idx

    # Process the final segment with proper padding if needed
    final_segment = original_wav[currunt_sec_hz:]

    # Check if segment is too short for the FFT window
    if len(final_segment) < n_fft:
        # Print warning and pad the segment
        print(f"Warning: Final segment of {audio_file} is too short ({len(final_segment)} samples). Padding to {n_fft} samples.")
        padding_needed = n_fft - len(final_segment)
        final_segment = np.pad(final_segment, (0, padding_needed), mode="constant", constant_values=0)

    # Process the properly sized segment
    tmp = librosa.cqt(final_segment,
                      sr=sr,
                      n_bins=config.feature['n_bins'],
                      bins_per_octave=config.feature['bins_per_octave'],
                      hop_length=config.feature['hop_length'])

    if feature is None:
        feature = tmp
    else:
        feature = np.concatenate((feature, tmp), axis=1)

    feature = np.log(np.abs(feature) + 1e-6)
    song_length_second = len(original_wav) / config.mp3['song_hz']

    # Calculate frames per second and frame duration
    # frame duration in seconds per frame
    frame_duration = config.feature['hop_length'] / config.mp3['song_hz']

    # Add diagnostic info for debugging
    # print(f"Audio file: {os.path.basename(audio_file)}")
    print(f"Frames: {feature.shape[1]}, Frame duration: {frame_duration:.5f}s, Total duration: {song_length_second:.2f}s")

    # Return frame-level CQT, frame_duration (s), and song length (s)
    return feature, frame_duration, song_length_second

# Audio files with format of wav and mp3
def get_audio_paths(audio_dir):
    return [os.path.join(root, fname) for (root, dir_names, file_names) in os.walk(audio_dir, followlinks=True)
            for fname in file_names if (fname.lower().endswith('.wav') or fname.lower().endswith('.mp3'))]

class metrics():
    def __init__(self):
        super(metrics, self).__init__()
        self.score_metrics = ['root', 'thirds', 'triads', 'sevenths', 'tetrads', 'majmin', 'mirex']
        self.score_list_dict = dict()
        for i in self.score_metrics:
            self.score_list_dict[i] = list()
        self.average_score = dict()

    def score(self, metric, gt_path, est_path):
        if metric == 'root':
            score = self.root_score(gt_path,est_path)
        elif metric == 'thirds':
            score = self.thirds_score(gt_path,est_path)
        elif metric == 'triads':
            score = self.triads_score(gt_path,est_path)
        elif metric == 'sevenths':
            score = self.sevenths_score(gt_path,est_path)
        elif metric == 'tetrads':
            score = self.tetrads_score(gt_path,est_path)
        elif metric == 'majmin':
            score = self.majmin_score(gt_path,est_path)
        elif metric == 'mirex':
            score = self.mirex_score(gt_path,est_path)
        else:
            raise NotImplementedError
        return score

    def root_score(self, gt_path, est_path):
        (ref_intervals, ref_labels) = mir_eval.io.load_labeled_intervals(gt_path)
        ref_labels = lab_file_error_modify(ref_labels)
        (est_intervals, est_labels) = mir_eval.io.load_labeled_intervals(est_path)
        est_intervals, est_labels = mir_eval.util.adjust_intervals(est_intervals, est_labels, ref_intervals.min(),
                                                                   ref_intervals.max(), mir_eval.chord.NO_CHORD,
                                                                   mir_eval.chord.NO_CHORD)
        (intervals, ref_labels, est_labels) = mir_eval.util.merge_labeled_intervals(ref_intervals, ref_labels,
                                                                                    est_intervals, est_labels)
        durations = mir_eval.util.intervals_to_durations(intervals)
        comparisons = mir_eval.chord.root(ref_labels, est_labels)
        score = mir_eval.chord.weighted_accuracy(comparisons, durations)
        return score

    def thirds_score(self, gt_path, est_path):
        (ref_intervals, ref_labels) = mir_eval.io.load_labeled_intervals(gt_path)
        ref_labels = lab_file_error_modify(ref_labels)
        (est_intervals, est_labels) = mir_eval.io.load_labeled_intervals(est_path)
        est_intervals, est_labels = mir_eval.util.adjust_intervals(est_intervals, est_labels, ref_intervals.min(),
                                                                   ref_intervals.max(), mir_eval.chord.NO_CHORD,
                                                                   mir_eval.chord.NO_CHORD)
        (intervals, ref_labels, est_labels) = mir_eval.util.merge_labeled_intervals(ref_intervals, ref_labels,
                                                                                    est_intervals, est_labels)
        durations = mir_eval.util.intervals_to_durations(intervals)
        comparisons = mir_eval.chord.thirds(ref_labels, est_labels)
        score = mir_eval.chord.weighted_accuracy(comparisons, durations)
        return score

    def triads_score(self, gt_path, est_path):
        (ref_intervals, ref_labels) = mir_eval.io.load_labeled_intervals(gt_path)
        ref_labels = lab_file_error_modify(ref_labels)
        (est_intervals, est_labels) = mir_eval.io.load_labeled_intervals(est_path)
        est_intervals, est_labels = mir_eval.util.adjust_intervals(est_intervals, est_labels, ref_intervals.min(),
                                                                   ref_intervals.max(), mir_eval.chord.NO_CHORD,
                                                                   mir_eval.chord.NO_CHORD)
        (intervals, ref_labels, est_labels) = mir_eval.util.merge_labeled_intervals(ref_intervals, ref_labels,
                                                                                    est_intervals, est_labels)
        durations = mir_eval.util.intervals_to_durations(intervals)
        comparisons = mir_eval.chord.triads(ref_labels, est_labels)
        score = mir_eval.chord.weighted_accuracy(comparisons, durations)
        return score

    def sevenths_score(self, gt_path, est_path):
        (ref_intervals, ref_labels) = mir_eval.io.load_labeled_intervals(gt_path)
        ref_labels = lab_file_error_modify(ref_labels)
        (est_intervals, est_labels) = mir_eval.io.load_labeled_intervals(est_path)
        est_intervals, est_labels = mir_eval.util.adjust_intervals(est_intervals, est_labels, ref_intervals.min(),
                                                                   ref_intervals.max(), mir_eval.chord.NO_CHORD,
                                                                   mir_eval.chord.NO_CHORD)
        (intervals, ref_labels, est_labels) = mir_eval.util.merge_labeled_intervals(ref_intervals, ref_labels,
                                                                                    est_intervals, est_labels)
        durations = mir_eval.util.intervals_to_durations(intervals)
        comparisons = mir_eval.chord.sevenths(ref_labels, est_labels)
        score = mir_eval.chord.weighted_accuracy(comparisons, durations)
        return score

    def tetrads_score(self, gt_path, est_path):
        (ref_intervals, ref_labels) = mir_eval.io.load_labeled_intervals(gt_path)
        ref_labels = lab_file_error_modify(ref_labels)
        (est_intervals, est_labels) = mir_eval.io.load_labeled_intervals(est_path)
        est_intervals, est_labels = mir_eval.util.adjust_intervals(est_intervals, est_labels, ref_intervals.min(),
                                                                   ref_intervals.max(), mir_eval.chord.NO_CHORD,
                                                                   mir_eval.chord.NO_CHORD)
        (intervals, ref_labels, est_labels) = mir_eval.util.merge_labeled_intervals(ref_intervals, ref_labels,
                                                                                    est_intervals, est_labels)
        durations = mir_eval.util.intervals_to_durations(intervals)
        comparisons = mir_eval.chord.tetrads(ref_labels, est_labels)
        score = mir_eval.chord.weighted_accuracy(comparisons, durations)
        return score

    def majmin_score(self, gt_path, est_path):
        (ref_intervals, ref_labels) = mir_eval.io.load_labeled_intervals(gt_path)
        ref_labels = lab_file_error_modify(ref_labels)
        (est_intervals, est_labels) = mir_eval.io.load_labeled_intervals(est_path)
        est_intervals, est_labels = mir_eval.util.adjust_intervals(est_intervals, est_labels, ref_intervals.min(),
                                                                   ref_intervals.max(), mir_eval.chord.NO_CHORD,
                                                                   mir_eval.chord.NO_CHORD)
        (intervals, ref_labels, est_labels) = mir_eval.util.merge_labeled_intervals(ref_intervals, ref_labels,
                                                                                    est_intervals, est_labels)
        durations = mir_eval.util.intervals_to_durations(intervals)
        comparisons = mir_eval.chord.majmin(ref_labels, est_labels)
        score = mir_eval.chord.weighted_accuracy(comparisons, durations)
        return score

    def mirex_score(self, gt_path, est_path):
        (ref_intervals, ref_labels) = mir_eval.io.load_labeled_intervals(gt_path)
        ref_labels = lab_file_error_modify(ref_labels)
        (est_intervals, est_labels) = mir_eval.io.load_labeled_intervals(est_path)
        est_intervals, est_labels = mir_eval.util.adjust_intervals(est_intervals, est_labels, ref_intervals.min(),
                                                                   ref_intervals.max(), mir_eval.chord.NO_CHORD,
                                                                   mir_eval.chord.NO_CHORD)
        (intervals, ref_labels, est_labels) = mir_eval.util.merge_labeled_intervals(ref_intervals, ref_labels,
                                                                                    est_intervals, est_labels)
        durations = mir_eval.util.intervals_to_durations(intervals)
        comparisons = mir_eval.chord.mirex(ref_labels, est_labels)
        score = mir_eval.chord.weighted_accuracy(comparisons, durations)
        return score

def standardize_chord_label(chord_label):
    """
    Standardize a single chord label to ensure compatibility with mir_eval.

    Args:
        chord_label: A single chord label string to standardize

    Returns:
        Standardized chord label string
    """
    # Skip empty or None labels
    if not chord_label:
        return "N"  # Map empty to NO_CHORD

    # Handle special cases but preserve X vs N distinction
    if chord_label in ["N", "None", "NC"]:
        return "N"  # NO_CHORD

    if chord_label in ["X", "Unknown"]:
        return "X"  # UNKNOWN_CHORD - preserve distinction from NO_CHORD

    # Fix common format issues
    if chord_label.endswith(':4'):
        chord_label = chord_label.replace(':4', ':sus4')
    elif chord_label.endswith(':6'):
        chord_label = chord_label.replace(':6', ':maj6')
    elif chord_label.endswith(':6/2'):
        chord_label = chord_label.replace(':6/2', ':maj6/2')
    elif chord_label == 'Emin/4':
        chord_label = 'E:min/4'
    elif chord_label == 'A7/3':
        chord_label = 'A:7/3'
    elif chord_label == 'Bb7/3':
        chord_label = 'Bb:7/3'
    elif chord_label == 'Bb7/5':
        chord_label = 'Bb:7/5'

    # Handle root-only chords (e.g., "C", "G#")
    root_notes = ["A", "B", "C", "D", "E", "F", "G"]
    if chord_label in root_notes or (len(chord_label) == 2 and chord_label[0] in root_notes and chord_label[1] in ['#', 'b']):
        chord_label = chord_label + ":maj"

    # Handle chords without colon separator
    elif ':' not in chord_label:
        # Handle minor chords
        if 'min' in chord_label:
            idx = chord_label.find('min')
            chord_label = chord_label[:idx] + ':' + chord_label[idx:]
        # Handle dominant 7th chords
        elif '7' in chord_label and 'maj7' not in chord_label and 'min7' not in chord_label:
            idx = chord_label.find('7')
            chord_label = chord_label[:idx] + ':7' + chord_label[idx+1:]
        # Handle major 7th chords
        elif 'maj7' in chord_label:
            idx = chord_label.find('maj7')
            chord_label = chord_label[:idx] + ':maj7' + chord_label[idx+4:]
        # Handle minor 7th chords
        elif 'min7' in chord_label:
            idx = chord_label.find('min7')
            chord_label = chord_label[:idx] + ':min7' + chord_label[idx+4:]
        # Handle diminished chords
        elif 'dim' in chord_label:
            idx = chord_label.find('dim')
            chord_label = chord_label[:idx] + ':dim' + chord_label[idx+3:]
        # Handle augmented chords
        elif 'aug' in chord_label:
            idx = chord_label.find('aug')
            chord_label = chord_label[:idx] + ':aug' + chord_label[idx+3:]
        # Handle sus chords
        elif 'sus' in chord_label:
            idx = chord_label.find('sus')
            chord_label = chord_label[:idx] + ':sus' + chord_label[idx+3:]
        # Default to major for other root-only chords
        elif any(chord_label.startswith(note) for note in root_notes):
            # Check if there's a flat or sharp
            if len(chord_label) > 1 and chord_label[1] in ['b', '#']:
                if len(chord_label) == 2:  # Just a root note with accidental
                    chord_label = chord_label + ":maj"
            else:
                if len(chord_label) == 1:  # Just a root note
                    chord_label = chord_label + ":maj"

    return chord_label

def lab_file_error_modify(ref_labels):
    """
    Standardize chord labels to ensure compatibility with mir_eval.

    Args:
        ref_labels: List of chord labels to standardize or a single chord label

    Returns:
        List of standardized chord labels or a single standardized chord label
    """
    # Handle single string case
    if isinstance(ref_labels, str):
        return standardize_chord_label(ref_labels)

    # Handle list case
    standardized_labels = []
    for label in ref_labels:
        standardized_labels.append(standardize_chord_label(label))

    return standardized_labels

def extract_chord_quality(chord):
    """
    Extract chord quality from a chord label, handling different formats.
    Supports both colon format (C:maj) and direct format (Cmaj).

    Args:
        chord: A chord label string

    Returns:
        The chord quality as a string
    """
    # Handle None or empty strings
    if not chord:
        return "N"  # Default to "N" for empty chords

    # Handle special cases
    if chord in ["N", "None", "NC"]:
        return "N"  # No chord
    if chord in ["X", "Unknown"]:
        return "X"  # Unknown chord

    # Handle colon format (e.g., "C:min")
    if ':' in chord:
        parts = chord.split(':')
        if len(parts) > 1:
            # Handle bass notes (e.g., "C:min/G")
            quality = parts[1].split('/')[0] if '/' in parts[1] else parts[1]
            return quality

    # Handle direct format without colon (e.g., "Cmin")
    import re
    root_pattern = r'^[A-G][#b]?'
    match = re.match(root_pattern, chord)
    if match:
        quality = chord[match.end():]
        if quality:
            # Handle bass notes (e.g., "Cmin/G")
            return quality.split('/')[0] if '/' in quality else quality

    # Default to major if we couldn't extract a quality
    return "maj"

def compute_individual_chord_accuracy(reference_labels, prediction_labels, chunk_size=10000):
    """
    Compute accuracy for individual chord qualities.
    Uses a robust approach to extract chord qualities from different formats.
    Maps chord qualities to broader categories to match validation reporting.
    Processes data in chunks to avoid memory issues.

    Args:
        reference_labels: List of reference chord labels
        prediction_labels: List of predicted chord labels
        chunk_size: Number of samples to process in each chunk

    Returns:
        acc: Dictionary mapping chord quality to accuracy
        stats: Dictionary with detailed statistics for each quality
    """
    from collections import defaultdict
    import time

    # Always import the map_chord_to_quality function from visualize.py for consistency
    try:
        from modules.utils.visualize import map_chord_to_quality, extract_chord_quality as visualize_extract_chord_quality
        use_quality_mapping = True
        print("Using quality mapping from visualize.py for consistent reporting")
    except ImportError:
        use_quality_mapping = False
        print("Quality mapping from visualize.py not available, using raw qualities")

    # Use two sets of statistics - one for raw qualities and one for mapped qualities
    raw_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    mapped_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

    # Quality mapping for consistent reporting with validation
    # This should match exactly the mapping in visualize.py
    quality_mapping = {
        # Major family
        "maj": "Major", "": "Major", "M": "Major", "major": "Major",
        # Minor family
        "min": "Minor", "m": "Minor", "minor": "Minor",
        # Dominant seventh family
        "7": "Dom7", "dom7": "Dom7", "dominant": "Dom7",
        # Major seventh family
        "maj7": "Maj7", "M7": "Maj7", "major7": "Maj7",
        # Minor seventh family
        "min7": "Min7", "m7": "Min7", "minor7": "Min7",
        # Diminished family
        "dim": "Dim", "°": "Dim", "o": "Dim", "diminished": "Dim",
        # Diminished seventh family
        "dim7": "Dim7", "°7": "Dim7", "o7": "Dim7", "diminished7": "Dim7",
        # Half-diminished family
        "hdim7": "Half-Dim", "m7b5": "Half-Dim", "ø": "Half-Dim", "half-diminished": "Half-Dim",
        # Augmented family
        "aug": "Aug", "+": "Aug", "augmented": "Aug",
        # Suspended family
        "sus2": "Sus", "sus4": "Sus", "sus": "Sus", "suspended": "Sus",
        # Additional common chord qualities
        "min6": "Min6", "m6": "Min6",
        "maj6": "Maj6", "6": "Maj6",
        "minmaj7": "Min-Maj7", "mmaj7": "Min-Maj7", "min-maj7": "Min-Maj7",
        # Special cases
        "N": "No Chord",
        "X": "No Chord",  # Map X to No Chord for consistency with validation
    }

    # Get total number of samples
    total_samples = min(len(reference_labels), len(prediction_labels))
    print(f"Processing {total_samples} samples for chord quality accuracy calculation")

    # Process data in chunks to avoid memory issues
    total_processed = 0
    malformed_chords = 0
    start_time = time.time()

    for chunk_start in range(0, total_samples, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_samples)
        chunk_size_actual = chunk_end - chunk_start

        # Get chunk of data
        ref_chunk = reference_labels[chunk_start:chunk_end]
        pred_chunk = prediction_labels[chunk_start:chunk_end]

        # Process chunk
        for i, (ref, pred) in enumerate(zip(ref_chunk, pred_chunk)):
            if not ref or not pred:
                malformed_chords += 1
                continue

            try:
                # Standardize chord labels for consistent evaluation
                # This is important to match the validation process
                ref = lab_file_error_modify(ref) if isinstance(ref, str) else ref
                pred = lab_file_error_modify(pred) if isinstance(pred, str) else pred

                # Extract chord qualities using the same method as validation
                if use_quality_mapping:
                    # Use the imported function from visualize.py for consistency
                    q_ref_raw = visualize_extract_chord_quality(ref)
                    q_pred_raw = visualize_extract_chord_quality(pred)

                    # Map to broader categories using the same function as validation
                    q_ref_mapped = map_chord_to_quality(ref)
                    q_pred_mapped = map_chord_to_quality(pred)
                else:
                    # Fallback to local implementation
                    q_ref_raw = extract_chord_quality(ref)
                    q_pred_raw = extract_chord_quality(pred)

                    # Use our local mapping
                    q_ref_mapped = quality_mapping.get(q_ref_raw, "Other")
                    q_pred_mapped = quality_mapping.get(q_pred_raw, "Other")

                # Update raw statistics
                raw_stats[q_ref_raw]['total'] += 1
                if q_ref_raw == q_pred_raw:
                    raw_stats[q_ref_raw]['correct'] += 1

                # Update mapped statistics
                mapped_stats[q_ref_mapped]['total'] += 1
                if q_ref_mapped == q_pred_mapped:
                    mapped_stats[q_ref_mapped]['correct'] += 1

            except Exception as e:
                malformed_chords += 1
                continue

        # Update total processed
        total_processed += chunk_size_actual

        # Print progress every 100,000 samples
        if total_processed % 100000 == 0 or total_processed == total_samples:
            elapsed = time.time() - start_time
            print(f"Processed {total_processed}/{total_samples} samples ({total_processed/total_samples*100:.1f}%) in {elapsed:.1f}s")

    print(f"Processed {total_processed} samples, {malformed_chords} were malformed or caused errors")

    # Calculate accuracy for each quality (both raw and mapped)
    raw_acc = {}
    for quality, vals in raw_stats.items():
        if vals['total'] > 0:
            raw_acc[quality] = vals['correct'] / vals['total']
        else:
            raw_acc[quality] = 0.0

    mapped_acc = {}
    for quality, vals in mapped_stats.items():
        if vals['total'] > 0:
            mapped_acc[quality] = vals['correct'] / vals['total']
        else:
            mapped_acc[quality] = 0.0

    # Print both raw and mapped statistics for comparison
    print("\nRaw Chord Quality Distribution:")
    total_raw = sum(stats['total'] for stats in raw_stats.values())
    for quality, stats in sorted(raw_stats.items(), key=lambda x: x[1]['total'], reverse=True):
        if stats['total'] > 0:
            percentage = (stats['total'] / total_raw) * 100
            print(f"  {quality}: {stats['total']} samples ({percentage:.2f}%)")

    print("\nMapped Chord Quality Distribution (matches validation):")
    total_mapped = sum(stats['total'] for stats in mapped_stats.values())
    for quality, stats in sorted(mapped_stats.items(), key=lambda x: x[1]['total'], reverse=True):
        if stats['total'] > 0:
            percentage = (stats['total'] / total_mapped) * 100
            print(f"  {quality}: {stats['total']} samples ({percentage:.2f}%)")

    print("\nRaw Accuracy by chord quality:")
    for quality, accuracy_val in sorted(raw_acc.items(), key=lambda x: x[1], reverse=True):
        if raw_stats[quality]['total'] >= 10:  # Only show meaningful stats
            print(f"  {quality}: {accuracy_val:.4f}")

    print("\nMapped Accuracy by chord quality (matches validation):")
    for quality, accuracy_val in sorted(mapped_acc.items(), key=lambda x: x[1], reverse=True):
        if mapped_stats[quality]['total'] >= 10:  # Only show meaningful stats
            print(f"  {quality}: {accuracy_val:.4f}")

    # Return the mapped statistics for consistency with validation
    return mapped_acc, mapped_stats

def root_majmin_score_calculation(valid_dataset, config, mean, std, device, model, model_type, verbose=False):
    """
    Calculate root and majmin scores for chord recognition using mir_eval framework.
    Optimized for ChordNet model with frame-level predictions.

    Parameters:
    -----------
    valid_dataset: Dataset
        Dataset with validation samples
    config: HParams
        Configuration parameters
    mean, std: float
        Normalization parameters
    device: torch.device
        Device for computation
    model: nn.Module
        ChordNet model
    model_type: str
        Model type identifier (should be 'ChordNet' in most cases)
    verbose: bool
        Whether to print detailed per-song scores

    Returns:
    --------
    score_list_dict: dict
        Dictionary of score lists for metrics
    song_length_list: list
        List of song lengths
    average_score_dict: dict
        Dictionary of average scores for metrics
    """
    valid_song_names = valid_dataset.song_names
    paths = valid_dataset.preprocessor.get_all_files()

    metrics_ = metrics()
    song_length_list = list()

    for path in paths:
        song_name, lab_file_path, mp3_file_path, _ = path
        if not song_name in valid_song_names:
            continue
        try:
            # Extract features from audio file
            n_timestep = config.model['timestep']
            feature, frame_duration, song_length_second = audio_file_to_features(mp3_file_path, config)
            feature = feature.T
            feature = (feature - mean) / std
            time_unit = frame_duration

            # Pad features to match the timestep size
            num_pad = n_timestep - (feature.shape[0] % n_timestep)
            feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
            num_instance = feature.shape[0] // n_timestep

            start_time = 0.0
            lines = []

            # Generate chord predictions
            with torch.no_grad():
                model.eval()
                feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)

                # Process each segment
                for t in range(num_instance):
                    # Extract frame-level predictions for this segment
                    try:
                        # Check if model is wrapped with DistributedDataParallel
                        if hasattr(model, 'module'):
                            # Access the underlying model
                            base_model = model.module
                            prediction = base_model.predict(
                                feature[:, n_timestep * t:n_timestep * (t + 1), :],
                                per_frame=True
                            ).squeeze()
                        else:
                            # Regular model (not DDP wrapped)
                            prediction = model.predict(
                                feature[:, n_timestep * t:n_timestep * (t + 1), :],
                                per_frame=True
                            ).squeeze()
                    except Exception as e:
                        # Fall back to direct model call if predict method fails
                        print(f"Warning: Error in model.predict: {e}. Falling back to direct model call.")
                        # Get predictions using direct model call
                        outputs = model(feature[:, n_timestep * t:n_timestep * (t + 1), :])

                        # Handle different output formats
                        if isinstance(outputs, tuple):
                            logits = outputs[0]
                        else:
                            logits = outputs

                        # Get predictions from logits
                        if logits.dim() == 3:  # [batch, time, classes]
                            prediction = logits.argmax(dim=2).squeeze()
                        else:
                            prediction = logits.argmax(dim=-1).squeeze()

                    # Ensure prediction is a 1D tensor
                    if prediction.dim() == 0:
                        prediction = prediction.unsqueeze(0)

                    # Process each frame in the segment
                    for i in range(prediction.size(0)):
                        # Skip first frame of first segment
                        if t == 0 and i == 0:
                            prev_chord = prediction[i].item()
                            continue

                        # Only record chord change points
                        if prediction[i].item() != prev_chord:
                            lines.append(
                                '%.6f %.6f %s\n' % (
                                    start_time,
                                    time_unit * (n_timestep * t + i),
                                    idx2voca_chord().get(prev_chord, "Unknown")
                                )
                            )
                            start_time = time_unit * (n_timestep * t + i)
                            prev_chord = prediction[i].item()

                        # Handle the final segment's final frame
                        if t == num_instance - 1 and i + num_pad == n_timestep:
                            if start_time != time_unit * (n_timestep * t + i):
                                lines.append(
                                    '%.6f %.6f %s\n' % (
                                        start_time,
                                        time_unit * (n_timestep * t + i),
                                        idx2voca_chord().get(prev_chord, "Unknown")
                                    )
                                )
                            break

            # Write predictions to a temporary file
            pid = os.getpid()
            tmp_path = f'tmp_{pid}.lab'
            with open(tmp_path, 'w') as f:
                for line in lines:
                    f.write(line)

            # Calculate scores for root and majmin metrics
            root_majmin = ['root', 'majmin']
            for m in root_majmin:
                metrics_.score_list_dict[m].append(
                    metrics_.score(metric=m, gt_path=lab_file_path, est_path=tmp_path)
                )

            song_length_list.append(song_length_second)

            if verbose:
                for m in root_majmin:
                    print('song name %s, %s score : %.4f' % (
                        song_name, m, metrics_.score_list_dict[m][-1])
                    )

        except Exception as e:
            print(f'song name {song_name} lab file error: {str(e)}')

    # Calculate weighted average scores based on song length
    tmp = np.array(song_length_list) / np.sum(song_length_list)
    for m in root_majmin:
        metrics_.average_score[m] = np.sum(np.multiply(metrics_.score_list_dict[m], tmp))

    return metrics_.score_list_dict, song_length_list, metrics_.average_score


def large_voca_score_calculation(valid_dataset, config, model, model_type, mean, std, device=None):
    """
    Calculate MIR evaluation scores using the model on a validation dataset.

    Args:
        valid_dataset: List of samples for evaluation
        config: Configuration object
        model: Model to evaluate
        model_type: Type of the model
        mean: Mean value for normalization
        std: Standard deviation for normalization
        device: Device to run evaluation on (default: None, will use model's device)

    Returns:
        score_list_dict: Dictionary of score lists for each metric
        song_length_list: List of song lengths for weighting
        average_score_dict: Dictionary of average scores for each metric
    """

    print(f"Processing list of {len(valid_dataset)} samples for evaluation")

    # Ensure model is in evaluation mode
    model.eval()

    # Get device from model if not specified
    if device is None:
        device = next(model.parameters()).device

    # Ensure mean and std are on the correct device
    if isinstance(mean, torch.Tensor) and mean.device != device:
        mean = mean.to(device)
    if isinstance(std, torch.Tensor) and std.device != device:
        std = std.to(device)

    # Convert mean/std to tensor if they're not already
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean, device=device)
    if not isinstance(std, torch.Tensor):
        std = torch.tensor(std, device=device)

    # Group samples by song_id to evaluate whole songs
    song_groups = {}
    for sample in valid_dataset:
        if 'song_id' in sample:
            song_id = sample['song_id']
            if song_id not in song_groups:
                song_groups[song_id] = []
            song_groups[song_id].append(sample)

    print(f"Grouped into {len(song_groups)} virtual songs for evaluation")

    # Evaluation metrics
    score_list_dict = {
        'root': [],
        'thirds': [],
        'triads': [],
        'sevenths': [],
        'tetrads': [],
        'majmin': [],
        'mirex': []
    }

    song_length_list = []
    collected_refs = []     # <-- New: collect reference chord labels
    collected_preds = []    # <-- New: collect predicted chord labels
    errors = 0

    # Process each song group
    for i, (song_id, samples) in enumerate(tqdm(song_groups.items(),
                                             desc="Evaluating songs",
                                             total=len(song_groups))):
        try:
            # Extract and sort samples by frame index if available
            if all('frame_idx' in sample for sample in samples):
                samples.sort(key=lambda x: x['frame_idx'])

            # Determine frame_duration from model config or fallback
            if hasattr(config, 'model') and 'frame_duration' in config.model:
                frame_duration = config.model['frame_duration']
            elif hasattr(config, 'feature') and hasattr(config, 'mp3'):
                # fallback to hop_length / sample rate
                hl = config.feature.get('hop_length', None)
                sr = config.mp3.get('song_hz', None)
                frame_duration = hl / sr if hl and sr else config.feature.get('hop_duration', 0.1)
            else:
                frame_duration = 0.09288

            # Build timestamps from sample frame_idx when available
            if samples and isinstance(samples[0], dict) and 'frame_idx' in samples[0]:
                timestamps = np.array([s['frame_idx'] * frame_duration for s in samples])
            else:
                timestamps = np.arange(len(samples)) * frame_duration

            # Pad durations so it matches timestamps length
            durations = np.diff(np.append(timestamps, timestamps[-1] + frame_duration))

            # Extract timestamps
            # frame_duration = config.feature.get('hop_duration', 0.1)
            # timestamps = np.arange(len(samples)) * frame_duration

            # Process samples to get spectrograms
            spectrograms = []
            reference_labels = []

            for sample in samples:
                # Handle different sample formats
                if 'spectro' in sample:
                    # Direct spectrogram data
                    spec = sample['spectro']
                elif 'spec_path' in sample:
                    # Load from file and handle frame indexing
                    spec_data = np.load(sample['spec_path'])
                    if 'frame_idx' in sample and len(spec_data.shape) > 1:
                        frame_idx = sample['frame_idx']
                        if frame_idx < spec_data.shape[0]:
                            spec = spec_data[frame_idx]
                        else:
                            # Use zeros for out-of-range frames
                            spec = np.zeros(spec_data.shape[1:] if len(spec_data.shape) > 1 else (144,))
                    else:
                        spec = spec_data
                else:
                    # Fall back to zeros if no spectrogram available
                    spec = np.zeros((144,))

                # Convert numpy array to tensor if needed
                if isinstance(spec, np.ndarray):
                    spec = torch.from_numpy(spec).float()

                # Add batch dimension if needed
                if spec.dim() == 1:
                    spec = spec.unsqueeze(0)

                # Move tensor to the correct device
                spec = spec.to(device)

                # Apply normalization
                spec = (spec - mean) / std

                spectrograms.append(spec)

                # Extract the reference chord label
                chord_label = sample['chord_label']

                # Add extra checking for chord_label structure
                if isinstance(chord_label, list) and len(chord_label) > 0 and isinstance(chord_label[0], list):
                    print(f"Warning: Detected nested chord_label in sample {i}. First element shape: {len(chord_label[0])} chords")

                reference_labels.append(chord_label)

            # Combine all frames into a single tensor
            if spectrograms:
                # Stack spectrograms along time dimension (dim=0)
                input_tensor = torch.stack(spectrograms, dim=0)

                # Reshape if needed
                if len(input_tensor.shape) == 2:  # [frames, features]
                    input_tensor = input_tensor.unsqueeze(1)  # Add time dimension [frames, 1, features]

                # Move tensor to correct device
                input_tensor = input_tensor.to(device)

                # Get predictions from the model
                with torch.no_grad():
                    try:
                        # Check if model is wrapped with DistributedDataParallel
                        if hasattr(model, 'module'):
                            # Access the underlying model
                            base_model = model.module

                            # Try to use the predict method if it exists on the base model
                            if hasattr(base_model, 'predict'):
                                output = base_model.predict(input_tensor)
                            else:
                                # Fall back to direct model call
                                output = model(input_tensor)
                        else:
                            # Regular model (not DDP wrapped)
                            if hasattr(model, 'predict'):
                                output = model.predict(input_tensor)
                            else:
                                # Fall back to direct model call for models like BTC_model
                                output = model(input_tensor)

                        # Handle different output formats
                        if isinstance(output, tuple):
                            output = output[0]  # Take first element if it's a tuple

                        # For BTC model, output is [batch, time, classes]
                        # We need to get the predicted class indices
                        if output.dim() == 3:
                            output = output.argmax(dim=2)  # Get predicted class indices
                    except Exception as e:
                        print(f"Error getting predictions from model: {e}")
                        raise

                # Move predictions to CPU for further processing
                predictions = output.cpu().numpy()

                # Ensure predictions have same length as reference labels
                min_len = min(len(predictions), len(reference_labels))
                predictions = predictions[:min_len]
                reference_labels = reference_labels[:min_len]
                timestamps = timestamps[:min_len]

                # Prepare for mir_eval format
                idx_to_chord = model.idx_to_chord if hasattr(model, 'idx_to_chord') else None

                # If model doesn't have idx_to_chord, create a proper mapping
                if idx_to_chord is None:
                    # Try to use the idx2voca_chord function from modules.utils.chords as a fallback
                    try:
                        from modules.utils.chords import idx2voca_chord
                        idx_to_chord = idx2voca_chord()
                        # Commented out to reduce log verbosity
                        # print("WARNING: model.idx_to_chord not found. Using idx2voca_chord() as fallback.")

                        # Print a sample of the mapping to verify it's correct (only once)
                        if not hasattr(large_voca_score_calculation, '_chord_mapping_logged'):
                            sample_keys = list(idx_to_chord.keys())[:5]
                            sample_mapping = {k: idx_to_chord[k] for k in sample_keys}
                            print(f"Using standard chord mapping: {sample_mapping}")
                            large_voca_score_calculation._chord_mapping_logged = True
                    except Exception as e:
                        print(f"Error using idx2voca_chord: {e}")

                        # Create a basic mapping with proper chord names
                        # Find the maximum prediction value to determine the size of the mapping
                        max_pred = 0
                        for pred in predictions:
                            if isinstance(pred, np.ndarray):
                                # Get the maximum value from the numpy array
                                pred_val = int(pred.item()) if pred.size == 1 else int(pred[0])
                                max_pred = max(max_pred, pred_val)
                            else:
                                # Already an integer or other comparable type
                                max_pred = max(max_pred, pred)

                        # Create a mapping that follows standard chord notation
                        idx_to_chord = {}

                        # Define root notes (12 semitones)
                        root_notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

                        # Define chord qualities
                        qualities = ['maj', 'min', '7', 'maj7', 'min7', 'dim', 'aug', 'sus4', 'sus2',
                                    'dim7', 'hdim7', 'minmaj7', 'maj6', 'min6']

                        # Create mapping for all root+quality combinations
                        idx = 0
                        for root in root_notes:
                            # Major chords (just the root note means major)
                            idx_to_chord[idx] = f"{root}:maj"
                            idx += 1

                            # Add all other qualities
                            for quality in qualities[1:]:  # Skip 'maj' as we already added it
                                if idx <= max_pred:
                                    idx_to_chord[idx] = f"{root}:{quality}"
                                    idx += 1

                        # Special handling for N and X chords
                        # These are typically at the end of the mapping
                        if max_pred >= 168:
                            idx_to_chord[168] = "X"  # Unknown chord
                        if max_pred >= 169:
                            idx_to_chord[169] = "N"  # No chord

                        # Commented out to reduce log verbosity
                        # print("WARNING: model.idx_to_chord not found. Created standard chord mapping.")

                        # Print a sample of the mapping to verify it's correct (only once)
                        if not hasattr(large_voca_score_calculation, '_chord_mapping_logged'):
                            sample_keys = sorted(list(idx_to_chord.keys()))[:5]
                            sample_mapping = {k: idx_to_chord[k] for k in sample_keys}
                            print(f"Using standard chord mapping: {sample_mapping}")

                            # Also print the special chord mappings
                            special_keys = [k for k in idx_to_chord.keys() if k >= 168]
                            if special_keys:
                                special_mapping = {k: idx_to_chord[k] for k in special_keys}
                                print(f"Special chord mappings: {special_mapping}")

                            large_voca_score_calculation._chord_mapping_logged = True

                    # This section is intentionally left empty as we've moved the code inside the try/except blocks

                # Convert indices to chord names
                # Convert numpy arrays to integers first to make them hashable
                pred_chords = []
                for pred in predictions:
                    # Handle different prediction types
                    if isinstance(pred, np.ndarray):
                        # Convert numpy array to integer
                        pred_int = int(pred.item()) if pred.size == 1 else int(pred[0])
                        pred_chords.append(idx_to_chord.get(pred_int, "N"))
                    else:
                        # Already an integer or other hashable type
                        pred_chords.append(idx_to_chord.get(pred, "N"))


                # Collect raw chord label lists for individual chord accuracy
                collected_refs.extend(reference_labels)
                collected_preds.extend(pred_chords)

                # Debug first few predicted chords to verify format (only for the first song)
                if i == 0 and not hasattr(large_voca_score_calculation, '_first_chords_logged'):
                    print(f"First 5 predicted chords: {pred_chords[:5]}")
                    print(f"First 5 reference chords: {reference_labels[:5]}")
                    large_voca_score_calculation._first_chords_logged = True

                # Calculate scores using the refactored function
                # Use the durations we already calculated earlier
                root_score, thirds_score, triads_score, sevenths_score, tetrads_score, majmin_score, mirex_score = calculate_chord_scores(
                    timestamps, frame_duration, reference_labels, pred_chords) # frame_duration is used to calculate durations inside the function

                # Store scores
                score_list_dict['root'].append(root_score)
                score_list_dict['thirds'].append(thirds_score)
                score_list_dict['triads'].append(triads_score)
                score_list_dict['sevenths'].append(sevenths_score)
                score_list_dict['tetrads'].append(tetrads_score)
                score_list_dict['majmin'].append(majmin_score)
                score_list_dict['mirex'].append(mirex_score)

                # Store song length for weighted averaging
                song_length = len(samples) * frame_duration
                song_length_list.append(song_length)

                # Debug first few songs - FIX: Convert numpy arrays to float values before formatting
                if i < 5 and not hasattr(large_voca_score_calculation, '_song_scores_logged'):
                    # Convert scores to float values if they're numpy arrays
                    root_score_val = float(root_score) if hasattr(root_score, 'item') else root_score
                    mirex_score_val = float(mirex_score) if hasattr(mirex_score, 'item') else mirex_score
                    print(f"Song {song_id}: length={song_length:.1f}s, root={root_score_val:.4f}, mirex={mirex_score_val:.4f}")
                    # Set flag after the first song is processed
                    if i == 0:
                        large_voca_score_calculation._song_scores_logged = True

        except Exception as e:
            import traceback
            errors += 1
            print(f"Error evaluating sample group {song_id}: {str(e)}")
            if errors <= 10:  # Only print detailed error for first 10 errors
                traceback.print_exc()

    # Extra: Debug print to ensure labels were collected
    if not collected_refs:
        print("Warning: No reference chord labels were collected for individual accuracy.")
        print(f"First few predictions (if any): {collected_preds[:5]}")
    if not collected_preds:
        print("Warning: No predicted chord labels were collected for individual accuracy.")
        print(f"First few references (if any): {collected_refs[:5]}")

    # Print collected counts regardless
    print(f"Collected {len(collected_refs)} reference labels and {len(collected_preds)} predictions for chord quality analysis")

    # Extra: Print individual chord accuracy computed over all processed songs
    # Make more robust by ensuring both lists have values and equal length
    if collected_refs and collected_preds:
        # Ensure lists are the same length
        min_len = min(len(collected_refs), len(collected_preds))
        if min_len > 0:
            # Check if chord labels have the expected format (with ":")
            has_colon_ref = any(':' in str(chord) for chord in collected_refs[:100] if chord)
            has_colon_pred = any(':' in str(chord) for chord in collected_preds[:100] if chord)

            if not has_colon_ref:
                print("Warning: Reference chord labels don't contain ':' format needed for quality analysis")
                print(f"Sample references: {collected_refs[:5]}")
            if not has_colon_pred:
                print("Warning: Predicted chord labels don't contain ':' format needed for quality analysis")
                print(f"Sample predictions: {collected_preds[:5]}")

            # Trim to equal length
            ref_sample = collected_refs[:min_len]
            pred_sample = collected_preds[:min_len]

            # Now compute accuracy using our improved function
            # This will print both raw and mapped statistics for comparison
            ind_acc, quality_stats = compute_individual_chord_accuracy(ref_sample, pred_sample)

            if not ind_acc:
                print("\nNo individual chord accuracy data computed despite having labels.")
                print("This may indicate a problem with the chord format.")
    else:
        print("Warning: Insufficient data for chord quality analysis. Need both reference and prediction labels.")

    # Calculate weighted average scores - FIX: Ensure consistent lengths and non-negative values
    average_score_dict = {}

    # First ensure all score lists are of the same length as song_length_list
    valid_length = len(song_length_list)
    for metric in score_list_dict:
        # Trim or pad score lists to match song_length_list length
        current_length = len(score_list_dict[metric])
        if current_length != valid_length:
            print(f"WARNING: Length mismatch for {metric} scores ({current_length}) vs song lengths ({valid_length})")
            if current_length > valid_length:
                # Trim the score list
                score_list_dict[metric] = score_list_dict[metric][:valid_length]
                print(f"Trimmed {metric} scores to length {valid_length}")
            else:
                # Pad with zeros
                padding = [0.0] * (valid_length - current_length)
                score_list_dict[metric].extend(padding)
                print(f"Padded {metric} scores to length {valid_length}")

    if song_length_list:
        for metric in score_list_dict:
            if score_list_dict[metric]:
                # Safely calculate weighted average with matching lengths and ensure non-negative values
                weighted_sum = 0.0
                total_length = 0.0

                # Use explicit loop instead of sum() to handle each element safely
                for idx, (score, length) in enumerate(zip(score_list_dict[metric], song_length_list)):
                    # Convert score to float if it's a numpy array and ensure non-negative
                    score_val = max(0.0, float(score) if hasattr(score, 'item') else score)
                    weighted_sum += score_val * length
                    total_length += length

                average_score_dict[metric] = weighted_sum / total_length if total_length > 0 else 0.0

                # Final clipping to ensure valid range
                average_score_dict[metric] = max(0.0, min(1.0, average_score_dict[metric]))
            else:
                average_score_dict[metric] = 0.0
    else:
        for metric in score_list_dict:
            average_score_dict[metric] = 0.0

    # Clean up to free memory
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return score_list_dict, song_length_list, average_score_dict

def calculate_chord_scores(timestamps, frame_duration, reference_labels, prediction_labels):
    """
    Calculate various chord evaluation metrics correctly using mir_eval.evaluate.

    Args:
        timestamps: Array of frame start timestamps.
        frame_duration: Duration of a single frame (used to calculate the end of the last frame).
        reference_labels: List of reference chord labels (already standardized).
        prediction_labels: List of predicted chord labels (already standardized).

    Returns:
        Tuple of evaluation scores (root, thirds, triads, sevenths, tetrads, majmin, mirex)
    """
    # Ensure inputs are lists
    reference_labels = list(reference_labels)
    prediction_labels = list(prediction_labels)

    # Ensure all inputs have the same length
    min_len = min(len(timestamps), len(reference_labels), len(prediction_labels))
    if min_len == 0:
        print("Warning: Zero length input to calculate_chord_scores. Returning all zeros.")
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    timestamps = timestamps[:min_len]
    reference_labels = reference_labels[:min_len]
    prediction_labels = prediction_labels[:min_len]

    # Calculate durations from timestamps (same approach as test_labeled_audio.py)
    durations = np.diff(np.append(timestamps, [timestamps[-1] + frame_duration]))

    # Create intervals using variable durations to avoid overlaps
    ref_intervals = np.zeros((min_len, 2))
    ref_intervals[:, 0] = timestamps
    ref_intervals[:, 1] = timestamps + durations

    est_intervals = np.zeros((min_len, 2))
    est_intervals[:, 0] = timestamps
    est_intervals[:, 1] = timestamps + durations

    # Use mir_eval.chord.evaluate for robust calculation
    scores = {}
    try:
        # mir_eval.chord.evaluate handles merging, weighting, and calculates all metrics
        scores = mir_eval.chord.evaluate(ref_intervals, reference_labels, est_intervals, prediction_labels)

        # Extract scores safely, defaulting to 0.0 if a metric is missing
        root_score = float(scores.get('root', 0.0))
        thirds_score = float(scores.get('thirds', 0.0))
        triads_score = float(scores.get('triads', 0.0))
        sevenths_score = float(scores.get('sevenths', 0.0))
        tetrads_score = float(scores.get('tetrads', 0.0))
        majmin_score = float(scores.get('majmin', 0.0))
        mirex_score = float(scores.get('mirex', 0.0))

    except Exception as e:
        print(f"Error during mir_eval.chord.evaluate: {e}")
        # Return default zero scores on error
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # Ensure all scores are within the valid range [0, 1]
    root_score = max(0.0, min(1.0, root_score))
    thirds_score = max(0.0, min(1.0, thirds_score))
    triads_score = max(0.0, min(1.0, triads_score))
    sevenths_score = max(0.0, min(1.0, sevenths_score))
    tetrads_score = max(0.0, min(1.0, tetrads_score))
    majmin_score = max(0.0, min(1.0, majmin_score))
    mirex_score = max(0.0, min(1.0, mirex_score))

    return root_score, thirds_score, triads_score, sevenths_score, tetrads_score, majmin_score, mirex_score