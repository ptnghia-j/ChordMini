import os
import librosa
import numpy as np
import mir_eval
import signal
import time
import re
import traceback
import torch # Add torch import

# Add soundfile import and handle potential missing attribute
import soundfile as sf
if not hasattr(sf, 'SoundFileRuntimeError'):
    sf.SoundFileRuntimeError = RuntimeError
from tqdm import tqdm
# Import _parse_chord_string and idx2voca_chord from chords.py
# PREFERRED_SPELLING_MAP, PITCH_CLASS, _parse_root are no longer needed here
from modules.utils.chords import idx2voca_chord, _parse_chord_string, QUALITY_CATEGORIES # Import QUALITY_CATEGORIES
from modules.utils.logger import info, warning, error, debug # Add logger imports

# --- Enharmonic Mapping ---
# Removed ENHARMONIC_MAP, as _parse_chord_string from chords.py handles this.

# --- Quality Simplification Mapping ---
# Maps complex/alternative quality names to the 14 target vocabulary names for the 170-class model
# The 14 target qualities are: min, maj, dim, aug, min6, maj6, min7, minmaj7, maj7, 7, dim7, hdim7, sus2, sus4
TARGET_QUALITIES_SET = {'min', 'maj', 'dim', 'aug', 'min6', 'maj6', 'min7', 'minmaj7', 'maj7', '7', 'dim7', 'hdim7', 'sus2', 'sus4'}

QUALITY_SIMPLIFICATION_MAP = {
    # Basic synonyms (ensure target is one of the 14 qualities)
    "major": "maj",
    "minor": "min",
    "dominant": "7",
    "major-seventh": "maj7",
    "minor-seventh": "min7",
    "diminished": "dim",
    "augmented": "aug",
    "suspended-fourth": "sus4",
    "suspended-second": "sus2",
    "dominant-seventh": "7",
    "half-diminished": "hdim7", # Usually m7b5
    "diminished-seventh": "dim7",
    "major-minor": "minmaj7", # minmaj7 is a target quality
    "major-sixth": "maj6",
    "minor-sixth": "min6",
    "6": "maj6", # Explicitly map '6' (often from :6) to 'maj6'
    "7sus": "sus4",
    "7sus4": "sus4",
    "sus": "sus4", # Default sus to sus4

    # Extensions (map to closest base type in TARGET_QUALITIES_SET)
    "9": "7",
    "maj9": "maj7",
    "min9": "min7",
    "11": "7",
    "maj11": "maj7",
    "min11": "min7",
    "13": "7",
    "maj13": "maj7",
    "min13": "min7",
    "69": "maj6",
    "7#9": "7",
    "7b9": "7",
    "maj7#11": "maj7",
    "maj7b5": "maj7",
    "m7b5": "hdim7", # This is hdim7
    "aug7": "aug", # Map aug7 to aug (aug is a target quality)

    # Power chords (map '5' to 'maj' as '5' is not in target qualities)
    "5": "maj",
    # Other variations (map to closest in TARGET_QUALITIES_SET)
    "add9": "maj",
    "add2": "maj",
    "add4": "maj", # Or sus4? 'maj' is safer if not specified as sus.
    "add11": "maj7", # C:maj(add11) -> C:maj7
    "maj(9)": "maj7",
    "min(9)": "min7",
    "min(11)": "min7",
    "maj(11)": "maj7",
    "maj(13)": "maj7",
    "min(13)": "min7",
    # ... (other existing mappings if they simplify to one of the 14 targets) ...
    # Ensure common outputs from _parse_chord_string's QUALITY_NORM_MAP are covered if they aren't target qualities
    # For example, if _parse_chord_string produces 'alt', it should be mapped here.
    # chords.py QUALITY_NORM_MAP has 'alt': '7', which is fine.

    # MIREX special qualities (map to one of 14 targets)
    "1": "maj", # Root only -> major
    # "3": "maj", # Root and third -> major (less common as direct quality string)
    # "4": "sus4", # Already a target
    # "2": "sus2", # Already a target

    # Ensure target qualities map to themselves (important!)
    **{q: q for q in TARGET_QUALITIES_SET}
}

# --- Root notes for validation ---
# Removed ROOT_NOTES and MODIFIERS, as _parse_chord_string from chords.py handles this.

def audio_file_to_features(audio_file, config, timeout=60):
    """
    Extract features from an audio file with timeout protection.

    Args:
        audio_file: Path to the audio file
        config: Configuration object
        timeout: Maximum time in seconds to spend loading the audio file (default: 60)

    Returns:
        feature: Extracted features
        frame_duration: Duration of each frame in seconds
        song_length_second: Total length of the song in seconds
    """

    # Define a timeout handler
    class TimeoutError(Exception):
        pass

    def timeout_handler(signum, frame):
        raise TimeoutError(f"Audio loading timed out after {timeout} seconds")

    if not os.path.isfile(audio_file):
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    # Debug info: file exists, log its size
    file_size = os.path.getsize(audio_file)
    if file_size < 5000: # Increased threshold slightly
        raise RuntimeError(f"Audio file '{audio_file}' (size: {file_size} bytes) is too small and may be corrupt.")

    # Check if we're in MIR evaluation mode and should skip this file
    # This is set by train_finetune.py when using small_dataset_percentage
    if hasattr(config, 'skip_audio_files') and isinstance(config.skip_audio_files, set):
        audio_basename = os.path.basename(audio_file)
        if audio_basename in config.skip_audio_files:
            from modules.utils.logger import info
            info(f"Skipping audio file due to small_dataset_percentage: {audio_file}")
            # Return dummy values instead of raising error to allow processing other files
            # raise RuntimeError(f"Audio file '{audio_file}' skipped due to small_dataset_percentage setting.")
            dummy_feature = np.zeros((config.feature['n_bins'], 10)) # Dummy shape
            dummy_frame_duration = config.feature['hop_length'] / config.mp3['song_hz']
            dummy_song_length = 1.0
            return dummy_feature, dummy_frame_duration, dummy_song_length


    try:
        from modules.utils.logger import info
        info(f"Loading features from audio file: {audio_file}")

        # Check if the file is in the cache first
        cache_path = None
        if hasattr(config, 'cache_dir') and config.cache_dir:
            # Use a consistent naming scheme for cache files
            audio_basename = os.path.basename(audio_file)
            cache_filename = f"{os.path.splitext(audio_basename)[0]}_cqt.npy"
            cache_path = os.path.join(config.cache_dir, cache_filename)
            if os.path.exists(cache_path):
                try:
                    info(f"Loading features from cache: {cache_path}")
                    data = np.load(cache_path, allow_pickle=True).item()
                    # Verify expected keys exist
                    if all(k in data for k in ['feature', 'frame_duration', 'song_length_second']):
                        return data['feature'], data['frame_duration'], data['song_length_second']
                    else:
                        info(f"Cache file {cache_path} is missing expected keys. Recomputing.")
                except Exception as e:
                    info(f"Error loading cached features from {cache_path}: {e}. Recomputing.")
                    # Attempt to remove corrupted cache file
                    try:
                        os.remove(cache_path)
                    except OSError:
                        pass # Ignore if removal fails

        # Set up timeout for audio loading
        original_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)

        try:
            # Load the audio file with timeout protection
            info(f"Loading audio file with {timeout}s timeout: {audio_file}")
            start_time = time.time()
            # Use soundfile for potentially more robust loading
            try:
                with sf.SoundFile(audio_file, 'r') as f:
                    sr_native = f.samplerate
                    # Read and resample if necessary
                    if sr_native != config.mp3['song_hz']:
                        info(f"Resampling from {sr_native} Hz to {config.mp3['song_hz']} Hz")
                        # Read frames and resample chunk by chunk if file is large?
                        # For now, load all and resample
                        original_wav_native = f.read(dtype='float32')
                        # Ensure mono
                        if original_wav_native.ndim > 1:
                            original_wav_native = np.mean(original_wav_native, axis=1)
                        original_wav = librosa.resample(original_wav_native, orig_sr=sr_native, target_sr=config.mp3['song_hz'])
                    else:
                        original_wav = f.read(dtype='float32')
                        # Ensure mono
                        if original_wav.ndim > 1:
                            original_wav = np.mean(original_wav, axis=1)
            except Exception as sf_err:
                 info(f"SoundFile failed ({sf_err}), falling back to librosa.load for {audio_file}")
                 original_wav, sr = librosa.load(audio_file, sr=config.mp3['song_hz'], mono=True)

            load_time = time.time() - start_time
            info(f"Audio loaded in {load_time:.2f}s: {audio_file} (length: {len(original_wav)} samples)")

            # Cancel the alarm
            signal.alarm(0)
        except TimeoutError as e:
            info(f"WARNING: {str(e)} - {audio_file}")
            raise RuntimeError(f"Audio loading timed out: {audio_file}")
        except sf.SoundFileRuntimeError as e:
             raise RuntimeError(f"SoundFile failed to load audio file '{audio_file}': {e}")
        except Exception as e:
            # Catch librosa specific errors if possible
            if "NoBackendError" in str(e):
                 raise RuntimeError(f"Librosa failed to load audio file '{audio_file}' due to missing backend. Install ffmpeg or soundfile.")
            else:
                 raise RuntimeError(f"Failed to load audio file '{audio_file}': {e}")
        finally:
            # Restore the original signal handler
            signal.signal(signal.SIGALRM, original_handler)

    except Exception as e:
        # Ensure any exception during loading is caught and re-raised
        raise RuntimeError(f"Failed during audio loading stage for '{audio_file}': {e}")

    # Get FFT size from config or use default
    n_fft = config.feature.get('n_fft', 2048) # Use a more standard n_fft for CQT
    hop_length = config.feature.get('hop_length', 512)
    n_bins = config.feature.get('n_bins', 144) # Usually 12*octaves*bins_per_octave
    bins_per_octave = config.feature.get('bins_per_octave', 24)

    # Calculate CQT for the entire audio at once (more efficient)
    try:
        # Pad the signal at the beginning and end for CQT calculation
        # This helps capture transients at the very start/end
        # Padding amount depends on the filter lengths used by CQT
        # A safe estimate is related to the lowest frequency analyzed
        # fmin = librosa.note_to_hz('C1') # Example minimum frequency
        # padding_samples = int(sr / fmin * 2) # Heuristic padding
        # original_wav_padded = np.pad(original_wav, padding_samples, mode='reflect')

        # Use librosa's padding handling within cqt
        feature = librosa.cqt(original_wav,
                              sr=config.mp3['song_hz'],
                              n_bins=n_bins,
                              bins_per_octave=bins_per_octave,
                              hop_length=hop_length,
                              fmin=librosa.note_to_hz('C1'), # Explicitly set fmin
                              tuning=0.0) # Assume standard tuning

    except Exception as cqt_err:
        raise RuntimeError(f"Error during CQT calculation for '{audio_file}': {cqt_err}")


    # Check for empty features which can happen with very short files
    if feature.shape[1] == 0:
        raise RuntimeError(f"CQT resulted in zero frames for '{audio_file}'. Audio might be too short or silent.")

    feature = np.log(np.abs(feature) + 1e-6)
    song_length_second = len(original_wav) / config.mp3['song_hz']

    # Calculate frames per second and frame duration
    # frame duration in seconds per frame
    frame_duration = hop_length / config.mp3['song_hz']

    # Add diagnostic info for debugging
    print(f"Audio file: {os.path.basename(audio_file)}")
    print(f"CQT shape: {feature.shape}, Frame duration: {frame_duration:.5f}s, Total duration: {song_length_second:.2f}s")

    # Save to cache if enabled
    if cache_path:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.save(cache_path, {
                'feature': feature,
                'frame_duration': frame_duration,
                'song_length_second': song_length_second
            })
            info(f"Saved features to cache: {cache_path}")
        except Exception as e:
            info(f"Warning: Failed to save features to cache {cache_path}: {e}")


    # Return frame-level CQT, frame_duration (s), and song length (s)
    return feature, frame_duration, song_length_second

# Audio files with format of wav and mp3
def get_audio_paths(audio_dir):
    return [os.path.join(root, fname) for (root, dir_names, file_names) in os.walk(audio_dir, followlinks=True)
            for fname in file_names if (fname.lower().endswith('.wav') or fname.lower().endswith('.mp3'))]

def standardize_chord_label(chord_label_orig):
    """
    Standardize a single chord label to ensure compatibility with mir_eval
    and map it to the 170-chord vocabulary space (based on 14 core qualities).
    Uses _parse_chord_string from chords.py for robust initial parsing.

    Args:
        chord_label_orig: A single chord label string to standardize

    Returns:
        Standardized chord label string (e.g., "C", "Bb:min", "N")
    """
    # 1. Handle None, empty, or explicit "No Chord" variants
    if not chord_label_orig or chord_label_orig.lower() in ["n", "none", "nc", "no chord"]:
        return "N"

    # 2. Handle explicit "Unknown" variants but map to "N" for mir_eval compatibility
    if chord_label_orig.upper() == "X" or chord_label_orig.lower() == "unknown":
        return "N" # Map X to N

    # 3. Use _parse_chord_string from chords.py for initial parsing and normalization
    # It returns (root_str, quality_str, bass_str)
    # root_str is normalized (e.g., Bb), quality_str is normalized (e.g., min, maj7, 7, maj9)
    # bass_str is normalized or scale degree (we ignore bass for this standardization)
    parsed_root, parsed_quality, _ = _parse_chord_string(chord_label_orig)

    # 4. Handle parsing results
    if parsed_root == "N" or parsed_root == "X":
        return "N"
    if parsed_root is None: # Indicates a more fundamental parsing failure
        # debug(f"Failed to parse chord label '{chord_label_orig}' using _parse_chord_string. Mapping to 'N'.")
        return "N"

    # 5. Apply QUALITY_SIMPLIFICATION_MAP to map the parsed_quality to one of the 14 target qualities
    # The map should ensure that if parsed_quality is already a target quality, it remains unchanged.
    simplified_quality = QUALITY_SIMPLIFICATION_MAP.get(parsed_quality, parsed_quality)

    # 6. If simplification results in a quality not in TARGET_QUALITIES_SET, map to "N"
    # This ensures the final quality is one of the 14 targets or the chord becomes "N".
    if simplified_quality not in TARGET_QUALITIES_SET:
        # debug(f"Quality '{parsed_quality}' from '{chord_label_orig}' simplified to '{simplified_quality}', which is not in TARGET_QUALITIES_SET. Mapping to 'N'.")
        return "N"

    # 7. Construct final label based on the 170-class vocabulary format
    # (Root for major, Root:Quality otherwise)
    final_label = "N" # Default to N
    # Cache voca_chords_set if this function is called in a tight loop for performance
    # For now, recreate it each time for simplicity.
    voca_chords_set = set(idx2voca_chord().values())

    if simplified_quality == 'maj':
        candidate_label = parsed_root # e.g., "C", "Bb"
        if candidate_label in voca_chords_set:
            final_label = candidate_label
        # else:
            # debug(f"Root-only major label '{candidate_label}' (from original '{chord_label_orig}') not in vocabulary. Mapping to 'N'.")
    else:
        candidate_label = f"{parsed_root}:{simplified_quality}" # e.g., "C:min", "Bb:7"
        if candidate_label in voca_chords_set:
            final_label = candidate_label
        # else:
            # debug(f"Label '{candidate_label}' (from original '{chord_label_orig}') not in vocabulary. Mapping to 'N'.")

    # Optional: Detailed logging for changes or failures
    # if final_label != "N" and chord_label_orig != final_label :
    #    debug(f"Standardized '{chord_label_orig}' to '{final_label}' (parsed root: '{parsed_root}', parsed qual: '{parsed_quality}', simplified qual: '{simplified_quality}')")
    # elif final_label == "N" and chord_label_orig.upper() not in ["N", "X", "NC", "NONE", "NO CHORD", "UNKNOWN"]:
    #    debug(f"Failed to standardize '{chord_label_orig}' (parsed root: '{parsed_root}', parsed qual: '{parsed_quality}', simplified qual: '{simplified_quality}'). Mapped to 'N'.")

    return final_label


def lab_file_error_modify(ref_labels):
    """
    Standardize chord labels (single or list) using the enhanced
    standardize_chord_label function.

    Args:
        ref_labels: List of chord labels to standardize or a single chord label

    Returns:
        List of standardized chord labels or a single standardized chord label
    """
    if isinstance(ref_labels, str):
        return standardize_chord_label(ref_labels)
    elif isinstance(ref_labels, (list, tuple)):
        # Check if it's a list of lists (e.g., from batch processing) - flatten if necessary
        if ref_labels and isinstance(ref_labels[0], (list, tuple)):
             # Flatten the list
             flattened_labels = [label for sublist in ref_labels for label in sublist]
             return [standardize_chord_label(label) for label in flattened_labels]
        else:
             # Standard list processing
             return [standardize_chord_label(label) for label in ref_labels]
    elif ref_labels is None:
        return "N" # Handle None input
    else:
        # Handle other potential types (e.g., numpy array elements)
        try:
            return standardize_chord_label(str(ref_labels))
        except Exception:
            # print(f"Warning: Could not convert label '{ref_labels}' (type: {type(ref_labels)}) to string for standardization. Mapping to 'N'.")
            return "N"


def extract_chord_quality(chord):
    """
    Extract chord quality from a standardized chord label (e.g., "C:maj").
    Assumes input chord is already standardized by lab_file_error_modify.

    Args:
        chord: A standardized chord label string (root:quality or root for major)

    Returns:
        The chord quality as a string, or "N" / "X" if applicable.
    """
    # Handle None or empty strings
    if not chord:
        return "N"

    # Handle special cases first (standardized form)
    if chord == "N":
        return "N"
    # We map X to N during standardization, so X shouldn't appear here normally
    if chord == "X":
        return "X" # Keep X if it somehow gets here

    # Standardized chords should have a colon, EXCEPT for major chords
    if ':' in chord:
        parts = chord.split(':', 1)
        if len(parts) == 2:
            quality = parts[1]
            # Further simplify quality if needed using the map?
            # No, assume quality is already simplified by standardization.
            return quality
        else:
            # Malformed standardized chord? e.g., "C:"
            # print(f"Warning: Malformed standardized chord '{chord}' in extract_chord_quality. Assuming major.")
            return "maj" # Fallback to major
    else:
        # If no colon, it should be a root-only label representing a major chord,
        # or an invalid label.
        # Check if it's just a root note (A-G with optional #/b)
        root_match = re.match(r'([A-G][#b]?)$', chord)
        if root_match:
            return "maj" # Root only implies major in the standardized vocabulary
        else:
            # If it's not just a root, it's likely invalid. Map to N.
            # print(f"Warning: Non-standardized chord '{chord}' passed to extract_chord_quality. Assuming N.")
            return "N"


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

    # Always import the map_chord_to_quality function from visualize.py for consistency
    try:
        # Use the standardized extract_chord_quality from this file now
        from modules.utils.visualize import map_chord_to_quality # Keep map_chord_to_quality
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
    # Note: This mapping is applied AFTER extracting the quality from the standardized label
    # REMOVE local quality_mapping_broad
    # quality_mapping_broad = {
    #     # Major family
    #     "maj": "Major", "": "Major", "M": "Major", "major": "Major",
    #     # Minor family
    #     "min": "Minor", "m": "Minor", "minor": "Minor",
    #     # Dominant seventh family
    #     "7": "Dom7", "dom7": "Dom7", "dominant": "Dom7",
    #     # Major seventh family
    #     "maj7": "Maj7", "M7": "Maj7", "major7": "Maj7",
    #     # Minor seventh family
    #     "min7": "Min7", "m7": "Min7", "minor7": "Min7",
    #     # Diminished family
    #     "dim": "Dim", "°": "Dim", "o": "Dim", "diminished": "Dim",
    #     # Diminished seventh family
    #     "dim7": "Dim7", "°7": "Dim7", "o7": "Dim7", "diminished7": "Dim7",
    #     # Half-diminished family
    #     "hdim7": "Half-Dim", "m7b5": "Half-Dim", "ø": "Half-Dim", "half-diminished": "Half-Dim",
    #     # Augmented family
    #     "aug": "Aug", "+": "Aug", "augmented": "Aug",
    #     # Suspended family
    #     "sus2": "Sus", "sus4": "Sus", "sus": "Sus", "suspended": "Sus",
    #     # Additional common chord qualities
    #     "min6": "Min6", "m6": "Min6",
    #     "maj6": "Maj6", "6": "Maj6",
    #     "minmaj7": "Min-Maj7", "mmaj7": "Min-Maj7", "min-maj7": "Min-Maj7",
    #     # Special cases
    #     "N": "No Chord",
    #     "X": "No Chord",  # Map X to No Chord for consistency with validation
    # }

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
        for i, (ref_orig, pred_orig) in enumerate(zip(ref_chunk, pred_chunk)):
            try:
                # Let mir_eval handle standardization internally
                # We'll use the original labels directly for quality extraction
                # This matches how we're now handling labels in calculate_chord_scores

                # Extract chord qualities directly from original labels
                q_ref_raw = extract_chord_quality(ref_orig)
                q_pred_raw = extract_chord_quality(pred_orig)

                # Handle special cases
                q_ref_mapped = "Other"  # Default mapping
                q_pred_mapped = "Other"  # Default mapping

                if ref_orig == "N" or ref_orig == "X" or q_ref_raw == "N":
                    q_ref_mapped = "No Chord"
                elif pred_orig == "N" or pred_orig == "X" or q_pred_raw == "N":
                    q_pred_mapped = "No Chord"
                else:
                    # Map to broader categories using the visualize.py mapping if available
                    if use_quality_mapping:
                        # map_chord_to_quality expects the full chord label
                        q_ref_mapped = map_chord_to_quality(ref_orig)
                        q_pred_mapped = map_chord_to_quality(pred_orig)
                    else:
                        # Fallback to local broad mapping using QUALITY_CATEGORIES from chords.py
                        q_ref_mapped = QUALITY_CATEGORIES.get(q_ref_raw, "Other")
                        q_pred_mapped = QUALITY_CATEGORIES.get(q_pred_raw, "Other")

                # Update raw statistics (using the extracted, simplified quality)
                raw_stats[q_ref_raw]['total'] += 1
                if q_ref_raw == q_pred_raw:
                    raw_stats[q_ref_raw]['correct'] += 1

                # Update mapped statistics (using the broad category mapping)
                mapped_stats[q_ref_mapped]['total'] += 1
                if q_ref_mapped == q_pred_mapped:
                    mapped_stats[q_ref_mapped]['correct'] += 1

            except Exception as e:
                # print(f"Error processing pair: ref='{ref_orig}', pred='{pred_orig}'. Error: {e}")
                malformed_chords += 1
                continue

        # Update total processed
        total_processed += chunk_size_actual

        # Print progress every 100,000 samples
        if total_processed % 100000 == 0 or total_processed == total_samples:
            elapsed = time.time() - start_time
            print(f"Processed {total_processed}/{total_samples} samples ({total_processed/total_samples*100:.1f}%) in {elapsed:.1f}s")

    print(f"Processed {total_processed} samples, {malformed_chords} were malformed or caused errors during quality analysis")

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
    print("\nRaw Chord Quality Distribution (Simplified):")
    total_raw = sum(stats['total'] for stats in raw_stats.values())
    if total_raw > 0:
        for quality, stats in sorted(raw_stats.items(), key=lambda x: x[1]['total'], reverse=True):
            if stats['total'] > 0:
                percentage = (stats['total'] / total_raw) * 100
                print(f"  {quality}: {stats['total']} samples ({percentage:.2f}%)")
    else:
        print("  No raw quality data collected.")


    print("\nMapped Chord Quality Distribution (Broad Categories - matches validation):")
    total_mapped = sum(stats['total'] for stats in mapped_stats.values())
    if total_mapped > 0:
        for quality, stats in sorted(mapped_stats.items(), key=lambda x: x[1]['total'], reverse=True):
            if stats['total'] > 0:
                percentage = (stats['total'] / total_mapped) * 100
                print(f"  {quality}: {stats['total']} samples ({percentage:.2f}%)")
    else:
        print("  No mapped quality data collected.")


    print("\nRaw Accuracy by chord quality (Simplified):")
    raw_accuracy_items = sorted(raw_acc.items(), key=lambda x: raw_stats[x[0]]['total'], reverse=True) # Sort by count
    count_raw = 0
    for quality, accuracy_val in raw_accuracy_items:
        if raw_stats[quality]['total'] >= 10:  # Only show meaningful stats
            print(f"  {quality}: {accuracy_val:.4f} (Count: {raw_stats[quality]['total']})")
            count_raw += 1
    if count_raw == 0: print("  No raw qualities with sufficient samples (>=10).")


    print("\nMapped Accuracy by chord quality (Broad Categories - matches validation):")
    mapped_accuracy_items = sorted(mapped_acc.items(), key=lambda x: mapped_stats[x[0]]['total'], reverse=True) # Sort by count
    count_mapped = 0
    for quality, accuracy_val in mapped_accuracy_items:
        if mapped_stats[quality]['total'] >= 10:  # Only show meaningful stats
            print(f"  {quality}: {accuracy_val:.4f} (Count: {mapped_stats[quality]['total']})")
            count_mapped += 1
    if count_mapped == 0: print("  No mapped qualities with sufficient samples (>=10).")


    # Return the mapped statistics for consistency with validation
    return mapped_acc, mapped_stats

def large_voca_score_calculation(valid_dataset, config, model, model_type, mean, std, device=None, sampled_song_ids=None):
    """
    Calculate MIR evaluation scores using the model on a validation dataset.
    Uses original chord labels for evaluation, letting mir_eval handle standardization internally.

    Args:
        valid_dataset: List of samples for evaluation (dictionaries expected)
        config: Configuration object
        model: Model to evaluate
        model_type: Type of the model (not used, kept for backward compatibility)
        mean: Mean value for normalization
        std: Standard deviation for normalization
        device: Device to run evaluation on (default: None, will use model's device)
        sampled_song_ids: Set of song IDs to include in evaluation (for small dataset percentage)

    Returns:
        score_list_dict: Dictionary of score lists for each metric
        song_length_list: List of song lengths for weighting
        average_score_dict: Dictionary of average scores for each metric
    """
    # Check if we're using small_dataset_percentage and have skip_audio_files set
    if hasattr(config, 'skip_audio_files') and isinstance(config.skip_audio_files, set):
        from modules.utils.logger import info
        info(f"Using skip_audio_files list with {len(config.skip_audio_files)} files to skip")
        if sampled_song_ids:
            info(f"Using sampled_song_ids with {len(sampled_song_ids)} song IDs to include")

    # Ensure valid_dataset is a list of dictionaries
    if not isinstance(valid_dataset, list) or (valid_dataset and not isinstance(valid_dataset[0], dict)):
         print(f"Error: large_voca_score_calculation expects valid_dataset to be a list of dictionaries, but got {type(valid_dataset)}")
         # Return empty results
         empty_scores = {m: [] for m in ['root', 'thirds', 'triads', 'sevenths', 'tetrads', 'majmin', 'mirex']}
         empty_avg = {m: 0.0 for m in empty_scores}
         return empty_scores, [], empty_avg

    print(f"Processing list of {len(valid_dataset)} samples for evaluation")

    # Log if we're using a subset of songs
    if sampled_song_ids:
        print(f"Using only {len(sampled_song_ids)} sampled song IDs for evaluation")

    # Ensure model is in evaluation mode
    model.eval()

    # Get device from model if not specified
    if device is None:
        device = next(model.parameters()).device

    # Log normalization parameters for debugging
    debug(f"MIR Eval received normalization parameters: mean={mean}, std={std}")

    # Ensure mean and std are on the correct device and are tensors
    if not isinstance(mean, torch.Tensor): mean = torch.tensor(mean)
    if not isinstance(std, torch.Tensor): std = torch.tensor(std)
    # Ensure std is not zero BEFORE converting to tensor if it's a float/int
    if isinstance(std, (float, int)) and std == 0:
        warning("Input std is zero, using 1.0 instead.")
        std = 1.0
    elif isinstance(std, torch.Tensor) and torch.all(std == 0):
        warning("Input std tensor is zero, using 1.0 instead.")
        std = torch.tensor(1.0)

    mean = mean.to(device, dtype=torch.float32)
    std = std.to(device, dtype=torch.float32)
    # Add epsilon during division later, not to the std tensor itself
    # std = std + 1e-6 # REMOVE THIS

    # Group samples by song_id to evaluate whole songs
    song_groups = {}
    for i, sample in enumerate(valid_dataset):
        if not isinstance(sample, dict):
            print(f"Warning: Skipping invalid sample at index {i} (not a dict): {sample}")
            continue
        song_id = sample.get('song_id', f'unknown_song_{i}') # Use index if song_id is missing
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
    all_metrics = list(score_list_dict.keys()) # For convenience

    song_length_list = []
    collected_refs_original = []     # <-- Collect original reference chord labels
    collected_preds_original = []    # <-- Collect original predicted chord labels
    errors = 0

    # Process each song group
    for i, (song_id, samples) in enumerate(tqdm(song_groups.items(),
                                             desc="Evaluating songs",
                                             total=len(song_groups))):
        try:
            # Skip if we're using sampled_song_ids and this song is not in the set
            if sampled_song_ids is not None and song_id not in sampled_song_ids:
                continue

            # Skip if any sample's audio file is in the skip_audio_files set
            if hasattr(config, 'skip_audio_files') and isinstance(config.skip_audio_files, set):
                should_skip = False
                for sample in samples:
                    audio_path = sample.get('audio_path', '')
                    if audio_path and os.path.basename(audio_path) in config.skip_audio_files:
                        from modules.utils.logger import info
                        info(f"Skipping song {song_id} due to audio file {os.path.basename(audio_path)} in skip_audio_files")
                        should_skip = True
                        break
                if should_skip:
                    continue

            # Extract and sort samples by frame index if available
            if samples and all('frame_idx' in sample for sample in samples):
                samples.sort(key=lambda x: x['frame_idx'])
            else:
                # If frame_idx is missing, assume samples are already in order
                pass # print(f"Warning: frame_idx missing for song {song_id}, assuming samples are ordered.")


            # Determine frame_duration from model config or fallback
            frame_duration = config.feature.get('hop_duration', 0.09288) # Use hop_duration from config

            # Build timestamps from sample frame_idx when available
            if samples and 'frame_idx' in samples[0]:
                timestamps = np.array([s['frame_idx'] * frame_duration for s in samples])
            else:
                # If frame_idx is missing, generate sequential timestamps
                timestamps = np.arange(len(samples)) * frame_duration

            # Extract spectrograms and reference labels
            spectrograms = []
            reference_labels_orig = [] # Store original labels before standardization

            for sample_idx, sample in enumerate(samples):
                # Handle different sample formats for spectrograms
                spec = None
                if 'spectro' in sample:
                    spec = sample['spectro']
                elif 'spec_path' in sample:
                    try:
                        spec_data = np.load(sample['spec_path'])
                        # Handle potential frame indexing within a larger file (less common with SynthDataset)
                        if 'frame_idx' in sample and len(spec_data.shape) > 1 and spec_data.shape[0] > 1:
                            frame_idx = sample['frame_idx']
                            # Find the corresponding frame in the loaded spec_data if it represents multiple frames
                            # This logic might need adjustment based on how spec_path files are structured
                            # Assuming spec_data is [time, features] and corresponds to the whole song
                            # We need the frame index relative to the start of the song's spectrogram
                            # This info isn't directly in the sample, SynthDataset usually provides single frames/sequences
                            # Let's assume spec_data is already the correct frame/sequence
                            if frame_idx < spec_data.shape[0]:
                                spec = spec_data[frame_idx] # Or handle sequence slicing if needed
                            else:
                                print(f"Warning: frame_idx {frame_idx} out of bounds for spec_path {sample['spec_path']} shape {spec_data.shape}. Using zeros.")
                                spec = np.zeros_like(spec_data[0]) if spec_data.shape[0] > 0 else np.zeros((144,)) # Use shape from config?
                        else:
                             # Assume spec_data is the correct single frame or sequence
                             spec = spec_data
                    except Exception as load_err:
                        print(f"Error loading spectrogram {sample['spec_path']}: {load_err}. Using zeros.")
                        spec = np.zeros((config.feature.get('n_bins', 144),)) # Use shape from config
                else:
                    print(f"Warning: No spectrogram data ('spectro' or 'spec_path') found for sample {sample_idx} in song {song_id}. Using zeros.")
                    spec = np.zeros((config.feature.get('n_bins', 144),)) # Use shape from config

                # Convert numpy array to tensor if needed
                if isinstance(spec, np.ndarray):
                    # Ensure correct dtype (float32)
                    spec = torch.from_numpy(spec.astype(np.float32))

                # Add batch/channel dimension if needed (assuming [features] -> [1, features])
                # Or handle sequence dimension [seq_len, features] -> [1, seq_len, features]?
                # SynthDataset usually yields [features], model expects [batch, seq, features] or [batch, features]
                # Let's assume model handles single frames correctly if needed. Stack later.
                # if spec.dim() == 1:
                #     spec = spec.unsqueeze(0) # Add a dimension

                spectrograms.append(spec)

                # Extract the reference chord label
                chord_label = sample.get('chord_label', 'N') # Default to 'N' if missing
                reference_labels_orig.append(chord_label)

            # Combine all frames into a single tensor for the song
            if not spectrograms:
                print(f"Warning: No spectrograms collected for song {song_id}. Skipping.")
                continue

            # Stack spectrograms along the time dimension (dim=0) -> [num_frames, features]
            try:
                # Ensure all tensors have the same feature dimension before stacking
                feature_dim = spectrograms[0].shape[-1]
                valid_spectrograms = []
                for idx, spec in enumerate(spectrograms):
                     if spec.shape[-1] == feature_dim:
                         # Ensure spec is 1D (features) before stacking if needed
                         if spec.dim() > 1: spec = spec.squeeze() # Adjust based on expected input
                         valid_spectrograms.append(spec)
                     else:
                         print(f"Warning: Spectrogram {idx} in song {song_id} has incorrect feature dim {spec.shape[-1]} (expected {feature_dim}). Skipping frame.")
                         # Also remove corresponding label and timestamp? This gets complex.
                         # For now, just skip the spectrogram. Length mismatch handled later.
                if not valid_spectrograms:
                     print(f"Warning: No valid spectrograms left for song {song_id} after shape check. Skipping.")
                     continue

                input_tensor = torch.stack(valid_spectrograms, dim=0) # [num_frames, features]
            except Exception as stack_err:
                print(f"Error stacking spectrograms for song {song_id}: {stack_err}")
                # Print shapes for debugging
                # for idx, s in enumerate(spectrograms): print(f"  Spectrogram {idx} shape: {s.shape}")
                errors += 1
                continue

            # Add batch dimension -> [1, num_frames, features]
            input_tensor = input_tensor.unsqueeze(0).to(device)

            # Apply normalization
            # Ensure shapes are compatible for broadcasting
            try:
                # Assuming mean/std are scalar or match the last dimension (features)
                norm_mean = mean
                norm_std = std
                # Reshape mean/std if they are per-feature and input is [B, T, F]
                if norm_mean.numel() > 1 and norm_mean.shape[-1] == input_tensor.shape[-1]:
                    norm_mean = norm_mean.view(1, 1, -1) # Reshape to [1, 1, F]
                if norm_std.numel() > 1 and norm_std.shape[-1] == input_tensor.shape[-1]:
                    norm_std = norm_std.view(1, 1, -1) # Reshape to [1, 1, F]

                # Check shapes before applying
                debug(f"MIR Eval Normalization: Input shape {input_tensor.shape}, Mean shape {norm_mean.shape}, Std shape {norm_std.shape}")

                input_tensor = (input_tensor - norm_mean) / (norm_std + 1e-6) # Add epsilon here
            except RuntimeError as norm_err:
                 error(f"Error applying normalization during MIR eval for song {song_id}: {norm_err}")
                 error(f"Input shape: {input_tensor.shape}, Mean shape: {mean.shape}, Std shape: {std.shape}")
                 errors += 1
                 continue


            # Get predictions from the model
            predictions_idx = []
            with torch.no_grad():
                try:
                    # Model might expect sequences, process in chunks if necessary
                    # For now, assume model can handle variable length [1, num_frames, features]
                    # Or that the trainer/model handles sequence batching internally
                    output = model(input_tensor) # Shape: [1, num_frames, n_classes]

                    # Handle different output formats
                    if isinstance(output, tuple):
                        logits = output[0]
                    else:
                        logits = output

                    # Get predicted class indices
                    if logits.dim() == 3: # [batch, time, classes]
                        predictions_idx = logits.argmax(dim=2).squeeze().cpu().numpy() # [num_frames]
                    elif logits.dim() == 2: # Maybe [time, classes]?
                         predictions_idx = logits.argmax(dim=1).cpu().numpy()
                    else:
                         print(f"Warning: Unexpected model output shape {logits.shape} for song {song_id}. Skipping.")
                         continue

                    # Ensure predictions_idx is iterable (handle single frame case)
                    if not hasattr(predictions_idx, '__len__'):
                        predictions_idx = [predictions_idx.item()]


                except Exception as e:
                    print(f"Error getting predictions from model for song {song_id}: {e}")
                    traceback.print_exc()
                    errors += 1
                    continue

            # Ensure predictions have same length as reference labels (use min_len)
            min_len = min(len(predictions_idx), len(reference_labels_orig), len(timestamps))
            if min_len == 0:
                 print(f"Warning: Zero length data after prediction/label alignment for song {song_id}. Skipping.")
                 continue

            predictions_idx = predictions_idx[:min_len]
            reference_labels_orig = reference_labels_orig[:min_len]
            timestamps = timestamps[:min_len]

            # Get chord mapping from model or fallback
            idx_to_chord_map = getattr(model, 'idx_to_chord', None)
            if idx_to_chord_map is None:
                # print("Warning: model.idx_to_chord not found. Using default idx2voca_chord().")
                idx_to_chord_map = idx2voca_chord() # Assumes 170 classes

            # Convert prediction indices to chord names
            pred_chords_orig = [idx_to_chord_map.get(int(idx), "N") for idx in predictions_idx]

            # We no longer standardize labels before passing to calculate_chord_scores
            # Let mir_eval handle standardization internally

            # Collect original labels for overall quality analysis
            collected_refs_original.extend(reference_labels_orig)
            collected_preds_original.extend(pred_chords_orig)

            # Debug first few predicted chords to verify format (only for the first song)
            if i == 0 and not hasattr(large_voca_score_calculation, '_first_chords_logged_std'):
                print(f"First 5 original predicted chords: {pred_chords_orig[:5]}")
                print(f"First 5 original reference chords: {reference_labels_orig[:5]}")
                large_voca_score_calculation._first_chords_logged_std = True

            # Calculate scores using the original labels (not pre-standardized)
            # mir_eval.chord.evaluate will handle standardization internally
            scores_tuple = calculate_chord_scores(
                timestamps, frame_duration, reference_labels_orig, pred_chords_orig)

            # Store scores
            for metric_idx, metric_name in enumerate(all_metrics):
                 score_list_dict[metric_name].append(scores_tuple[metric_idx])

            # Store song length for weighted averaging
            song_length = min_len * frame_duration # Use actual number of frames processed
            song_length_list.append(song_length)

            # Debug first few songs
            if i < 5 and not hasattr(large_voca_score_calculation, '_song_scores_logged_std'):
                root_score_val = scores_tuple[all_metrics.index('root')]
                mirex_score_val = scores_tuple[all_metrics.index('mirex')]
                print(f"Song {song_id}: length={song_length:.1f}s, root={root_score_val:.4f}, mirex={mirex_score_val:.4f}")
                if i == 0: large_voca_score_calculation._song_scores_logged_std = True

        except FileNotFoundError as e:
             print(f"Skipping song {song_id} due to missing file: {e}")
             errors += 1
        except RuntimeError as e:
             print(f"Skipping song {song_id} due to runtime error: {e}")
             errors += 1
             if errors <= 10: traceback.print_exc() # Print traceback for first few errors
        except Exception as e:
            print(f"Unexpected error evaluating song group {song_id}: {str(e)}")
            errors += 1
            if errors <= 10: traceback.print_exc() # Print traceback for first few errors


    print(f"Finished evaluation with {errors} errors.")

    # Extra: Debug print to ensure labels were collected
    print(f"\nCollected {len(collected_refs_original)} original reference labels and {len(collected_preds_original)} original predictions for chord quality analysis")

    # Extra: Print individual chord accuracy computed over all processed songs
    if collected_refs_original and collected_preds_original:
        min_len_overall = min(len(collected_refs_original), len(collected_preds_original))
        if min_len_overall > 0:
            ref_sample = collected_refs_original[:min_len_overall]
            pred_sample = collected_preds_original[:min_len_overall]

            print("\n--- Overall Chord Quality Accuracy Analysis ---")
            # This function now works with original labels (not pre-standardized)
            ind_acc, quality_stats = compute_individual_chord_accuracy(ref_sample, pred_sample)

            if not ind_acc:
                print("\nNo individual chord accuracy data computed despite having labels.")
        else:
             print("Warning: Zero length label lists after collection for chord quality analysis.")
    else:
        print("Warning: Insufficient data for chord quality analysis. Need both reference and prediction labels.")

    # Calculate weighted average scores
    average_score_dict = {}
    valid_length = len(song_length_list)

    # Ensure all score lists have the same length as song_length_list
    for metric in all_metrics:
        current_length = len(score_list_dict[metric])
        if current_length != valid_length:
            print(f"WARNING: Length mismatch for {metric} scores ({current_length}) vs song lengths ({valid_length}). Adjusting...")
            # Adjust score list to match valid_length (pad with 0 or truncate)
            score_list_dict[metric] = (score_list_dict[metric] + [0.0] * valid_length)[:valid_length]

    if song_length_list and sum(song_length_list) > 0:
        total_duration = sum(song_length_list)
        weights = np.array(song_length_list) / total_duration
        for metric in all_metrics:
            scores = np.array(score_list_dict[metric])
            # Ensure scores are numeric and handle potential NaNs
            scores = np.nan_to_num(scores.astype(float))
            avg_score = np.sum(scores * weights)
            # Clip score to be within [0, 1]
            average_score_dict[metric] = max(0.0, min(1.0, avg_score))
    else:
        print("Warning: No valid song lengths recorded. Average scores will be 0.")
        for metric in all_metrics:
            average_score_dict[metric] = 0.0

    # Clean up to free memory
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return score_list_dict, song_length_list, average_score_dict


def calculate_chord_scores(timestamps, frame_duration, reference_labels, prediction_labels):
    """
    Calculate various chord evaluation metrics correctly using mir_eval.evaluate.
    Let mir_eval handle standardization internally to avoid double standardization.

    IMPORTANT: Do NOT standardize labels before passing them to this function,
    as mir_eval.chord.evaluate already performs its own standardization internally.

    Args:
        timestamps: Array of frame start timestamps.
        frame_duration: Duration of a single frame.
        reference_labels: List of reference chord labels (not pre-standardized).
        prediction_labels: List of predicted chord labels (not pre-standardized).

    Returns:
        Tuple of evaluation scores (root, thirds, triads, sevenths, tetrads, majmin, mirex)
    """
    # Ensure inputs are lists
    reference_labels = list(reference_labels)
    prediction_labels = list(prediction_labels)

    # Ensure all inputs have the same length
    min_len = min(len(timestamps), len(reference_labels), len(prediction_labels))
    if min_len == 0:
        # print("Warning: Zero length input to calculate_chord_scores. Returning all zeros.")
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    timestamps = np.array(timestamps[:min_len])
    reference_labels = reference_labels[:min_len]
    prediction_labels = prediction_labels[:min_len]

    # Create intervals: [start_time, end_time]
    # Ensure end times don't overlap start times of the next frame
    ref_intervals = np.zeros((min_len, 2))
    ref_intervals[:, 0] = timestamps
    # Calculate end times based on the start of the next frame, or add duration for the last frame
    ref_intervals[:-1, 1] = timestamps[1:]
    ref_intervals[-1, 1] = timestamps[-1] + frame_duration

    # Use the same intervals for estimated chords
    est_intervals = np.copy(ref_intervals)

    # Check for non-increasing intervals which mir_eval dislikes
    invalid_intervals = ref_intervals[:, 0] >= ref_intervals[:, 1]
    if np.any(invalid_intervals):
        print(f"Warning: Found {np.sum(invalid_intervals)} invalid intervals (start >= end). Fixing by adding small duration.")
        ref_intervals[invalid_intervals, 1] = ref_intervals[invalid_intervals, 0] + 1e-6 # Add tiny duration
        est_intervals[invalid_intervals, 1] = est_intervals[invalid_intervals, 0] + 1e-6

    # Check for NaN or infinite values in intervals
    if np.isnan(ref_intervals).any() or np.isinf(ref_intervals).any() or \
       np.isnan(est_intervals).any() or np.isinf(est_intervals).any():
        print("Warning: Intervals contain NaN or infinite values. Attempting to clean.")
        ref_intervals = np.nan_to_num(ref_intervals, nan=0.0, posinf=timestamps[-1]+frame_duration, neginf=0.0)
        est_intervals = np.nan_to_num(est_intervals, nan=0.0, posinf=timestamps[-1]+frame_duration, neginf=0.0)
        # Re-check validity after cleaning
        invalid_intervals = ref_intervals[:, 0] >= ref_intervals[:, 1]
        ref_intervals[invalid_intervals, 1] = ref_intervals[invalid_intervals, 0] + 1e-6
        est_intervals[invalid_intervals, 1] = est_intervals[invalid_intervals, 0] + 1e-6


    # Use mir_eval.chord.evaluate for robust calculation
    scores = {}
    default_scores = (0.0,) * 7 # Tuple of 7 zeros
    try:
        # mir_eval.chord.evaluate handles merging, weighting, and calculates all metrics
        # It requires standardized labels ('N' for no chord)
        scores = mir_eval.chord.evaluate(ref_intervals, reference_labels, est_intervals, prediction_labels)

        # Extract scores safely, defaulting to 0.0 if a metric is missing
        root_score = float(scores.get('root', 0.0))
        thirds_score = float(scores.get('thirds', 0.0))
        triads_score = float(scores.get('triads', 0.0))
        sevenths_score = float(scores.get('sevenths', 0.0))
        tetrads_score = float(scores.get('tetrads', 0.0))
        majmin_score = float(scores.get('majmin', 0.0))
        mirex_score = float(scores.get('mirex', 0.0))

        results = (root_score, thirds_score, triads_score, sevenths_score, tetrads_score, majmin_score, mirex_score)

    except ValueError as ve:
        # Catch specific mir_eval errors if possible
        if "completely empty" in str(ve) or "has zero duration" in str(ve):
             print(f"Warning during mir_eval.chord.evaluate: {ve}. Returning zero scores.")
             return default_scores
        else:
             print(f"ValueError during mir_eval.chord.evaluate: {ve}")
             # print(traceback.format_exc()) # More detailed traceback if needed
             return default_scores
    except Exception as e:
        print(f"Unexpected error during mir_eval.chord.evaluate: {e}")
        # print(traceback.format_exc())
        # print(f"Input shapes - ref_intervals: {ref_intervals.shape}, est_intervals: {est_intervals.shape}")
        # print(f"Reference labels length: {len(reference_labels)}, Prediction labels length: {len(prediction_labels)}")
        # print(f"Sample ref labels: {reference_labels[:10]}")
        # print(f"Sample pred labels: {prediction_labels[:10]}")
        return default_scores

    # Ensure all scores are within the valid range [0, 1]
    clipped_results = tuple(max(0.0, min(1.0, score)) for score in results)

    return clipped_results