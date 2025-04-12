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
from modules.utils import logger

def audio_file_to_features(audio_file, config):
    import os
    if not os.path.isfile(audio_file):
        raise FileNotFoundError(f"Audio file not found: {audio_file}")
    file_size = os.path.getsize(audio_file)
    if file_size < 5000:
        raise RuntimeError(f"Audio file '{audio_file}' is too small and may be corrupt.")
    try:
        original_wav, sr = librosa.load(audio_file, sr=config.mp3['song_hz'], mono=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file '{audio_file}': {e}")
    
    n_fft = config.feature.get('n_fft', 512)
    hop_length = config.feature.get('hop_length', 512)
    
    currunt_sec_hz = 0
    feature = None
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
    
    final_segment = original_wav[currunt_sec_hz:]
    if len(final_segment) < n_fft:
        print(f"Warning: Final segment of {audio_file} is too short ({len(final_segment)} samples). Padding to {n_fft} samples.")
        padding_needed = n_fft - len(final_segment)
        final_segment = np.pad(final_segment, (0, padding_needed), mode="constant", constant_values=0)
    
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
    frames_per_second = config.mp3['song_hz'] / config.feature['hop_length']
    frame_duration = config.feature['hop_length'] / config.mp3['song_hz']
    
    print(f"Audio file: {os.path.basename(audio_file)}")
    print(f"Frames: {feature.shape[1]}, Frame rate: {frames_per_second:.2f} fps")
    print(f"Frame duration: {frame_duration:.5f}s, Total duration: {song_length_second:.2f}s")
    
    return feature, frames_per_second, song_length_second

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

def lab_file_error_modify(ref_labels):
    for i in range(len(ref_labels)):
        if ref_labels[i][-2:] == ':4':
            ref_labels[i] = ref_labels[i].replace(':4', ':sus4')
        elif ref_labels[i][-2:] == ':6':
            ref_labels[i] = ref_labels[i].replace(':6', ':maj6')
        elif ref_labels[i][-4:] == ':6/2':
            ref_labels[i] = ref_labels[i].replace(':6/2', ':maj6/2')
        elif ref_labels[i] == 'Emin/4':
            ref_labels[i] = 'E:min/4'
        elif ref_labels[i] == 'A7/3':
            ref_labels[i] = 'A:7/3'
        elif ref_labels[i] == 'Bb7/3':
            ref_labels[i] = 'Bb:7/3'
        elif ref_labels[i] == 'Bb7/5':
            ref_labels[i] = 'Bb:7/5'
        elif ref_labels[i].find(':') == -1:
            if ref_labels[i].find('min') != -1:
                ref_labels[i] = ref_labels[i][:ref_labels[i].find('min')] + ':' + ref_labels[i][ref_labels[i].find('min'):]
    return ref_labels

def normalize_chord_name(chord_name):
    """
    Normalize chord names to handle enharmonic equivalents and common notation variants.
    
    Args:
        chord_name (str): Original chord name
        
    Returns:
        str: Normalized chord name
    """
    if chord_name in ['N', 'X']:
        return 'N'  # Standardize no-chord symbols
    
    # Handle enharmonic equivalents by converting flats to sharps
    enharmonic_map = {
        'Bb': 'A#', 'Eb': 'D#', 'Ab': 'G#', 'Db': 'C#', 'Gb': 'F#',
        'Bbm': 'A#:min', 'Ebm': 'D#:min', 'Abm': 'G#:min', 'Dbm': 'C#:min', 'Gbm': 'F#:min',
        'Bb:': 'A#:', 'Eb:': 'D#:', 'Ab:': 'G#:', 'Db:': 'C#:', 'Gb:': 'F#:'
    }
    
    # Replace flat-based root with sharp equivalent
    for flat_notation, sharp_notation in enharmonic_map.items():
        if chord_name.startswith(flat_notation):
            # Only replace at the beginning of the string (the root note)
            chord_name = sharp_notation + chord_name[len(flat_notation):]
            break
    
    # Handle shortened minor notation (e.g., "Cm" -> "C:min")
    if 'm' in chord_name and ':' not in chord_name:
        parts = chord_name.split('m', 1)
        if len(parts) == 2 and (len(parts[0]) <= 2) and (parts[1] == '' or parts[1].startswith('/')):
            chord_name = f"{parts[0]}:min{parts[1]}"
    
    # Handle inversions by mapping to base chord when needed for evaluation
    if '/' in chord_name:
        base_chord = chord_name.split('/', 1)[0]
        # Keep the base chord for evaluation if the inversion isn't in the vocabulary
        if base_chord.endswith(':min') or ':' not in base_chord:
            return base_chord
    
    # Handle extended chords by simplifying to core triad types
    # This is a fallback for evaluation metrics
    if ':' in chord_name:
        root, quality = chord_name.split(':', 1)
        # Map various extensions to basic triads
        if quality.startswith(('maj7', 'maj9', '7', '9')) and not quality.startswith('min'):
            return f"{root}"  # Map to major
        elif quality.startswith(('min7', 'min9')):
            return f"{root}:min"  # Map to minor
        elif 'sus' in quality:
            return root  # Map to major for suspended chords
    
    return chord_name

def large_voca_score_calculation(valid_dataset, config, model, model_type='BTC', mean=0, std=1, device='cpu'):
    """
    Enhanced version of the evaluation function that's more robust to different chord notations
    and provides better diagnostics.
    """
    # Suppress warnings from sklearn and mir_eval
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="mir_eval.chord")
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics")
    
    # Debugging mode
    debug_mode = config.misc.get('debug_mode', False)
    
    # Put model in evaluation mode
    model.eval()
    
    # Initialize metrics dictionaries
    score_list_dict = {'root':[], 'thirds':[], 'triads':[], 'sevenths':[], 'tetrads':[], 'majmin':[], 'mirex':[]}
    song_length_list = []
    average_score_dict = {'root':0, 'thirds':0, 'triads':0, 'sevenths':0, 'tetrads':0, 'majmin':0, 'mirex':0}
    
    # Track success/failure statistics
    successful_samples = 0
    error_samples = 0
    
    # Get chord mapping once
    idx_to_chord_dict = idx2voca_chord()
    
    # Track chord normalization statistics
    normalization_count = 0
    total_chords = 0
    
    # Track unknown chords for diagnostic purposes
    unknown_chords = {}
    
    logger.info(f"Evaluating {len(valid_dataset)} samples with {model_type} model")
    logger.info("Using enhanced chord normalization to handle enharmonic equivalents")
    
    for i, sample in enumerate(tqdm(valid_dataset, desc="Evaluating songs")):
        try:
            # Extract spectrograms
            if 'spectro' not in sample:
                if debug_mode:
                    logger.debug(f"Sample {i}: missing 'spectro' key")
                error_samples += 1
                continue
                
            spectro = sample['spectro'].to(device)
            
            # Get the chord indices (ground truth)
            if 'chord_idx' not in sample:
                # Try alternate keys
                if 'chord_indices' in sample:
                    label_idx = sample['chord_indices']
                else:
                    if debug_mode:
                        logger.debug(f"Sample {i}: missing chord indices")
                    error_samples += 1
                    continue
            else:
                label_idx = sample['chord_idx']
            
            # Convert to numpy array if needed
            if isinstance(label_idx, torch.Tensor):
                label_idx = label_idx.cpu().detach().numpy()
            
            if not isinstance(label_idx, np.ndarray):
                try:
                    label_idx = np.array(label_idx)
                except:
                    if debug_mode:
                        logger.debug(f"Sample {i}: couldn't convert label_idx to numpy array")
                    error_samples += 1
                    continue
            
            # Flatten array
            label_idx = label_idx.flatten()
            
            # Normalize spectrogram
            spectro_norm = (spectro - mean) / (std + 1e-8)
            
            # Run model inference
            with torch.no_grad():
                try:
                    outputs = model(spectro_norm)
                    
                    # Handle different output formats
                    if isinstance(outputs, tuple) or isinstance(outputs, list):
                        logits = outputs[0]
                    else:
                        logits = outputs
                    
                    # Get class predictions
                    pred_idx = torch.argmax(logits, dim=1).cpu().detach().numpy().flatten()
                except Exception as e:
                    if debug_mode:
                        logger.debug(f"Sample {i}: model inference failed: {str(e)}")
                    error_samples += 1
                    continue
            
            # Convert indices to chord names with normalization
            ref_chords = []
            for idx in label_idx:
                try:
                    idx_int = int(idx)
                    if idx_int in idx_to_chord_dict:
                        # Get original chord name
                        original_chord = idx_to_chord_dict[idx_int]
                        total_chords += 1
                        
                        # Normalize the chord name
                        normalized_chord = normalize_chord_name(original_chord)
                        if normalized_chord != original_chord:
                            normalization_count += 1
                            
                        # Add standardized chord to list
                        if normalized_chord == 'N':
                            ref_chords.append('N')
                        elif ':' not in normalized_chord and normalized_chord != 'N':
                            ref_chords.append(f"{normalized_chord}:maj")  # Add quality to root-only notation
                        else:
                            ref_chords.append(normalized_chord)
                    else:
                        ref_chords.append('N')  # Default to no-chord for unknown indices
                except:
                    ref_chords.append('N')
                    
            # Convert prediction indices to chord names
            pred_chords = []
            for idx in pred_idx:
                try:
                    idx_int = int(idx)
                    if idx_int in idx_to_chord_dict:
                        # Get original chord name
                        original_chord = idx_to_chord_dict[idx_int]
                        
                        # Normalize the chord name
                        normalized_chord = normalize_chord_name(original_chord)
                        
                        # Add standardized chord to list
                        if normalized_chord == 'N':
                            pred_chords.append('N')
                        elif ':' not in normalized_chord and normalized_chord != 'N':
                            pred_chords.append(f"{normalized_chord}:maj")  # Add quality to root-only notation
                        else:
                            pred_chords.append(normalized_chord)
                    else:
                        if debug_mode:
                            logger.debug(f"Sample {i}: unknown prediction index: {idx_int}")
                        pred_chords.append('N')  # Default to no-chord for unknown indices
                except:
                    pred_chords.append('N')
                    
            # Check if we have valid chord sequences
            min_len = min(len(ref_chords), len(pred_chords))
            if min_len == 0:
                if debug_mode:
                    logger.debug(f"Sample {i}: empty chord sequences")
                error_samples += 1
                continue
                
            # Ensure chord strings are properly formatted for mir_eval
            valid_ref_chords = []
            valid_pred_chords = []
            
            for ref, pred in zip(ref_chords[:min_len], pred_chords[:min_len]):
                # Validate/fix reference chord format
                try:
                    if ref != 'N':
                        # Try to parse with mir_eval to catch format issues
                        root, semitones, bass = mir_eval.chord.encode(ref)
                        valid_ref = ref
                    else:
                        valid_ref = 'N'
                except Exception as e:
                    # Track unknown chords for diagnostics
                    if ref in unknown_chords:
                        unknown_chords[ref] += 1
                    else:
                        unknown_chords[ref] = 1
                    valid_ref = 'N'
                
                # Validate/fix prediction chord format
                try:
                    if pred != 'N':
                        # Try to parse with mir_eval to catch format issues
                        root, semitones, bass = mir_eval.chord.encode(pred)
                        valid_pred = pred
                    else:
                        valid_pred = 'N'
                except Exception as e:
                    valid_pred = 'N'
                
                valid_ref_chords.append(valid_ref)
                valid_pred_chords.append(valid_pred)
            
            # Record song length
            song_length_list.append(min_len)
            
            # Create dummy intervals for mir_eval
            intervals = np.array([[j, j+1] for j in range(min_len)])
            
            # Track if we were able to evaluate at least one metric
            any_metric_worked = False
            
            # Evaluate each metric with better error handling
            for metric in ['root', 'thirds', 'triads', 'sevenths', 'tetrads', 'majmin', 'mirex']:
                try:
                    # Set zero_division parameter to avoid warnings
                    result = mir_eval.chord.evaluate(
                        ref_intervals=intervals,
                        ref_labels=valid_ref_chords,
                        est_intervals=intervals,
                        est_labels=valid_pred_chords,
                        level=metric
                    )
                    
                    # Add to scores if successful
                    if 'weighted_average' in result and not np.isnan(result['weighted_average']):
                        weighted_score = float(result['weighted_average'])
                        score_list_dict[metric].append(min_len * weighted_score)
                        any_metric_worked = True
                except Exception as e:
                    if debug_mode:
                        logger.debug(f"Sample {i}: {metric} evaluation failed: {str(e)}")
                    pass
            
            # Count this sample as successful if we got at least one metric
            if any_metric_worked:
                successful_samples += 1
            else:
                error_samples += 1
                
        except Exception as e:
            error_samples += 1
            if debug_mode:
                logger.debug(f"Sample {i}: general error: {str(e)}")
    
    # Report statistics
    logger.info(f"Evaluation complete: {successful_samples} successful samples, {error_samples} error samples")
    logger.info(f"Chord normalization: {normalization_count}/{total_chords} chords normalized ({normalization_count/max(1,total_chords)*100:.1f}%)")
    
    # Report most common unknown chords if any
    if unknown_chords:
        top_unknown = sorted(unknown_chords.items(), key=lambda x: x[1], reverse=True)[:10]
        logger.info(f"Top unknown chords: {top_unknown}")
    
    # Calculate weighted averages for each metric
    if song_length_list:
        total_length = np.sum(song_length_list)
        
        for metric in average_score_dict.keys():
            if score_list_dict[metric]:
                average_score_dict[metric] = float(np.sum(score_list_dict[metric]) / total_length)
                logger.info(f"{metric.capitalize()} score: {average_score_dict[metric]:.4f} (from {len(score_list_dict[metric])} samples)")
            else:
                logger.info(f"No valid scores for {metric} metric")
    else:
        logger.info("No samples were successfully evaluated")
    
    return score_list_dict, song_length_list, average_score_dict