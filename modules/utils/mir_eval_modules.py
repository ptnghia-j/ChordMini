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

def large_voca_score_calculation(valid_dataset, config, model, model_type='BTC', mean=0, std=1, device='cpu'):
    model.eval()
    score_list_dict = {'root':[], 'thirds':[], 'triads':[], 'sevenths':[], 'tetrads':[], 'majmin':[], 'mirex':[]}
    song_length_list = []
    average_score_dict = {'root':0, 'thirds':0, 'triads':0, 'sevenths':0, 'tetrads':0, 'majmin':0, 'mirex':0}
    
    # Count successfully evaluated samples
    successful_samples = 0
    error_samples = 0
    
    # Set filter level for warnings
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="mir_eval.chord")
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics")
    
    # Ensure logger is available
    try:
        from modules.utils import logger
    except ImportError:
        import logging
        logger = logging.getLogger("ChordMini")
    
    logger.info(f"Evaluating {len(valid_dataset)} samples with {model_type} model")
    
    for i, sample in enumerate(tqdm(valid_dataset, desc="Evaluating songs")):
        try:
            # Extract spectrograms and convert to device
            spectro = sample['spectro'].to(device)
            
            # Get the chord indices from the sample and ensure they're in the right format
            if 'chord_idx' not in sample:
                logger.debug(f"Sample {i}: 'chord_idx' not found in sample")
                error_samples += 1
                continue
                
            label_idx = sample['chord_idx']
            
            # Convert to numpy array if it's a tensor
            if isinstance(label_idx, torch.Tensor):
                label_idx = label_idx.cpu().detach().numpy()
            
            # Ensure we have a NumPy array to work with
            if not isinstance(label_idx, np.ndarray):
                try:
                    label_idx = np.array(label_idx)
                except Exception as e:
                    logger.debug(f"Sample {i}: couldn't convert label_idx to NumPy array - {e}")
                    error_samples += 1
                    continue
            
            # Ensure it's a 1D array
            label_idx = np.atleast_1d(label_idx.flatten())
            
            # Normalize spectrogram
            spectro = (spectro - mean) / (std + 1e-8)
            
            # Model inference with error handling
            try:
                with torch.no_grad():
                    model_output = model(spectro)
                    
                    # Handle different model output formats
                    if isinstance(model_output, tuple):
                        pred_idx = torch.argmax(model_output[0], dim=1).cpu().detach().numpy()
                    else:
                        pred_idx = torch.argmax(model_output, dim=1).cpu().detach().numpy()
                    
                    # Ensure prediction is a flattened 1D array
                    pred_idx = pred_idx.flatten()
            except Exception as e:
                logger.debug(f"Sample {i}: error during model inference - {e}")
                error_samples += 1
                continue
            
            # Get the chord mapping dictionary
            idx_to_chord = idx2voca_chord()
            
            # Convert prediction indices to chord labels
            pred_chords = []
            for idx in pred_idx:
                try:
                    # Convert to Python scalar safely
                    if isinstance(idx, np.ndarray):
                        if idx.size == 1:
                            idx = idx.item()
                        else:
                            idx = int(idx[0])
                    else:
                        idx = int(idx)
                        
                    # Look up chord name
                    if idx in idx_to_chord:
                        chord_name = idx_to_chord[idx]
                        # Check for mir_eval compatibility and make necessary format adjustments
                        if chord_name in ['X', 'N']:
                            chord_name = 'N'  # Standardize no-chord symbols to "N"
                        # Ensure proper formatting for mir_eval
                        if ':' not in chord_name and chord_name != 'N':
                            chord_name = f"{chord_name}:maj"  # Add ':maj' for root-only chord names
                        pred_chords.append(chord_name)
                    else:
                        pred_chords.append('N')  # Default to no-chord
                except Exception as e:
                    logger.debug(f"Sample {i}: error processing prediction index {idx} - {e}")
                    pred_chords.append('N')  # Default to no-chord on error
            
            # Convert reference indices to chord labels
            ref_chords = []
            for idx in label_idx:
                try:
                    # Convert to Python scalar safely
                    if isinstance(idx, np.ndarray):
                        if idx.size == 1:
                            idx = idx.item()
                        else:
                            idx = int(idx[0])
                    else:
                        idx = int(idx)
                        
                    # Look up chord name
                    if idx in idx_to_chord:
                        chord_name = idx_to_chord[idx]
                        # Check for mir_eval compatibility and make necessary format adjustments
                        if chord_name in ['X', 'N']:
                            chord_name = 'N'  # Standardize no-chord symbols to "N"
                        # Ensure proper formatting for mir_eval
                        if ':' not in chord_name and chord_name != 'N':
                            chord_name = f"{chord_name}:maj"  # Add ':maj' for root-only chord names
                        ref_chords.append(chord_name)
                    else:
                        ref_chords.append('N')  # Default to no-chord
                except Exception as e:
                    logger.debug(f"Sample {i}: error processing reference index {idx} - {e}")
                    ref_chords.append('N')  # Default to no-chord on error
            
            # Check if we have valid chord sequences to compare
            if not pred_chords or not ref_chords:
                logger.debug(f"Sample {i}: empty chord sequences")
                error_samples += 1
                continue
                
            # Find the minimum length to compare
            min_len = min(len(pred_chords), len(ref_chords))
            if min_len == 0:
                logger.debug(f"Sample {i}: zero-length chord sequences")
                error_samples += 1
                continue
                
            # Record song length for weighted averaging
            song_length = min_len
            song_length_list.append(song_length)
            
            # Create time intervals for each frame (needed by mir_eval)
            dummy_intervals = np.array([[j, j+1] for j in range(min_len)])
            
            # Flag to track if at least one metric was calculated successfully
            any_metric_succeeded = False
            
            # Sanitize chord labels to ensure mir_eval compatibility
            clean_ref_chords = []
            clean_pred_chords = []
            
            for ref, pred in zip(ref_chords[:min_len], pred_chords[:min_len]):
                # Ensure all chords have proper format for mir_eval
                try:
                    # Try to normalize the format using mir_eval's parser
                    if ref != 'N':
                        mir_eval.chord.tokenize(ref)  # This will raise an error if format is invalid
                    clean_ref = ref
                except Exception as e:
                    clean_ref = 'N'  # Default to no-chord on error
                    
                try:
                    if pred != 'N':
                        mir_eval.chord.tokenize(pred)  # This will raise an error if format is invalid
                    clean_pred = pred
                except Exception as e:
                    clean_pred = 'N'  # Default to no-chord on error
                    
                clean_ref_chords.append(clean_ref)
                clean_pred_chords.append(clean_pred)
            
            # Root note recognition
            try:
                root_result = mir_eval.chord.evaluate(
                    ref_intervals=dummy_intervals,
                    ref_labels=clean_ref_chords,
                    est_intervals=dummy_intervals,
                    est_labels=clean_pred_chords,
                    level='root'
                )
                if 'weighted_average' in root_result:
                    score_list_dict['root'].append(song_length * root_result['weighted_average'])
                    any_metric_succeeded = True
            except Exception as e:
                logger.debug(f"Sample {i}: error calculating root score - {e}")
            
            # Thirds recognition (major/minor)
            try:
                thirds_result = mir_eval.chord.evaluate(
                    ref_intervals=dummy_intervals,
                    ref_labels=clean_ref_chords,
                    est_intervals=dummy_intervals,
                    est_labels=clean_pred_chords,
                    level='thirds'
                )
                if 'weighted_average' in thirds_result:
                    score_list_dict['thirds'].append(song_length * thirds_result['weighted_average'])
                    any_metric_succeeded = True
            except Exception as e:
                logger.debug(f"Sample {i}: error calculating thirds score - {e}")
            
            # Triads recognition
            try:
                triads_result = mir_eval.chord.evaluate(
                    ref_intervals=dummy_intervals,
                    ref_labels=clean_ref_chords,
                    est_intervals=dummy_intervals,
                    est_labels=clean_pred_chords,
                    level='triads'
                )
                if 'weighted_average' in triads_result:
                    score_list_dict['triads'].append(song_length * triads_result['weighted_average'])
                    any_metric_succeeded = True
            except Exception as e:
                logger.debug(f"Sample {i}: error calculating triads score - {e}")
            
            # Sevenths recognition
            try:
                sevenths_result = mir_eval.chord.evaluate(
                    ref_intervals=dummy_intervals,
                    ref_labels=clean_ref_chords,
                    est_intervals=dummy_intervals,
                    est_labels=clean_pred_chords,
                    level='sevenths'
                )
                if 'weighted_average' in sevenths_result:
                    score_list_dict['sevenths'].append(song_length * sevenths_result['weighted_average'])
                    any_metric_succeeded = True
            except Exception as e:
                logger.debug(f"Sample {i}: error calculating sevenths score - {e}")
            
            # Tetrads recognition
            try:
                tetrads_result = mir_eval.chord.evaluate(
                    ref_intervals=dummy_intervals,
                    ref_labels=clean_ref_chords,
                    est_intervals=dummy_intervals,
                    est_labels=clean_pred_chords,
                    level='tetrads'
                )
                if 'weighted_average' in tetrads_result:
                    score_list_dict['tetrads'].append(song_length * tetrads_result['weighted_average'])
                    any_metric_succeeded = True
            except Exception as e:
                logger.debug(f"Sample {i}: error calculating tetrads score - {e}")
            
            # Major/minor recognition
            try:
                majmin_result = mir_eval.chord.evaluate(
                    ref_intervals=dummy_intervals,
                    ref_labels=clean_ref_chords,
                    est_intervals=dummy_intervals,
                    est_labels=clean_pred_chords,
                    level='majmin'
                )
                if 'weighted_average' in majmin_result:
                    score_list_dict['majmin'].append(song_length * majmin_result['weighted_average'])
                    any_metric_succeeded = True
            except Exception as e:
                logger.debug(f"Sample {i}: error calculating majmin score - {e}")
            
            # MIREX recognition
            try:
                mirex_result = mir_eval.chord.evaluate(
                    ref_intervals=dummy_intervals,
                    ref_labels=clean_ref_chords,
                    est_intervals=dummy_intervals,
                    est_labels=clean_pred_chords,
                    level='mirex'
                )
                if 'weighted_average' in mirex_result:
                    score_list_dict['mirex'].append(song_length * mirex_result['weighted_average'])
                    any_metric_succeeded = True
            except Exception as e:
                logger.debug(f"Sample {i}: error calculating mirex score - {e}")
            
            # Count this sample as successful if any metric was calculated
            if any_metric_succeeded:
                successful_samples += 1
            else:
                error_samples += 1
                
        except Exception as e:
            logger.debug(f"Sample {i}: general processing error - {e}")
            error_samples += 1
    
    # Report statistics
    logger.info(f"Evaluation complete: {successful_samples} successful samples, {error_samples} error samples")
    
    # Check if we have valid results
    if not song_length_list:
        logger.warning("No valid samples processed, cannot calculate averages")
        return score_list_dict, song_length_list, average_score_dict
    
    # Calculate total song length for weighting
    total_length = np.sum(song_length_list)
    if total_length <= 0:
        logger.warning("Total song length is zero, cannot calculate weighted averages")
        return score_list_dict, song_length_list, average_score_dict
    
    # Calculate weighted average for each metric
    for key in average_score_dict.keys():
        if score_list_dict[key]:
            average_score_dict[key] = float(np.sum(score_list_dict[key]) / total_length)
            logger.info(f"{key.capitalize()} score: {average_score_dict[key]:.4f} (from {len(score_list_dict[key])} valid samples)")
        else:
            average_score_dict[key] = 0.0
            logger.warning(f"No valid scores for {key} metric")
    
    return score_list_dict, song_length_list, average_score_dict