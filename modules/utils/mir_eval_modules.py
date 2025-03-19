import soundfile as sf
if not hasattr(sf, 'SoundFileRuntimeError'):
    sf.SoundFileRuntimeError = RuntimeError

import numpy as np
import librosa
import mir_eval
import torch
import os
import audioread

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
    tmp = librosa.cqt(original_wav[currunt_sec_hz:],
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
    feature_per_second = config.feature['hop_length'] / config.mp3['song_hz']
    return feature, feature_per_second, song_length_second

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
            feature, feature_per_second, song_length_second = audio_file_to_features(mp3_file_path, config)
            feature = feature.T
            feature = (feature - mean) / std
            time_unit = feature_per_second

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
                    prediction = model.predict(
                        feature[:, n_timestep * t:n_timestep * (t + 1), :], 
                        per_frame=True
                    ).squeeze()
                    
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

def root_majmin_score_calculation_crf(valid_dataset, config, mean, std, device, pre_model, model, model_type, verbose=False):
    """
    Calculate root and majmin scores for chord recognition using a CRF post-processing model.
    Updated to work with ChordNet architecture with frame-level predictions.
    
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
    pre_model: nn.Module
        Feature extraction model (ChordNet)
    model: nn.Module
        CRF model for sequence modeling
    model_type: str
        Model type identifier (should be 'ChordNet' in most cases)
    verbose: bool
        Whether to print detailed per-song scores
        
    Returns:
    --------
    score_list_dict, song_length_list, average_score_dict: as described in root_majmin_score_calculation
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
            feature, feature_per_second, song_length_second = audio_file_to_features(mp3_file_path, config)
            feature = feature.T
            feature = (feature - mean) / std
            time_unit = feature_per_second

            # Pad features to match the timestep size
            num_pad = n_timestep - (feature.shape[0] % n_timestep)
            feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
            num_instance = feature.shape[0] // n_timestep

            start_time = 0.0
            lines = []
            
            # Generate chord predictions
            with torch.no_grad():
                model.eval()
                pre_model.eval()
                feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)
                
                # Process each segment
                for t in range(num_instance):
                    # First pass through ChordNet to get logits
                    logits, _ = pre_model(feature[:, n_timestep * t:n_timestep * (t + 1), :])
                    
                    # Then apply CRF model for sequence modeling
                    # Use random chord targets for CRF as placeholder (as per original code)
                    rand_targets = torch.randint(config.model.get('num_chords', logits.size(-1)), 
                                                (logits.size(1),)).to(device)
                    prediction, _ = model(logits, rand_targets)
                    
                    # Process each frame in the segment
                    for i in range(n_timestep):
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

def large_voca_score_calculation(valid_dataset, config, model, model_type, mean=None, std=None, device=None, verbose=False):
    """
    Calculate chord recognition scores using mir_eval framework.
    Modified to support both dataset objects and lists of samples.
    
    Parameters:
    -----------
    valid_dataset: Dataset or list
        Dataset with validation samples or a list of sample dictionaries
    config: HParams
        Configuration parameters
    model: nn.Module
        ChordNet model
    model_type: str
        Model type identifier
    mean, std: float
        Normalization parameters
    device: torch.device
        Device for computation
    verbose: bool
        Whether to print detailed per-song scores
        
    Returns:
    --------
    score_list_dict: dict
        Dictionary of score lists for each metric
    song_length_list: list
        List of song lengths
    average_score_dict: dict
        Dictionary of average scores for each metric
    """
    master_mapping = idx2voca_chord()
    metrics_ = metrics()
    song_length_list = list()
    
    # Check if valid_dataset is a list of samples
    if isinstance(valid_dataset, list):
        print(f"Processing list of {len(valid_dataset)} samples for evaluation")
        
        # Group samples by song_id for evaluation
        song_samples = {}
        for idx, sample in enumerate(valid_dataset):
            # Use song_id if available, otherwise create a virtual ID based on position
            song_id = sample.get('song_id', f'song_{idx//100}')
            if song_id not in song_samples:
                song_samples[song_id] = []
            song_samples[song_id].append(sample)
        
        # Process each group of samples as a "song"
        print(f"Grouped into {len(song_samples)} virtual songs for evaluation")
        for song_id, samples in song_samples.items():
            try:
                # Create temporary lab file for ground truth
                pid = os.getpid()
                lab_file_path = f'tmp_gt_{pid}_{song_id}.lab'
                
                # Write chord labels to temporary file
                with open(lab_file_path, 'w') as f:
                    time_per_frame = 0.1  # Assume 10 frames per second
                    for i, sample in enumerate(samples):
                        start_time = i * time_per_frame
                        end_time = (i + 1) * time_per_frame
                        chord_label = sample['chord_label']
                        f.write(f"{start_time:.6f} {end_time:.6f} {chord_label}\n")
                
                # Prepare spectrograms for prediction
                spectros = [torch.tensor(sample['spectro'], dtype=torch.float32) for sample in samples]
                if spectros:
                    feature = torch.stack(spectros, dim=0).unsqueeze(0)  # Add batch dimension
                    
                    # Apply normalization
                    if mean is not None and std is not None:
                        feature = (feature - mean) / std
                    
                    feature = feature.to(device)
                    song_length_second = len(samples) * time_per_frame
                    
                    # Generate predictions
                    with torch.no_grad():
                        model.eval()
                        prediction = model.predict(feature, per_frame=True).squeeze()
                        
                        # Write predictions to temporary file
                        tmp_path = f'tmp_pred_{pid}_{song_id}.lab'
                        with open(tmp_path, 'w') as f:
                            prev_chord = prediction[0].item() if prediction.numel() > 0 else 0
                            start_time = 0.0
                            
                            for i in range(1, len(prediction)):
                                if prediction[i].item() != prev_chord:
                                    f.write(f"{start_time:.6f} {i * time_per_frame:.6f} {master_mapping.get(prev_chord, 'N')}\n")
                                    start_time = i * time_per_frame
                                    prev_chord = prediction[i].item()
                            
                            # Write the final chord segment
                            if start_time < len(prediction) * time_per_frame:
                                f.write(f"{start_time:.6f} {len(prediction) * time_per_frame:.6f} {master_mapping.get(prev_chord, 'N')}\n")
                    
                    # Calculate scores for all metrics
                    for m in metrics_.score_metrics:
                        try:
                            score = metrics_.score(metric=m, gt_path=lab_file_path, est_path=tmp_path)
                            metrics_.score_list_dict[m].append(score)
                            if verbose:
                                print(f'Sample group {song_id}, {m} score: {score:.4f}')
                        except Exception as e:
                            print(f"Error calculating {m} score for {song_id}: {e}")
                    
                    # Clean up temporary files
                    os.remove(tmp_path)
                    os.remove(lab_file_path)
                    
                    song_length_list.append(song_length_second)
            except Exception as e:
                print(f'Error evaluating sample group {song_id}: {str(e)}')
    else:
        # Try to handle dataset objects with expected attributes
        try:
            valid_song_names = valid_dataset.song_names
            paths = valid_dataset.preprocessor.get_all_files()
            
            print(f"Processing {len(valid_song_names)} songs from dataset")
            for path in paths:
                song_name, lab_file_path, mp3_file_path, _ = path
                if song_name not in valid_song_names:
                    continue
                try:
                    # Extract features from audio file
                    n_timestep = config.model['timestep']
                    feature, feature_per_second, song_length_second = audio_file_to_features(mp3_file_path, config)
                    feature = feature.T
                    if mean is not None and std is not None:
                        feature = (feature - mean) / std
                    time_unit = feature_per_second

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
                            # Process with frame-level predictions
                            prediction = model.predict(
                                feature[:, n_timestep * t:n_timestep * (t + 1), :], 
                                per_frame=True
                            ).squeeze()
                            
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
                                            master_mapping.get(prev_chord, "N")
                                        )
                                    )
                                    start_time = time_unit * (n_timestep * t + i)
                                    prev_chord = prediction[i].item()
                                    
                                # Handle the final segment's final frame
                                if t == num_instance - 1 and i + num_pad >= n_timestep:
                                    if start_time < time_unit * (n_timestep * t + i):
                                        lines.append(
                                            '%.6f %.6f %s\n' % (
                                                start_time, 
                                                time_unit * (n_timestep * t + i), 
                                                master_mapping.get(prev_chord, "N")
                                            )
                                        )
                                    break
                    
                    # Write predictions to a temporary file
                    pid = os.getpid()
                    tmp_path = f'tmp_pred_{pid}_{song_name}.lab'
                    with open(tmp_path, 'w') as f:
                        for line in lines:
                            f.write(line)

                    # Calculate scores for all metrics
                    for m in metrics_.score_metrics:
                        metrics_.score_list_dict[m].append(
                            metrics_.score(metric=m, gt_path=lab_file_path, est_path=tmp_path)
                        )
                        
                    # Clean up temporary file
                    os.remove(tmp_path)
                    
                    song_length_list.append(song_length_second)
                    
                    if verbose:
                        for m in metrics_.score_metrics:
                            print(f'Song {song_name}, {m} score: {metrics_.score_list_dict[m][-1]:.4f}')
                        
                except Exception as e:
                    print(f'Song name {song_name} evaluation error: {str(e)}')
        except AttributeError as e:
            print(f"Warning: valid_dataset doesn't have expected attributes: {e}")
            # Return empty results if we can't process the dataset
            return {m: [] for m in metrics_.score_metrics}, [], {m: 0 for m in metrics_.score_metrics}

    # Calculate weighted average scores
    if song_length_list:
        tmp = np.array(song_length_list) / np.sum(song_length_list)
        for m in metrics_.score_metrics:
            if metrics_.score_list_dict[m]:
                metrics_.average_score[m] = np.sum(np.multiply(metrics_.score_list_dict[m], tmp))
            else:
                metrics_.average_score[m] = 0.0
    else:
        for m in metrics_.score_metrics:
            metrics_.average_score[m] = 0.0

    return metrics_.score_list_dict, song_length_list, metrics_.average_score

def large_voca_score_calculation_crf(valid_dataset, config, mean, std, device, pre_model, model, model_type, verbose=False):
    """
    Calculate large vocabulary chord recognition scores using a CRF post-processing model.
    Updated to work with ChordNet architecture with frame-level predictions.
    
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
    pre_model: nn.Module
        Feature extraction model (ChordNet)
    model: nn.Module
        CRF model for sequence modeling
    model_type: str
        Model type identifier (should be 'ChordNet' in most cases)
    verbose: bool
        Whether to print detailed per-song scores
        
    Returns:
    --------
    score_list_dict, song_length_list, average_score_dict: as described in large_voca_score_calculation
    """
    master_mapping = idx2voca_chord()
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
            feature, feature_per_second, song_length_second = audio_file_to_features(mp3_file_path, config)
            feature = feature.T
            feature = (feature - mean) / std
            time_unit = feature_per_second

            # Pad features to match the timestep size
            num_pad = n_timestep - (feature.shape[0] % n_timestep)
            feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
            num_instance = feature.shape[0] // n_timestep

            start_time = 0.0
            lines = []
            
            # Generate chord predictions
            with torch.no_grad():
                model.eval()
                pre_model.eval()
                feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)
                
                # Process each segment
                for t in range(num_instance):
                    # First pass through ChordNet to get logits
                    logits, _ = pre_model(feature[:, n_timestep * t:n_timestep * (t + 1), :])
                    
                    # Then apply CRF model for sequence modeling
                    # Use random chord targets for CRF as placeholder (as per original code)
                    rand_targets = torch.randint(config.model.get('num_chords', logits.size(-1)), 
                                                (logits.size(1),)).to(device)
                    prediction, _ = model(logits, rand_targets)
                    
                    # Process each frame in the segment
                    for i in range(n_timestep):
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
                                    master_mapping.get(prev_chord, "Unknown")
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
                                        master_mapping.get(prev_chord, "Unknown")
                                    )
                                )
                            break
            
            # Write predictions to a temporary file
            pid = os.getpid()
            tmp_path = f'tmp_{pid}.lab'
            with open(tmp_path, 'w') as f:
                for line in lines:
                    f.write(line)

            # Calculate scores for all metrics
            for m in metrics_.score_metrics:
                metrics_.score_list_dict[m].append(
                    metrics_.score(metric=m, gt_path=lab_file_path, est_path=tmp_path)
                )
                
            song_length_list.append(song_length_second)
            
            if verbose:
                for m in metrics_.score_metrics:
                    print('song name %s, %s score : %.4f' % (
                        song_name, m, metrics_.score_list_dict[m][-1])
                    )
                    
        except Exception as e:
            print(f'song name {song_name} lab file error: {str(e)}')

    # Calculate weighted average scores based on song length
    tmp = np.array(song_length_list) / np.sum(song_length_list)
    for m in metrics_.score_metrics:
        metrics_.average_score[m] = np.sum(np.multiply(metrics_.score_list_dict[m], tmp))

    return metrics_.score_list_dict, song_length_list, metrics_.average_score
