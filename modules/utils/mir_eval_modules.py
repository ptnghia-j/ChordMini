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
    errors = 0
    
    # Process each song group
    for i, (song_id, samples) in enumerate(tqdm(song_groups.items(), 
                                             desc="Evaluating songs", 
                                             total=len(song_groups))):
        try:
            # Extract and sort samples by frame index if available
            if all('frame_idx' in sample for sample in samples):
                samples.sort(key=lambda x: x['frame_idx'])
            
            # Extract timestamps
            frame_duration = config.feature.get('hop_duration', 0.1)
            timestamps = np.arange(len(samples)) * frame_duration
            
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
                    output = model.predict(input_tensor)
                
                # Move predictions to CPU for further processing
                predictions = output.cpu().numpy()
                
                # Ensure predictions have same length as reference labels
                min_len = min(len(predictions), len(reference_labels))
                predictions = predictions[:min_len]
                reference_labels = reference_labels[:min_len]
                timestamps = timestamps[:min_len]
                
                # Prepare for mir_eval format
                idx_to_chord = model.idx_to_chord if hasattr(model, 'idx_to_chord') else None
                if idx_to_chord is None:
                    # Try to infer from dataset or fall back to default
                    idx_to_chord = {i: f"class_{i}" for i in range(max(predictions) + 1)}
                
                # Convert indices to chord names
                pred_chords = [idx_to_chord.get(pred, "N") for pred in predictions]
                
                # Calculate scores
                durations = np.diff(np.append(timestamps, [timestamps[-1] + frame_duration]))
                root_score, thirds_score, triads_score, sevenths_score, tetrads_score, majmin_score, mirex_score = calculate_mir_scores(
                    timestamps, pred_chords, reference_labels, durations)
                
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
                
                # Debug first few songs
                if i < 5:
                    print(f"Song {song_id}: length={song_length:.1f}s, root={root_score:.4f}, mirex={mirex_score:.4f}")
            
        except Exception as e:
            import traceback
            errors += 1
            print(f"Error evaluating sample group {song_id}: {str(e)}")
            if errors <= 10:  # Only print detailed error for first 10 errors
                traceback.print_exc()
    
    # Calculate weighted average scores
    average_score_dict = {}
    if song_length_list:
        for metric in score_list_dict:
            if score_list_dict[metric]:
                weighted_sum = sum(score * length for score, length in zip(score_list_dict[metric], song_length_list))
                total_length = sum(song_length_list)
                average_score_dict[metric] = weighted_sum / total_length if total_length > 0 else 0.0
            else:
                average_score_dict[metric] = 0.0
    else:
        for metric in score_list_dict:
            average_score_dict[metric] = 0.0
    
    # Clean up to free memory
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return score_list_dict, song_length_list, average_score_dict


def calculate_mir_scores(timestamps, pred_chords, ref_chords, durations):
    """
    Calculate MIR evaluation scores.
    
    Args:
        timestamps: Array of timestamps
        pred_chords: List of predicted chord labels
        ref_chords: List of reference chord labels
        durations: Array of durations for each frame
    
    Returns:
        Various MIR evaluation scores
    """
    import mir_eval
    import numpy as np
    
    # Prepare interval boundaries
    boundaries = np.append([0], np.cumsum(durations))
    
    # Convert to mir_eval's expected format
    intervals = np.array([boundaries[:-1], boundaries[1:]]).T
    
    # Calculate scores
    try:
        root_score = mir_eval.chord.root(intervals, ref_chords, intervals, pred_chords)
        thirds_score = mir_eval.chord.thirds(intervals, ref_chords, intervals, pred_chords)
        triads_score = mir_eval.chord.triads(intervals, ref_chords, intervals, pred_chords)
        sevenths_score = mir_eval.chord.sevenths(intervals, ref_chords, intervals, pred_chords)
        tetrads_score = mir_eval.chord.tetrads(intervals, ref_chords, intervals, pred_chords)
        majmin_score = mir_eval.chord.majmin(intervals, ref_chords, intervals, pred_chords)
        mirex_score = mir_eval.chord.mirex(intervals, ref_chords, intervals, pred_chords)
    except Exception as e:
        # Handle potential errors in mir_eval
        print(f"Error in mir_eval scoring: {e}")
        # Return zeros for all scores on error
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    return root_score, thirds_score, triads_score, sevenths_score, tetrads_score, majmin_score, mirex_score

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
