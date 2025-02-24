import os
import time
import numpy as np
import torch
import madmom
import librosa
from torch.utils.data import Dataset
from scipy.ndimage import maximum_filter1d
from tqdm import tqdm
from matplotlib import pyplot as plt
import librosa.display
from scipy.interpolate import interp1d
from scipy.signal import argrelmax

class AudioProcessingDataset(Dataset):
    """
    Dataset for processing audio signals along with beat, downbeat, and tempo annotations.
    Supports train, validation, and test modes.
    """
    def __init__(self, full_data, full_annotation, audio_files=None,
                 mode='train', fold=0, fps=44100/1024, sample_size=512,
                 num_folds=8, mask_value=-1, test_only=None):
        # ...existing code...
        if test_only is None:
            test_only = []
        self.fold = fold
        self.num_folds = num_folds
        self.fps = fps
        self.mode = mode
        self.sample_size = sample_size
        self.MASK_VALUE = mask_value

        self.data = []
        self.beats = []
        self.downbeats = []
        self.tempi = []
        self.root = []
        self.dataset_name = []

        if self.mode == 'train':
            self._train_clip(full_data, full_annotation, test_only=test_only)
        elif self.mode in ['validation', 'test']:
            self.audio_files = []
            self._val_test_clip(full_data, full_annotation, audio_files, test_only=test_only)
        full_data = None
        full_annotation = None

    def _train_clip(self, full_data, full_annotation, num_tempo_bins=300, test_only=None):
        # ...existing code...
        if test_only is None:
            test_only = []
        for fold_idx in tqdm(range(self.num_folds)):
            if fold_idx != self.fold and fold_idx != (self.fold+1) % self.num_folds:
                for key in full_data:
                    if key in test_only:
                        continue
                    for song_idx in range(len(full_data[key][fold_idx])):
                        song = full_data[key][fold_idx][song_idx]  # (T, channels, mel)
                        annotation = full_annotation[key][fold_idx][song_idx]
                        try:
                            if len(annotation.shape) == 2:
                                beat = madmom.utils.quantize_events(annotation[:, 0], fps=self.fps, length=len(song))
                            else:
                                beat = madmom.utils.quantize_events(annotation[:], fps=self.fps, length=len(song))
                            beat = np.maximum(beat, maximum_filter1d(beat, size=3) * 0.5)
                            beat = np.maximum(beat, maximum_filter1d(beat, size=3) * 0.5)
                        except:
                            beat = np.ones(len(song), dtype='float32') * self.MASK_VALUE
                        try:
                            downbeat = annotation[annotation[:, 1] == 1][:, 0]
                            downbeat = madmom.utils.quantize_events(downbeat, fps=self.fps, length=len(song))
                            downbeat = np.maximum(downbeat, maximum_filter1d(downbeat, size=3) * 0.5)
                            downbeat = np.maximum(downbeat, maximum_filter1d(downbeat, size=3) * 0.5)
                        except:
                            downbeat = np.ones(len(song), dtype='float32') * self.MASK_VALUE
                        try:
                            tempo = np.zeros(num_tempo_bins, dtype='float32')
                            if len(annotation.shape) == 2:
                                tempo_idx = int(np.round(self._infer_tempo(annotation[:, 0])))
                            else:
                                tempo_idx = int(np.round(self._infer_tempo(annotation[:])))
                            tempo[tempo_idx] = 1
                            tempo = np.maximum(tempo, maximum_filter1d(tempo, size=3) * 0.5)
                            tempo = np.maximum(tempo, maximum_filter1d(tempo, size=3) * 0.5)
                            tempo = tempo / np.sum(tempo) if np.sum(tempo) > 0 else tempo
                        except:
                            tempo = np.ones(num_tempo_bins, dtype='float32') * self.MASK_VALUE

                        if self.sample_size is None or len(song) <= self.sample_size:
                            self.dataset_name.append(key)
                            self.data.append(song)
                            self.beats.append(beat)
                            self.downbeats.append(downbeat)
                            self.tempi.append(tempo)
                        else:
                            for i in range(0, len(song) - self.sample_size + 1, self.sample_size):
                                self.dataset_name.append(key)
                                self.data.append(song[i: i+self.sample_size])
                                self.beats.append(beat[i: i+self.sample_size])
                                self.downbeats.append(downbeat[i: i+self.sample_size])
                                self.tempi.append(tempo)
                            if i + self.sample_size < len(song):
                                self.dataset_name.append(key)
                                self.data.append(song[len(song)-self.sample_size:])
                                self.beats.append(beat[len(song)-self.sample_size:])
                                self.downbeats.append(downbeat[len(song)-self.sample_size:])
                                self.tempi.append(tempo)

    def _val_test_clip(self, full_data, full_annotation, audio_files, num_tempo_bins=300, test_only=None):
        # ...existing code...
        if test_only is None:
            test_only = []
        fold_idx = (self.fold+1) % self.num_folds if self.mode == 'validation' else self.fold
        for key in tqdm(full_data, total=len(full_data)):
            if self.mode == 'validation' and key in test_only:
                continue
            for song_idx in range(len(full_data[key][fold_idx])):
                song = full_data[key][fold_idx][song_idx]
                annotation = full_annotation[key][fold_idx][song_idx]
                audio_file = audio_files[key][fold_idx][song_idx]
                try:
                    if len(annotation.shape) == 2:
                        beat = madmom.utils.quantize_events(annotation[:, 0], fps=self.fps, length=len(song))
                    else:
                        beat = madmom.utils.quantize_events(annotation[:], fps=self.fps, length=len(song))
                    beat = np.maximum(beat, maximum_filter1d(beat, size=3) * 0.5)
                    beat = np.maximum(beat, maximum_filter1d(beat, size=3) * 0.5)
                except:
                    beat = np.ones(len(song), dtype='float32') * self.MASK_VALUE
                try:
                    downbeat = annotation[annotation[:, 1] == 1][:, 0]
                    downbeat = madmom.utils.quantize_events(downbeat, fps=self.fps, length=len(song))
                    downbeat = np.maximum(downbeat, maximum_filter1d(downbeat, size=3) * 0.5)
                    downbeat = np.maximum(downbeat, maximum_filter1d(downbeat, size=3) * 0.5)
                except:
                    downbeat = np.ones(len(song), dtype='float32') * self.MASK_VALUE
                try:
                    tempo = np.zeros(num_tempo_bins, dtype='float32')
                    if len(annotation.shape) == 2:
                        tempo_idx = int(np.round(self._infer_tempo(annotation[:, 0])))
                    else:
                        tempo_idx = int(np.round(self._infer_tempo(annotation[:])))
                    tempo[tempo_idx] = 1
                    tempo = np.maximum(tempo, maximum_filter1d(tempo, size=3) * 0.5)
                    tempo = np.maximum(tempo, maximum_filter1d(tempo, size=3) * 0.5)
                    tempo = tempo / np.sum(tempo) if np.sum(tempo) > 0 else tempo
                except:
                    tempo = np.ones(num_tempo_bins, dtype='float32') * self.MASK_VALUE
                if self.sample_size is None or len(song) <= int(44100/1024 * 420):
                    self.dataset_name.append(key)
                    self.root.append(audio_file)
                    self.data.append(song)
                    self.beats.append(beat)
                    self.downbeats.append(downbeat)
                    self.tempi.append(tempo)
                else:
                    eval_sample_size = int(44100/1024 * 420)
                    for i in range(0, len(song) - eval_sample_size + 1, eval_sample_size):
                        self.dataset_name.append(key)
                        self.root.append(audio_file)
                        self.data.append(song[i: i+eval_sample_size])
                        self.beats.append(beat[i: i+eval_sample_size])
                        self.downbeats.append(downbeat[i: i+eval_sample_size])
                        self.tempi.append(tempo)
                    if i + eval_sample_size < len(song):
                        self.dataset_name.append(key)
                        self.root.append(audio_file)
                        self.data.append(song[len(song)-eval_sample_size:])
                        self.beats.append(beat[len(song)-eval_sample_size:])
                        self.downbeats.append(downbeat[len(song)-eval_sample_size:])
                        self.tempi.append(tempo)

    def _infer_tempo(self, beats, hist_smooth=4, no_tempo=-1):
        # ...existing code...
        ibis = np.diff(beats) * self.fps
        bins = np.bincount(np.round(ibis).astype(int))
        if not bins.any():
            return no_tempo
        if hist_smooth > 0:
            bins = madmom.audio.signal.smooth(bins, hist_smooth)
        intervals = np.arange(len(bins))
        interpolation_fn = interp1d(intervals, bins, kind='quadratic')
        intervals = np.arange(intervals[0], intervals[-1], 0.001)
        tempi = 60.0 * self.fps / intervals
        bins = interpolation_fn(intervals)
        peaks = argrelmax(bins, mode='wrap')[0]
        if len(peaks) == 0:
            return no_tempo
        else:
            sorted_peaks = peaks[np.argsort(bins[peaks])[::-1]]
            return tempi[sorted_peaks][0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # ...existing code...
        x = np.transpose(self.data[index], (1, 2, 0))  # (channels, dmodel, T)
        np.random.seed()
        if self.mode == 'train':
            p = np.random.rand()
            if p >= 0.5:
                idx_sum = np.random.choice(len(x), size=2, replace=False)
                x = [x[i] for i in range(len(x)) if i not in idx_sum] + [x[idx_sum[0]] + x[idx_sum[1]]]
                q = np.random.rand()
                if q >= 0.6:
                    idx_sum = np.random.choice(len(x), size=2, replace=False)
                    x = [x[i] for i in range(len(x)) if i not in idx_sum] + [x[idx_sum[0]] + x[idx_sum[1]]]
                    r = np.random.rand()
                    if r >= 0.5:
                        idx_sum = np.random.choice(len(x), size=2, replace=False)
                        x = [x[i] for i in range(len(x)) if i not in idx_sum] + [x[idx_sum[0]] + x[idx_sum[1]]]
        x = [librosa.power_to_db(xi, ref=np.max) for xi in x]
        x = np.transpose(np.array(x), (0, 2, 1))
        if self.mode == 'test':
            return self.dataset_name[index], x, self.beats[index], self.downbeats[index], self.tempi[index], self.root[index]
        else:
            return self.dataset_name[index], x, self.beats[index], self.downbeats[index], self.tempi[index]


class AudioDatasetLoader:
    """
    Loader for AudioProcessingDataset that creates train, validation, and test splits.
    """
    def __init__(self, data_to_load, test_only_data, data_path, annotation_path,
                 fps=44100/1024, seed=0, num_folds=8, mask_value=-1, sample_size=512):
        # ...existing code...
        self.fps = fps
        self.sample_size = sample_size
        self.mask_value = mask_value
        self.num_folds = num_folds
        self.test_only_data = test_only_data
        np.random.seed(seed)
        load_data = np.load(data_path, allow_pickle=True)
        load_annotation = np.load(annotation_path, allow_pickle=True)

        self.full_data = {}
        self.full_annotation = {}
        self.audio_files = {}
        for key in load_data:
            if key in data_to_load:
                data = load_data[key]
                annotation = load_annotation[key]
                with open('./data/audio_lists/{}.txt'.format(key), 'r') as f:
                    audio_root = [line.strip() for line in f.readlines()]
                self.full_data[key] = {}
                self.full_annotation[key] = {}
                self.audio_files[key] = {}
                if key in self.test_only_data:
                    for i in range(self.num_folds):
                        self.full_data[key][i] = data[:]
                        self.full_annotation[key][i] = annotation[:]
                        self.audio_files[key][i] = audio_root[:]
                else:
                    fold_size = len(data) // self.num_folds
                    for i in range(self.num_folds - 1):
                        self.full_data[key][i] = data[i*fold_size:(i+1)*fold_size]
                        self.full_annotation[key][i] = annotation[i*fold_size:(i+1)*fold_size]
                        self.audio_files[key][i] = audio_root[i*fold_size:(i+1)*fold_size]
                    self.full_data[key][self.num_folds - 1] = data[(self.num_folds-1)*fold_size:]
                    self.full_annotation[key][self.num_folds - 1] = annotation[(self.num_folds-1)*fold_size:]
                    self.audio_files[key][self.num_folds - 1] = audio_root[(self.num_folds-1)*fold_size:]
        load_data = None
        load_annotation = None

    def get_fold(self, fold=0):
        train_set = AudioProcessingDataset(
            full_data=self.full_data,
            full_annotation=self.full_annotation,
            mode='train',
            fold=fold,
            fps=self.fps,
            sample_size=self.sample_size,
            num_folds=self.num_folds,
            mask_value=self.mask_value,
            test_only=self.test_only_data
        )
        val_set = AudioProcessingDataset(
            full_data=self.full_data,
            full_annotation=self.full_annotation,
            audio_files=self.audio_files,
            mode='validation',
            fold=fold,
            fps=self.fps,
            sample_size=self.sample_size,
            num_folds=self.num_folds,
            mask_value=self.mask_value,
            test_only=self.test_only_data
        )
        test_set = AudioProcessingDataset(
            full_data=self.full_data,
            full_annotation=self.full_annotation,
            audio_files=self.audio_files,
            mode='test',
            fold=fold,
            fps=self.fps,
            sample_size=self.sample_size,
            num_folds=self.num_folds,
            mask_value=self.mask_value,
            test_only=self.test_only_data
        )
        return train_set, val_set, test_set

# ...existing code...
