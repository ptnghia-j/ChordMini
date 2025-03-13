import numpy as np
import librosa
import os
from scipy import linalg
from scipy.optimize import nnls  # Add this import for the NNLS function

# Constants similar to the C++ implementation
nBPS = 3  # bins per semitone
nNote = 84 * nBPS  # number of notes
MIDI_basenote = 21  # MIDI number for lowest note (A0)

# Windows for weighting treble and bass ranges
treblewindow = np.concatenate((np.zeros(12), 
                              np.ones(24) * 0.5, 
                              np.ones(24), 
                              np.ones(24) * 0.5))
treblewindow = treblewindow[:84]

def cospuls(x, centre, width):
    """
    Cosine pulse function centered at 'centre' with given width
    """
    recipwidth = 1.0 / width
    abs_diff = abs(x - centre)
    if abs_diff <= 0.5 * width:
        return np.cos((x - centre) * 2 * np.pi * recipwidth) * 0.5 + 0.5
    return 0.0

def pitchCospuls(x, centre, binsperoctave):
    """
    Pitch-scaled cosine pulse function
    """
    warpedf = -binsperoctave * (np.log2(centre) - np.log2(x))
    out = cospuls(warpedf, 0.0, 2.0)
    # Scale to correct for note density
    c = np.log(2.0) / binsperoctave
    return out / (c * x) if x > 0 else 0.0

def specialConvolution(convolvee, kernel):
    """
    Special convolution as defined in the C++ code
    """
    len_convolvee = len(convolvee)
    len_kernel = len(kernel)
    
    Z = np.zeros(len_convolvee)
    
    # Main convolution part
    for n in range(len_kernel - 1, len_convolvee):
        s = 0.0
        for m in range(len_kernel):
            s += convolvee[n-m] * kernel[m]
        Z[n - len_kernel//2] = s
    
    # Fill upper and lower pads
    for n in range(len_kernel//2):
        Z[n] = Z[len_kernel//2]
    
    for n in range(len_convolvee, len_convolvee + len_kernel//2):
        Z[n - len_kernel//2] = Z[len_convolvee - len_kernel//2 - 1]
    
    return Z

def logFreqMatrix(sample_rate, blocksize):
    """
    Calculate matrix that maps from magnitude spectrum to pitch-scale spectrum
    """
    binspersemitone = nBPS
    minoctave = 0
    maxoctave = 7
    oversampling = 80
    
    # Linear frequency vector
    fft_f = np.array([i * (sample_rate / blocksize) for i in range(blocksize//2)])
    fft_width = sample_rate * 2.0 / blocksize
    
    # Linear oversampled frequency vector
    oversampled_f = np.array([i * ((sample_rate / blocksize) / oversampling) for i in range(oversampling * blocksize//2)])
    
    # Pitch-spaced frequency vector
    minMIDI = 21 + minoctave * 12
    maxMIDI = 21 + maxoctave * 12
    oob = 1.0 / binspersemitone
    
    cq_f = []
    for i in range(minMIDI, maxMIDI):
        for k in range(binspersemitone):
            cq_f.append(440 * (2.0 ** (0.083333333333 * (i + oob * k - 69))))
    cq_f.append(440 * (2.0 ** (0.083333 * (maxMIDI - 69))))
    
    cq_f = np.array(cq_f)
    nFFT = len(fft_f)
    
    # FFT activation
    fft_activation = []
    for iOS in range(2 * oversampling):
        cosp = cospuls(oversampled_f[iOS], fft_f[1], fft_width)
        fft_activation.append(cosp)
    
    # Create the log frequency matrix
    outmatrix = np.zeros((nFFT, len(cq_f)))
    
    for iFFT in range(1, nFFT):
        curr_start = oversampling * iFFT - oversampling
        curr_end = oversampling * iFFT + oversampling
        
        for iCQ in range(len(cq_f)):
            if (cq_f[iCQ] * (2.0 ** 0.084) + fft_width > fft_f[iFFT] and 
                cq_f[iCQ] * (2.0 ** (-0.084 * 2)) - fft_width < fft_f[iFFT]):
                
                for iOS in range(curr_start, curr_end):
                    if iOS < len(oversampled_f):
                        cq_activation = pitchCospuls(oversampled_f[iOS], cq_f[iCQ], binspersemitone * 12)
                        outmatrix[iFFT, iCQ] += cq_activation * fft_activation[iOS - curr_start]
    
    # Convert to sparse format
    kernel_value = []
    kernel_fft_index = []
    kernel_note_index = []
    
    for iNote in range(len(cq_f)):
        for iFFT in range(nFFT):
            if outmatrix[iFFT, iNote] > 0:
                # Only include indices that are within the valid range for nNote
                if iNote < nNote:
                    kernel_value.append(outmatrix[iFFT, iNote])
                    kernel_fft_index.append(iFFT)
                    kernel_note_index.append(iNote)
    
    return kernel_value, kernel_fft_index, kernel_note_index

def dictionaryMatrix(s_param=0.7):
    """
    Create a dictionary matrix for note mapping
    """
    binspersemitone = nBPS
    minoctave = 0
    maxoctave = 7
    
    # Pitch-spaced frequency vector
    minMIDI = 21 + minoctave * 12 - 1
    maxMIDI = 21 + maxoctave * 12
    oob = 1.0 / binspersemitone
    
    cq_f = []
    for i in range(minMIDI, maxMIDI):
        for k in range(binspersemitone):
            cq_f.append(440 * (2.0 ** (0.083333333333 * (i + oob * k - 69))))
    cq_f.append(440 * (2.0 ** (0.083333 * (maxMIDI - 69))))
    
    dm = np.zeros((nNote, 12 * (maxoctave - minoctave)))
    
    for iOut in range(12 * (maxoctave - minoctave)):
        for iHarm in range(1, 21):
            floatbin = ((iOut + 1) * binspersemitone + 1) + binspersemitone * 12 * np.log2(iHarm)
            curr_amp = s_param ** (iHarm - 1)
            
            for iNote in range(nNote):
                if abs(iNote + 1.0 - floatbin) < 2:
                    dm[iNote, iOut] += cospuls(iNote + 1.0, floatbin, binspersemitone + 0.0) * curr_amp
    
    return dm

class NNLSChroma:
    def __init__(self, sample_rate=44100, blocksize=16384, stepsize=2048):
        """
        Initialize the NNLS Chroma extractor
        
        Parameters:
        -----------
        sample_rate : int
            Sample rate of the audio signal (default: 44100)
        blocksize : int
            Frame length for FFT (default: 16384)
        stepsize : int
            Hop size between frames (default: 2048)
        """
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.stepsize = stepsize
        
        # Parameters
        self.whitening = 1.0
        self.s = 0.7  # Spectral shape parameter (0.6 to 0.9 in the paper)
        self.doNormalizeChroma = 3  # L2 norm
        self.tuneLocal = 0.0  # global tuning
        self.boostN = 0.1
        self.useNNLS = 1.0
        self.rollon = 0.0
        
        # Pre-processing method as described in the paper:
        # 0: original - no pre-processing
        # 1: subtraction - subtract the background spectrum 
        # 2: standardization - subtract background and divide by running std dev
        self.preprocessing_method = 0
        
        # Create the dictionary matrix and log freq matrix
        self.dict = dictionaryMatrix(self.s)
        self.kernel_value, self.kernel_fft_index, self.kernel_note_index = logFreqMatrix(
            self.sample_rate, self.blocksize)
        
        # For tuning estimation
        self.sinvalues = np.sin(2 * np.pi * (np.arange(nBPS) / nBPS))
        self.cosvalues = np.cos(2 * np.pi * (np.arange(nBPS) / nBPS))
        
        # Make hamming window of length 1/2 octave
        hamwinlength = nBPS * 6 + 1
        hamwinsum = 0
        hw = []
        for i in range(hamwinlength):
            hw_val = 0.54 - 0.46 * np.cos((2 * np.pi * i) / (hamwinlength - 1))
            hw.append(hw_val)
            hamwinsum += hw_val
        self.hw = np.array(hw) / hamwinsum
        
        # Initialize tuning
        self.mean_tunings = np.zeros(nBPS)
        self.local_tunings = np.zeros(nBPS)
        self.local_tuning = []
        self.frame_count = 0
        self.log_spectrum = []
    
    def process_frame(self, fft_data):
        """
        Process a single FFT frame
        """
        self.frame_count += 1
        
        # Extract magnitude from FFT data
        magnitude = np.sqrt(fft_data[0:self.blocksize//2, 0]**2 + fft_data[0:self.blocksize//2, 1]**2)
        
        # Apply rollon if needed
        if self.rollon > 0:
            energysum = np.sum(magnitude**2)
            cumenergy = 0
            for i in range(2, self.blocksize//2):
                cumenergy += magnitude[i]**2
                if cumenergy < energysum * self.rollon / 100:
                    magnitude[i-2] = 0
                else:
                    break
        
        # Note magnitude mapping using pre-calculated matrix
        nm = np.zeros(nNote)
        
        for i, k_val in enumerate(self.kernel_value):
            # Add safety check to prevent index out of bounds
            if self.kernel_note_index[i] < nNote and self.kernel_fft_index[i] < len(magnitude):
                nm[self.kernel_note_index[i]] += magnitude[self.kernel_fft_index[i]] * k_val
        
        one_over_N = 1.0 / self.frame_count
        
        # Update means of complex tuning variables
        self.mean_tunings *= float(self.frame_count - 1) * one_over_N
        
        for iTone in range(0, round(nNote * 0.62 / nBPS) * nBPS + 1, nBPS):
            self.mean_tunings += nm[iTone:iTone+nBPS] * one_over_N
            
            ratioOld = 0.997
            self.local_tunings *= ratioOld
            self.local_tunings += nm[iTone:iTone+nBPS] * (1 - ratioOld)
        
        # Local tuning
        localTuningImag = np.sum(self.local_tunings * self.sinvalues)
        localTuningReal = np.sum(self.local_tunings * self.cosvalues)
        
        normalisedtuning = np.arctan2(localTuningImag, localTuningReal) / (2 * np.pi)
        self.local_tuning.append(normalisedtuning)
        
        # Store log spectrum
        self.log_spectrum.append(nm)
        
        return nm
    
    def extract_chroma(self):
        """
        Extract chromagram features from processed frames
        """
        if len(self.log_spectrum) == 0:
            return None
        
        # Calculate tuning
        meanTuningImag = np.sum(self.mean_tunings * self.sinvalues)
        meanTuningReal = np.sum(self.mean_tunings * self.cosvalues)
        
        normalisedtuning = np.arctan2(meanTuningImag, meanTuningReal) / (2 * np.pi)
        intShift = int(np.floor(normalisedtuning * 3))
        floatShift = normalisedtuning * 3 - intShift
        
        # Process each frame
        tuned_log_spectrum = []
        
        for frame_idx, log_frame in enumerate(self.log_spectrum):
            # Apply tuning
            if self.tuneLocal:
                intShift = int(np.floor(self.local_tuning[frame_idx] * 3))
                floatShift = self.local_tuning[frame_idx] * 3 - intShift
            
            # Create tuned log frame
            tuned_frame = np.zeros_like(log_frame)
            tuned_frame[:2] = 0  # set lower edge to zero
            
            # Interpolate inner bins
            for k in range(2, len(log_frame) - 3):
                if k + intShift < len(log_frame) and k + intShift + 1 < len(log_frame):
                    tuned_frame[k] = log_frame[k + intShift] * (1 - floatShift) + log_frame[k + intShift + 1] * floatShift
            
            tuned_frame[-3:] = 0  # upper edge
            
            # Apply pre-processing as described in the paper
            if self.preprocessing_method > 0:
                # Calculate the running mean (background spectrum)
                # Using octave-wide Hamming-windowed neighborhood (+-18 bins = 6 semitones)
                running_mean = specialConvolution(tuned_frame, self.hw)
                
                # For standardization (method 2), calculate running standard deviation
                if self.preprocessing_method == 2:
                    running_std = np.zeros_like(tuned_frame)
                    for i in range(nNote):
                        running_std[i] = (tuned_frame[i] - running_mean[i]) ** 2
                    running_std = specialConvolution(running_std, self.hw)
                    running_std = np.sqrt(running_std)
                
                # Apply the pre-processing as per equation (2) in the paper
                for i in range(nNote):
                    if tuned_frame[i] - running_mean[i] > 0:
                        if self.preprocessing_method == 1:  # subtraction
                            tuned_frame[i] = tuned_frame[i] - running_mean[i]
                        elif self.preprocessing_method == 2:  # standardization
                            if running_std[i] > 0:
                                tuned_frame[i] = (tuned_frame[i] - running_mean[i]) / running_std[i]
                            else:
                                tuned_frame[i] = 0
                    else:
                        tuned_frame[i] = 0
                    
            tuned_log_spectrum.append(tuned_frame)
        
        # Extract semitone spectrum and chromagram
        chromagrams = []
        
        for tuned_frame in tuned_log_spectrum:
            chroma = np.zeros(12)
            
            if self.useNNLS == 0:
                # Simple mapping approach - just copy the centre bin of every semitone
                for iNote in range(nBPS//2 + 2, nNote - nBPS//2, nBPS):
                    semitone_idx = (iNote - (nBPS//2 + 2)) // nBPS
                    chroma[semitone_idx % 12] += tuned_frame[iNote]
            else:
                # Use NNLS approach as described in the paper
                # Create semitone spectrum with indices that have energy
                semitone_spectrum = np.zeros(84)
                signif_index = []
                
                for index, iNote in enumerate(range(nBPS//2 + 2, nNote - nBPS//2, nBPS)):
                    curr_val = 0
                    for iBPS in range(-nBPS//2, nBPS//2 + 1):
                        if iNote + iBPS < len(tuned_frame):
                            curr_val += tuned_frame[iNote + iBPS]
                    
                    if curr_val > 0:
                        signif_index.append(index)
                
                if signif_index:
                    try:
                        # Create dictionary for NNLS
                        curr_dict = np.zeros((nNote, len(signif_index)))
                        
                        for i, note_idx in enumerate(signif_index):
                            curr_dict[:, i] = self.dict[:, note_idx % 12]
                        
                        # Add a small regularization term to prevent singular matrix
                        reg_lambda = 1e-10
                        
                        # Solve NNLS with robust error handling
                        try:
                            x, _ = nnls(curr_dict, tuned_frame)
                        except np.linalg.LinAlgError:
                            # If we get a singular matrix, try with more regularization
                            try:
                                # Add a small identity component to make the matrix better conditioned
                                curr_dict = curr_dict + np.random.normal(0, reg_lambda, curr_dict.shape)
                                x, _ = nnls(curr_dict, tuned_frame)
                            except:
                                # If still failing, fall back to simple method
                                x = np.zeros(len(signif_index))
                                for i, note_idx in enumerate(signif_index):
                                    x[i] = tuned_frame[nBPS//2 + 2 + note_idx * nBPS]
                        
                        # Map back to semitone spectrum
                        for i, note_idx in enumerate(signif_index):
                            if note_idx < len(semitone_spectrum):
                                semitone_spectrum[note_idx] = x[i]
                                if note_idx % 12 < len(chroma) and note_idx < len(treblewindow):
                                    chroma[note_idx % 12] += x[i] * treblewindow[note_idx]
                    except Exception as e:
                        print(f"Error in NNLS processing: {e}")
                        # Fall back to simple method if anything goes wrong
                        for index, iNote in enumerate(range(nBPS//2 + 2, nNote - nBPS//2, nBPS)):
                            if iNote < len(tuned_frame):
                                semitone_idx = (iNote - (nBPS//2 + 2)) // nBPS
                                if semitone_idx % 12 < len(chroma) and semitone_idx < len(treblewindow):
                                    chroma[semitone_idx % 12] += tuned_frame[iNote] * treblewindow[semitone_idx]
            
            # Normalize chroma if needed
            if self.doNormalizeChroma > 0:
                if self.doNormalizeChroma == 1:  # max norm
                    max_val = np.max(chroma)
                    if max_val > 0:
                        chroma /= max_val
                elif self.doNormalizeChroma == 2:  # L1 norm
                    sum_val = np.sum(chroma)
                    if sum_val > 0:
                        chroma /= sum_val
                elif self.doNormalizeChroma == 3:  # L2 norm
                    sum_squared = np.sum(chroma**2)
                    if sum_squared > 0:
                        chroma /= np.sqrt(sum_squared)
            
            chromagrams.append(chroma)
        
        return np.array(chromagrams)
    
    def extract_features(self, audio_data, sr=None):
        """
        Extract NNLS chroma features from audio data
        
        Parameters:
        -----------
        audio_data : numpy array
            Audio signal (mono)
        sr : int, optional
            Sample rate of the audio data. If not provided, uses the class's sample_rate.
            
        Returns:
        --------
        chromagram : numpy array
            Matrix of chroma features (n_frames x 12)
        """
        # Use provided sample rate if given
        if sr is not None:
            self.sample_rate = sr
            
        # Reset state
        self.log_spectrum = []
        self.local_tuning = []
        self.mean_tunings = np.zeros(nBPS)
        self.local_tunings = np.zeros(nBPS)
        self.frame_count = 0
        
        # Compute STFT
        stft = librosa.stft(
            audio_data,
            n_fft=self.blocksize,
            hop_length=self.stepsize,
            window='hamming',  # Use Hamming window as specified in the paper
            center=True
        )
        
        # Convert to complex array
        complex_stft = stft.T
        real_part = np.real(complex_stft)
        imag_part = np.imag(complex_stft)
        
        # Format as needed for our algorithm
        fft_data = np.zeros((len(complex_stft), self.blocksize//2, 2))
        fft_data[:, :, 0] = real_part[:, :self.blocksize//2]
        fft_data[:, :, 1] = imag_part[:, :self.blocksize//2]
        
        # Process each frame
        for i in range(len(fft_data)):
            self.process_frame(fft_data[i])
        
        # Extract chroma features
        return self.extract_chroma()

def extract_chroma(audio_path, hop_size=2048, output_csv=None, preprocessing_method=1):
    """
    Extract chromagram from audio file using librosa's built-in functions
    
    Parameters:
    -----------
    audio_path : str
        Path to the audio file
    hop_size : int, optional
        Hop size in samples (default: 2048)
    output_csv : str, optional
        Path to save the CSV output file. If None, will not save to CSV.
    preprocessing_method : int, optional
        Pre-processing method: 0=none, 1=subtraction (default), 2=standardization
        
    Returns:
    --------
    timestamps : numpy array
        Array of timestamps for each frame
    chromagram : numpy array
        Matrix of chroma features (n_frames x 12)
    """
    # Load the audio file
    print(f"Loading audio file: {audio_path}")
    y, sr = librosa.load(audio_path, sr=22050)  # Standard sample rate
    
    # Set parameters for feature extraction
    n_fft = 4096  # FFT window size
    
    # Extract chromagram using librosa
    print(f"Extracting chromagram...")
    
    # Choose extraction method based on preprocessing_method
    if preprocessing_method == 0:
        # Basic chromagram with no preprocessing
        chromagram = librosa.feature.chroma_stft(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_size, norm=None
        )
    else:
        # With harmonic separation for better chord detection
        y_harmonic = librosa.effects.harmonic(y=y, margin=4.0)
        
        if preprocessing_method == 1:  # Subtraction method
            # Use CQT-based chromagram for better pitch detection
            chromagram = librosa.feature.chroma_cqt(
                y=y_harmonic, sr=sr, hop_length=hop_size
            )
            
            # Apply log transformation and subtract median
            chromagram = np.log1p(chromagram)
            chromagram = chromagram - np.median(chromagram, axis=1, keepdims=True)
            chromagram = np.maximum(chromagram, 0.0)  # Keep only positive values
            
        elif preprocessing_method == 2:  # Standardization
            # NNLS chromagram for improved chord extraction
            chromagram = librosa.feature.chroma_cens(
                y=y_harmonic, sr=sr, hop_length=hop_size
            )
            
            # Standardize features
            chromagram = (chromagram - np.mean(chromagram, axis=1, keepdims=True)) / (
                np.std(chromagram, axis=1, keepdims=True) + 1e-8
            )
            chromagram = np.maximum(chromagram, 0.0)  # Keep only positive values
    
    # Rotate the chromagram to shift from C-based to A-based (rotate by 3 positions)
    # This changes C->C#->D->D#->E->F->F#->G->G#->A->A#->B to A->A#->B->C->C#->D->D#->E->F->F#->G->G#
    chromagram = np.roll(chromagram, 3, axis=0)
    
    # Transpose to get [n_frames, 12] shape
    chromagram = chromagram.T
    
    # Calculate timestamps
    timestamps = librosa.times_like(chromagram, sr=sr, hop_length=hop_size)
    
    # Save to CSV if output_csv is provided
    if output_csv is not None:
        # Get the filename for the CSV header
        filename = os.path.basename(audio_path)
        
        print(f"Writing CSV to: {output_csv}")
        with open(output_csv, 'w') as f:
            # Write header row with filename in first column
            f.write(f'"{filename}",Time,A,A#,B,C,C#,D,D#,E,F,F#,G,G#\n')
            
            # Write each row with timestamp and chroma values
            for i, (time, chroma) in enumerate(zip(timestamps, chromagram)):
                # Format with precision but without scientific notation
                chroma_str = [f"{c:.7f}" if c >= 0.0001 else "0" for c in chroma]
                f.write(f',{time:.7f},{",".join(chroma_str)}\n')
        
        print(f"Completed! CSV saved to {output_csv}")
    
    return timestamps, chromagram
