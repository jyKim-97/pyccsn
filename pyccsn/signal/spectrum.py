# from scipy.signal import welch, butter
from scipy import signal
import numpy as np
import fcwt

def compute_spectrum(x, fs, window="boxcar", frange=None, axis=-1):
    """
    Estimate the power spectrum of a signal

    Args:
        x (array_like): Input signal
        fs (float): sample rate of the signal
        window (str, optional): Type of window to use. Defaults to "boxcar".
        frange (tuple, optional): Frequency range to consider. Defaults to None.
        axis (int, optional): Axis along which the spectrum is coputed. Defaults to -1.
    """
    f, px = signal.welch(x, fs=fs,
                         window=window, 
                         nperseg=np.shape(x)[axis], noverlap=None, axis=axis, 
                         detrend=None, 
                         return_onesided=True, scaling='spectrum')
    px *= 2
    
    if frange is not None:
        idf = (f >= frange[0]) & (f <= frange[1])
        f = f[idf]
        px = px[idf]
    
    return f, px


def compute_welch_spectrum(x, fs, nperseg=256, noverlap=None, window="hann", axis=-1):
    """
    Estimate the power spectrum of a signal using Welch's method

    Args:
        x (array_like): Input signal
        fs (float): sample rate of the signal
        nperseg (int, optional): Length of each segment for Welch's method. Defaults to 256.
        noverlap (int, optional): Number of points to overlap between segments. Defaults to None.
        axis (int, optional): Axis along which the spectrum is coputed. Defaults to -1.
    """
    f, px = signal.welch(x, fs=fs,
                         nperseg=nperseg, noverlap=noverlap, axis=axis, 
                         window=window,
                         return_onesided=True, scaling='spectrum', detrend=None)
    px *= 2
    return f, px


def compute_spectrogram(x, fs, t=None, wbin_t=1, mbin_t=0.1, frange=None):
    """
    Estimate the power spectrogram by short-term Fourier transform (STFT)

    Args:
        x (array_like): 1D or 2D input signal
                        For 2D signal, the second dimension is assumed to be time.
        fs (float): sample rate of the signal
        t (array_like, optional): Time vector corresponding to the signal. 
                                    If it is not given, it is computed from the signal and sample rate.
        wbin_t (float, optional): Time window for each segment in seconds. Defaults to 1.
        mbin_t (float, optional): Time bin for the spectrogram in seconds. Defaults to 0.1.
        axis (int, optional): Axis along which the spectrum is coputed. Defaults to -1.
    """
    mbin = int(mbin_t * fs)
    wbin = int(wbin_t * fs)
    
    x = np.array(x)
    shrink_axis = False
    if len(np.shape(x)) == 1:
        x = np.expand_dims(x, axis=0)
        shrink_axis = True
    
    if np.shape(x)[1] < wbin:
        raise ValueError("Length of signal (%d) must be greater than wbin (%d)"%(np.shape(x,2), wbin))
    
    if t is None:
        t = np.arange(0, len(x)/fs)
    else:
        t = np.asarray(t)
        assert np.shape(x)[1] == len(t), "Length of signal and time vector must match."
        
    # nbin_set = np.arange(int(t[0]*fs), int(t[-1]*fs)-wbin, mbin)
    nbin_set = np.arange(0, int((t[-1]-t[0])*fs)-wbin, mbin)
    num_f = wbin//2 + 1
    
    pxx = np.zeros((x.shape[0], num_f, len(nbin_set)))
    for i, n0 in enumerate(nbin_set):
        xseg = x[:, n0:n0+wbin]
        f, pxx[:,:,i] = compute_spectrum(xseg, fs=fs, window="hann", axis=1)
        
    if frange is not None:
        idf = (f >= frange[0]) & (f <= frange[1])
        f = f[idf]
        pxx = pxx[:, idf, :]
    
    if shrink_axis:
        pxx = pxx[0]
        
    tp = (nbin_set + wbin/2)/fs + t[0]
    
    return f, tp, pxx
    
    
import numpy as np

def compute_wavelet_spectrogram(
    x: np.ndarray,
    fs: float,
    frange=None,                 # (f0, f1) in Hz; if None, defaults to (lowest, Nyquist)
    fn: int = 100,               # number of frequency bins
    axis: int = -1,              # time axis
    scaling: str = "lin",        # "lin" or "log"
    nthreads: int = None,        # None -> use os.cpu_count()
    fast: bool = False,           # use optimization plans
    norm: bool = True,           # normalize time-frequency output
    mode: str = "power",         # "power" or "amplitude"
    baseline=None,               # (t_start, t_end) in seconds, optional
    baseline_mode: str = "none", # "db" | "zscore" | "percent" | "div" | "none"
    decim: int = 1               # temporal decimation factor
):
    """
    fCWT-based wavelet spectrogram with output shaped [F, T, ...].
    # https://github.com/fastlib/fCWT/blob/main/tutorial.ipynb

    Args:
        x: Input array; time must be along `axis`. Any leading/trailing dims are allowed.
        fs: Sampling rate (Hz).
        frange: Tuple (f0, f1) in Hz. If None, f0 uses the lowest allowed and f1=Nyquist.
        fn: Number of frequency bins.
        axis: Time axis in `x`.
        scaling: Frequency spacing, "lin" or "log".
        nthreads: Threads used by fCWT (defaults to os.cpu_count()).
        fast: Use fCWT optimization plans (may run a planning step).
        norm: Normalize the output amplitudes (as in fCWT default).
        mode: "power" (|C|^2) or "amplitude" (|C|).
        baseline: (t0, t1) seconds for baseline normalization.
        baseline_mode: 'db' | 'zscore' | 'percent' | 'div' | 'none'.
        decim: Downsample factor along time for the output.

    Returns:
        f: (F,) frequency vector in Hz.
        t: (T',) time vector in seconds (after decimation).
        S: (F, T', ...) spectrogram (power or amplitude).
    """
    try:
        import fcwt
    except ImportError as e:
        raise ImportError("fcwt is not installed. Install with `pip install fCWT`.") from e

    # Ensure array and move time axis to the end for iteration
    x = np.asarray(x)
    x = np.moveaxis(x, axis, -1)            # shape: (..., T)
    orig_shape = x.shape
    T = orig_shape[-1]
    lead = int(np.prod(orig_shape[:-1]))    # flatten all non-time dims
    x_flat = x.reshape(lead, T)             # (N, T) where N is product of other dims

    # Frequencies range
    if frange is None:
        f0 = 0.0    # fCWT will clamp to min allowed
        f1 = fs / 2
    else:
        f0, f1 = frange

    # Threads default
    if nthreads is None:
        try:
            import os
            nthreads = os.cpu_count() or 1
        except Exception:
            nthreads = 1

    # Time vector (seconds)
    t = np.arange(T) / fs

    # Compute CWT channel-by-channel
    f = None
    S_list = []
    for i in range(lead):
        sig = x_flat[i]

        # fCWT returns (freqs, coeffs) where coeffs is (F, T) complex
        freqs, coeffs = fcwt.cwt(
            sig.astype(np.float64, copy=False),
            fs=fs,
            f0=f0, f1=f1, fn=fn,
            nthreads=nthreads,
            scaling=scaling,
            fast=fast,
            norm=norm
        )
        if f is None:
            f = freqs

        if mode == "power":
            S_list.append(np.abs(coeffs) ** 2)
        elif mode == "amplitude":
            S_list.append(np.abs(coeffs))
        else:
            raise ValueError("mode must be 'power' or 'amplitude' for fCWT wrapper.")
        
    S = np.stack(S_list, axis=0)                # (N, F, T)
    idf = np.argsort(f)
    f = f[idf]
    S = S[:,idf,...]
    # Stack back: (N, F, T) → (F, T, ...)
    S = np.moveaxis(S, 1, -1)                   # (N, T, F)
    S = np.moveaxis(S, -1, 0)                   # (F, N, T)
    # Now reorder to [F, T, ...]
    S = np.moveaxis(S, -1, 1)                   # (F, T, N)
    S = S.reshape((len(f), T) + orig_shape[:-1])  # (F, T, ...)

    # Decimation along time
    if decim > 1:
        S = S[:, ::decim, ...]
        t = t[::decim]

    # Optional baseline normalization (on last time axis=1 in [F, T, ...])
    if baseline is not None and baseline_mode != "none":
        t0, t1 = baseline
        i0 = max(0, int(np.floor(t0 * fs / decim)))
        i1 = min(S.shape[1], int(np.ceil(t1 * fs / decim)))
        if i1 <= i0:
            raise ValueError("Invalid baseline window.")
        B = S[:, i0:i1, ...]  # (F, Tb, ...)
        m = np.mean(B, axis=1, keepdims=True)
        if baseline_mode == "db":
            S = 10 * np.log10(S / (m + 1e-20))
        elif baseline_mode == "percent":
            S = 100 * (S - m) / (m + 1e-20)
        elif baseline_mode == "div":
            S = S / (m + 1e-20)
        elif baseline_mode == "zscore":
            s = np.std(B, axis=1, keepdims=True) + 1e-20
            S = (S - m) / s
        else:
            raise ValueError("Unknown baseline_mode: {baseline_mode}")

    return f, t, S
