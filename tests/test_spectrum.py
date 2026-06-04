"""dsp.spectrum birim testleri: apply_window ve compute_psd."""

from __future__ import annotations

import numpy as np
import pytest

from eh_system.dsp.spectrum import (
    WINDOW_BLACKMAN_HARRIS,
    WINDOW_HANN,
    apply_window,
    compute_psd,
)


def test_apply_window_length_and_dtype() -> None:
    """Pencere uygulaması uzunluğu ve karmaşık tipi korur."""
    n = 256
    iq = (np.random.randn(n) + 1j * np.random.randn(n)).astype(np.complex64)
    out = apply_window(iq, WINDOW_HANN)
    assert out.shape == iq.shape
    assert np.iscomplexobj(out)


def test_apply_window_hann_tapers_edges() -> None:
    """Hann penceresi kenarları sıfıra yakın bastırır."""
    n = 128
    iq = np.ones(n, dtype=np.complex64)
    out = apply_window(iq, WINDOW_HANN)
    assert abs(out[0]) < abs(out[n // 2])


def test_apply_window_unknown_raises() -> None:
    """Bilinmeyen pencere tipi ValueError fırlatır."""
    iq = np.ones(64, dtype=np.complex64)
    with pytest.raises(ValueError):
        apply_window(iq, "kaiser-not-supported")


def test_compute_psd_shapes() -> None:
    """compute_psd doğru uzunlukta freqs ve psd_db döndürür."""
    fft_size = 512
    iq = (np.random.randn(fft_size) + 1j * np.random.randn(fft_size)).astype(
        np.complex64
    )
    freqs, psd_db = compute_psd(iq, fft_size, sample_rate=1e6)
    assert freqs.shape == (fft_size,)
    assert psd_db.shape == (fft_size,)
    # Frekans ekseni artan olmalı.
    assert np.all(np.diff(freqs) > 0)


def test_compute_psd_detects_tone() -> None:
    """Tek tonun tepe frekansı doğru bölmeye düşer."""
    fft_size = 1024
    fs = 1.0e6
    # Bölmeye tam oturan bir ton seç.
    bin_idx = 200
    f_tone = bin_idx * fs / fft_size
    t = np.arange(fft_size) / fs
    iq = np.exp(2j * np.pi * f_tone * t).astype(np.complex64)
    freqs, psd_db = compute_psd(iq, fft_size, sample_rate=fs, window=WINDOW_HANN)
    peak_freq = freqs[int(np.argmax(psd_db))]
    assert abs(peak_freq - f_tone) < (fs / fft_size)


def test_compute_psd_center_freq_offset() -> None:
    """center_freq frekans eksenini kaydırır."""
    fft_size = 256
    fs = 2.0e6
    fc = 433.0e6
    iq = np.ones(fft_size, dtype=np.complex64)
    freqs, _ = compute_psd(iq, fft_size, sample_rate=fs, center_freq=fc)
    assert abs(freqs[fft_size // 2] - fc) < 1.0


def test_compute_psd_blackman_harris_runs() -> None:
    """Blackman-Harris penceresi ile de çalışır."""
    fft_size = 256
    iq = (np.random.randn(fft_size) + 1j * np.random.randn(fft_size)).astype(
        np.complex64
    )
    freqs, psd_db = compute_psd(
        iq, fft_size, sample_rate=1e6, window=WINDOW_BLACKMAN_HARRIS
    )
    assert np.all(np.isfinite(psd_db))


def test_compute_psd_too_short_raises() -> None:
    """IQ uzunluğu fft_size'tan küçükse ValueError."""
    iq = np.ones(100, dtype=np.complex64)
    with pytest.raises(ValueError):
        compute_psd(iq, 256)
