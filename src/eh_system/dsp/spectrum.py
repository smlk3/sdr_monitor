"""Spektrum DSP: pencereleme ve güç spektral yoğunluğu (PSD).

Tüm fonksiyonlar saftır (yan etkisiz), yalnızca numpy/scipy kullanır ve pytest
ile test edilebilir. Qt/UI bilmez.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import windows

# Desteklenen pencere tipleri.
WINDOW_HANN = "hann"
WINDOW_BLACKMAN_HARRIS = "blackman-harris"

# log10(0) kaçınmak için taban (lineer güç).
_DB_FLOOR = 1e-20


def apply_window(iq: np.ndarray, window: str = WINDOW_HANN) -> np.ndarray:
    """IQ bloğuna pencere uygula.

    Args:
        iq: complex64/complex128 IQ örnekleri (1B).
        window: "hann" veya "blackman-harris".

    Returns:
        Pencerelenmiş IQ (giriş ile aynı uzunluk ve karmaşık tip).
    """
    n = iq.shape[0]
    w = _get_window(window, n)
    return iq * w.astype(iq.dtype if np.iscomplexobj(iq) else np.float64)


def _get_window(window: str, n: int) -> np.ndarray:
    """İsme göre pencere katsayılarını (float64, uzunluk ``n``) döndür."""
    key = window.lower()
    if key == WINDOW_HANN:
        return windows.hann(n, sym=False)
    if key in (WINDOW_BLACKMAN_HARRIS, "blackmanharris", "blackman_harris"):
        return windows.blackmanharris(n, sym=False)
    raise ValueError(f"Bilinmeyen pencere tipi: {window!r}")


def compute_psd(
    iq: np.ndarray,
    fft_size: int,
    sample_rate: float = 1.0,
    center_freq: float = 0.0,
    window: str = WINDOW_HANN,
) -> tuple[np.ndarray, np.ndarray]:
    """Tek taraflı değil, karmaşık IQ için iki taraflı PSD hesapla.

    Pencereleme uygulanır, FFT alınır, fftshift ile DC ortaya getirilir ve
    güç dB ölçeğine çevrilir. Pencere güç kaybı telafi edilir.

    Args:
        iq: IQ örnekleri (en az ``fft_size`` uzunlukta; ilk ``fft_size`` kullanılır).
        fft_size: FFT uzunluğu.
        sample_rate: Örnekleme hızı (Hz). Frekans eksenini ölçekler.
        center_freq: Merkez frekansı (Hz). Frekans ekseni buna kaydırılır.
        window: Pencere tipi.

    Returns:
        (freqs, psd_db):
            freqs: Hz cinsinden frekans ekseni (uzunluk ``fft_size``), artan.
            psd_db: dB cinsinden güç (uzunluk ``fft_size``).
    """
    if fft_size <= 0:
        raise ValueError("fft_size pozitif olmalı.")
    if iq.shape[0] < fft_size:
        raise ValueError(
            f"IQ uzunluğu ({iq.shape[0]}) fft_size'tan ({fft_size}) küçük olamaz."
        )

    seg = iq[:fft_size]
    w = _get_window(window, fft_size)
    # Pencere kazanç düzeltmesi (koherent kazanç): 1/sum(w).
    win_sum = np.sum(w)
    seg_w = seg * w.astype(seg.dtype if np.iscomplexobj(seg) else np.float64)

    spec = np.fft.fftshift(np.fft.fft(seg_w, n=fft_size))
    # Genlik normalizasyonu: pencere toplamına böl.
    spec = spec / win_sum
    power = (np.abs(spec) ** 2)
    psd_db = 10.0 * np.log10(power + _DB_FLOOR)

    freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1.0 / sample_rate))
    freqs = freqs + center_freq

    return freqs.astype(np.float64), psd_db.astype(np.float64)
