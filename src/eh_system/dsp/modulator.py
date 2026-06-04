"""Elektronik Taarruz (ET) — analog telsiz aldatma için ses zinciri + NBFM.

Tipik bir dar bant FM (NBFM) telsiz vericisinin ses işleme zincirini ve FM
modülasyonunu üretir:
1. Bant geçiren (300–3000 Hz): telsiz ses bandı dışını süzer.
2. Pre-emphasis (50 µs): yüksek frekansları yükselterek FM gürültü başarımını
   artırır (alıcıda de-emphasis ile düzleştirilir).
3. CTCSS: alt-ses (sub-audible) ton ekler; uyumlu alıcının squelch'ini açar.
4. NBFM modülasyon: işlenmiş sesi sabit zarflı IQ'ya çevirir.

Tüm fonksiyonlar saftır (yan etkisiz), yalnız numpy/scipy kullanır, pytest ile
test edilebilir. Qt/UI bilmez. Çıktı baseband ``complex64`` IQ; TX zincirinde
merkez frekansa taşınır ve gücü yazılımsal limitle ölçeklenir (emniyet).

UYARI: Telsiz aldatma yalnız yetkili test ortamında ve yasal izinle yapılır.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

# Telsiz ses bandı (Hz).
AUDIO_BAND_LOW = 300.0
AUDIO_BAND_HIGH = 3000.0
# Pre-emphasis zaman sabiti (s) — 50 µs (telsiz standardı).
PRE_EMPHASIS_TAU = 50.0e-6
# Yaygın CTCSS ton frekansları (Hz) — bilgilendirme amaçlı.
CTCSS_TONES = (67.0, 71.9, 88.5, 100.0, 123.0, 151.4, 203.5, 250.3)
# Varsayılan NBFM tepe sapması (Hz) — dar bant (±2.5 kHz).
DEFAULT_DEVIATION_HZ = 2500.0
# CTCSS varsayılan seviyesi (ses tepe genliğine oran).
_CTCSS_LEVEL = 0.15
_EPS = 1e-12


def _normalize(audio: np.ndarray) -> np.ndarray:
    """Sesi [-1, 1] tepe aralığına normalize et (sessizlik korunur)."""
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > _EPS:
        return audio / peak
    return audio


def bandpass_filter(
    audio: np.ndarray,
    sample_rate: float,
    low_hz: float = AUDIO_BAND_LOW,
    high_hz: float = AUDIO_BAND_HIGH,
    order: int = 4,
) -> np.ndarray:
    """Sesi telsiz ses bandına (``low_hz``–``high_hz``) sınırla (Butterworth)."""
    nyq = sample_rate / 2.0
    if not 0.0 < low_hz < high_hz < nyq:
        raise ValueError("0 < low_hz < high_hz < Nyquist olmalı.")
    sos = butter(order, [low_hz / nyq, high_hz / nyq], btype="bandpass", output="sos")
    return sosfilt(sos, audio).astype(np.float64)


def pre_emphasis(
    audio: np.ndarray,
    sample_rate: float,
    tau: float = PRE_EMPHASIS_TAU,
) -> np.ndarray:
    """Birinci derece pre-emphasis (yüksek frekansları yükseltir).

    y[n] = x[n] − a·x[n−1],  a = exp(−1 / (sample_rate·tau)).
    """
    if tau <= 0.0:
        raise ValueError("tau pozitif olmalı.")
    a = float(np.exp(-1.0 / (sample_rate * tau)))
    y = np.empty_like(audio, dtype=np.float64)
    y[0] = audio[0]
    y[1:] = audio[1:] - a * audio[:-1]
    return y


def add_ctcss(
    audio: np.ndarray,
    sample_rate: float,
    ctcss_freq_hz: float,
    level: float = _CTCSS_LEVEL,
) -> np.ndarray:
    """Sese alt-ses CTCSS tonu ekle (uyumlu alıcı squelch'i için).

    Args:
        audio: Ses örnekleri (tercihen [-1,1]).
        sample_rate: Örnekleme hızı (Hz).
        ctcss_freq_hz: CTCSS ton frekansı (Hz, tipik 67–250).
        level: Tonun ses tepesine göre seviyesi (0..1).
    """
    if not 0.0 < ctcss_freq_hz < sample_rate / 2.0:
        raise ValueError("CTCSS frekansı 0 < f < Nyquist olmalı.")
    t = np.arange(audio.shape[0], dtype=np.float64) / sample_rate
    tone = level * np.sin(2.0 * np.pi * ctcss_freq_hz * t)
    return (audio + tone).astype(np.float64)


def nbfm_modulate(
    audio: np.ndarray,
    sample_rate: float,
    deviation: float = DEFAULT_DEVIATION_HZ,
) -> np.ndarray:
    """Dar bant FM modülasyon: işlenmiş sesi sabit zarflı IQ'ya çevir.

    Anlık frekans = deviation · audio_norm(t); faz integral ile bulunur:
        φ[n] = 2π·deviation/sr · Σ audio_norm[0..n]
        iq[n] = exp(j·φ[n])  → |iq| = 1 (sabit zarf).

    Ses [-1, 1]'e normalize edildiğinden tepe frekans sapması tam ``deviation``
    olur.

    Args:
        audio: Ses örnekleri (modüle edilecek mesaj).
        sample_rate: Örnekleme hızı (Hz).
        deviation: Tepe frekans sapması (Hz).

    Returns:
        ``complex64`` IQ dizisi (uzunluk = audio uzunluğu), |IQ| = 1.
    """
    if deviation <= 0.0:
        raise ValueError("deviation pozitif olmalı.")
    if audio.shape[0] == 0:
        raise ValueError("Boş ses girişi.")

    norm = _normalize(audio.astype(np.float64))
    # Faz = 2π·dev/sr · kümülatif toplam (ayrık integral).
    phase = 2.0 * np.pi * deviation / sample_rate * np.cumsum(norm)
    return np.exp(1j * phase).astype(np.complex64)


def process_voice(
    audio: np.ndarray,
    sample_rate: float,
    ctcss_freq_hz: float | None = None,
    deviation: float = DEFAULT_DEVIATION_HZ,
) -> np.ndarray:
    """Tam aldatma ses zinciri: bandpass → pre-emphasis → CTCSS → NBFM.

    Args:
        audio: Ham ses (mic/wav).
        sample_rate: Örnekleme hızı (Hz).
        ctcss_freq_hz: Eklenecek CTCSS tonu (None → eklenmez).
        deviation: NBFM tepe sapması (Hz).

    Returns:
        Modüle edilmiş ``complex64`` IQ (sabit zarf).
    """
    shaped = bandpass_filter(audio.astype(np.float64), sample_rate)
    shaped = pre_emphasis(shaped, sample_rate)
    shaped = _normalize(shaped)
    if ctcss_freq_hz is not None:
        shaped = add_ctcss(shaped, sample_rate, ctcss_freq_hz)
    return nbfm_modulate(shaped, sample_rate, deviation)
