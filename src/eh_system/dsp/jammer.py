"""Elektronik Taarruz (ET) — karıştırma (jamming) dalga formu üreteçleri.

Üç temel karıştırma türü üretir; hepsi baseband ``complex64`` IQ döndürür ve
TX zincirinde merkez frekansa taşınır:
- ``barrage_jam``: bant sınırlı AWGN (geniş bant baraj karıştırma).
- ``multi_tone_jam``: çoklu CW ton (belirli kanalları nokta karıştırma).
- ``sweep_jam``: doğrusal chirp (taramalı/sweep karıştırma).

Tüm fonksiyonlar saftır (yan etkisiz), yalnızca numpy/scipy kullanır ve pytest
ile test edilebilir. Qt/UI bilmez. Genlik [0, 1] normalizedir; gerçek TX gücü
HAL/TX worker tarafında yazılımsal limitle ölçeklenir (emniyet).

UYARI: Karıştırma yalnızca yetkili test ortamında (Faraday kafesi / düşük güç)
ve yasal izinle yapılır.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import firwin, lfilter

# FIR alçak geçiren filtre uzunluğu (bant sınırlama için).
_FIR_NUM_TAPS = 129
# Sayısal kararlılık için çok küçük güç tabanı.
_EPS = 1e-12


def _normalize_peak(iq: np.ndarray, amplitude: float) -> np.ndarray:
    """IQ'yu tepe genliği ``amplitude`` olacak şekilde ölçekle (complex64)."""
    peak = float(np.max(np.abs(iq))) if iq.size else 0.0
    if peak > _EPS:
        iq = iq * (amplitude / peak)
    return iq.astype(np.complex64)


def _normalize_rms(iq: np.ndarray, amplitude: float) -> np.ndarray:
    """IQ'yu RMS genliği ``amplitude`` olacak şekilde ölçekle (complex64).

    Gürültü tabanlı sinyaller için tepe yerine RMS normalizasyonu uygundur
    (tepe değerleri istatistikseldir).
    """
    rms = float(np.sqrt(np.mean(np.abs(iq) ** 2))) if iq.size else 0.0
    if rms > _EPS:
        iq = iq * (amplitude / rms)
    return iq.astype(np.complex64)


def barrage_jam(
    num_samples: int,
    sample_rate: float,
    bandwidth_hz: float,
    amplitude: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """Bant sınırlı karmaşık AWGN (baraj karıştırma).

    Beyaz karmaşık gürültü üretilir ve ``bandwidth_hz`` genişliğine (baseband,
    ±bandwidth/2) FIR alçak geçiren ile sınırlanır.

    Args:
        num_samples: Üretilecek örnek sayısı.
        sample_rate: Örnekleme hızı (Hz).
        bandwidth_hz: Karıştırma bant genişliği (Hz). 0 < BW < sample_rate.
        amplitude: Hedef RMS genliği (0..1).
        seed: Tekrarlanabilirlik için RNG tohumu (None → rastgele).

    Returns:
        ``complex64`` IQ dizisi (uzunluk ``num_samples``), RMS = amplitude.
    """
    if num_samples <= 0:
        raise ValueError("num_samples pozitif olmalı.")
    if not 0.0 < bandwidth_hz < sample_rate:
        raise ValueError("0 < bandwidth_hz < sample_rate olmalı.")

    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(num_samples) + 1j * rng.standard_normal(num_samples)

    # Baseband bant sınırlama: kesim = (BW/2) / Nyquist = BW / sample_rate.
    cutoff = bandwidth_hz / sample_rate  # 0..1 (Nyquist'e oranlı)
    taps = firwin(_FIR_NUM_TAPS, cutoff)
    filtered = lfilter(taps, 1.0, noise)
    return _normalize_rms(filtered, amplitude)


def multi_tone_jam(
    num_samples: int,
    sample_rate: float,
    tone_offsets_hz: list[float] | np.ndarray,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Çoklu CW ton karıştırma (belirli kanalları nokta karıştırma).

    Verilen baseband ofset frekanslarında eşit genlikli karmaşık tonların
    toplamı. Toplam tepe genliği ``amplitude``'a normalize edilir.

    Args:
        num_samples: Üretilecek örnek sayısı.
        sample_rate: Örnekleme hızı (Hz).
        tone_offsets_hz: Merkez frekansa göre ton ofsetleri (Hz). |ofset| < Nyquist.
        amplitude: Hedef tepe genliği (0..1).

    Returns:
        ``complex64`` IQ dizisi (uzunluk ``num_samples``).
    """
    if num_samples <= 0:
        raise ValueError("num_samples pozitif olmalı.")
    offsets = np.asarray(tone_offsets_hz, dtype=np.float64)
    if offsets.size == 0:
        raise ValueError("En az bir ton ofseti gerekli.")
    nyq = sample_rate / 2.0
    if np.any(np.abs(offsets) >= nyq):
        raise ValueError("Ton ofsetleri |f| < Nyquist olmalı.")

    t = np.arange(num_samples, dtype=np.float64) / sample_rate
    iq = np.zeros(num_samples, dtype=np.complex128)
    for f_off in offsets:
        iq += np.exp(2j * np.pi * f_off * t)
    return _normalize_peak(iq, amplitude)


def sweep_jam(
    num_samples: int,
    sample_rate: float,
    f_start_hz: float,
    f_stop_hz: float,
    sweep_time_s: float,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Doğrusal chirp (taramalı/sweep karıştırma).

    Anlık frekans, ``sweep_time_s`` boyunca ``f_start_hz``'ten ``f_stop_hz``'e
    doğrusal artar ve testere dişi gibi tekrar eder. Sabit zarflı (yalnız faz
    modülasyonu) → tüm bant boyunca eşit güç süpürülür.

    Args:
        num_samples: Üretilecek örnek sayısı.
        sample_rate: Örnekleme hızı (Hz).
        f_start_hz, f_stop_hz: Süpürme baseband frekans sınırları (Hz).
        sweep_time_s: Tek süpürme süresi (s).
        amplitude: Sabit zarf genliği (0..1).

    Returns:
        ``complex64`` IQ dizisi (uzunluk ``num_samples``), |IQ| = amplitude.
    """
    if num_samples <= 0:
        raise ValueError("num_samples pozitif olmalı.")
    if sweep_time_s <= 0.0:
        raise ValueError("sweep_time_s pozitif olmalı.")
    nyq = sample_rate / 2.0
    if abs(f_start_hz) >= nyq or abs(f_stop_hz) >= nyq:
        raise ValueError("Süpürme frekansları |f| < Nyquist olmalı.")

    t = np.arange(num_samples, dtype=np.float64) / sample_rate
    # Testere dişi süpürme fazı: her periyotta f_start→f_stop.
    tau = np.mod(t, sweep_time_s)  # periyot içindeki zaman
    rate = (f_stop_hz - f_start_hz) / sweep_time_s  # Hz/s
    inst_freq = f_start_hz + rate * tau
    # Faz = 2π ∫ f dt = 2π (f_start·tau + 0.5·rate·tau²); periyot başında sıfırlanır.
    phase = 2.0 * np.pi * (f_start_hz * tau + 0.5 * rate * tau**2)
    iq = amplitude * np.exp(1j * phase)
    # inst_freq yalnız dokümantasyon/doğrulama amaçlı; çıktı sabit zarflıdır.
    del inst_freq
    return iq.astype(np.complex64)
