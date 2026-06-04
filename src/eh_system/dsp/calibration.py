"""İki kanal faz/zaman kalibrasyonu.

İki RX kanalı arasındaki donanımsal uyumsuzluğu (kablo uzunluğu, ADC zaman
kayması, sabit faz farkı) kestirir ve giderir. Yön bulma (DF) faz
interferometrisinin doğru çalışması için kanalların eşlenmesi şarttır.

Model:  rx1[n] ≈ A·exp(j·φ)·rx0[n − k]
    k = örnek kayması (tam sayı, çapraz korelasyonla bulunur),
    φ = sabit faz farkı (rad).

Kalibrasyon, geliş açısı bilinen bir REFERANS kaynakla (tercihen broadside,
θ=0) yapılmalıdır; aksi halde kestirilen faz, geometrik (yön) bileşeni de
içerir. CLAUDE.md: faz kalibrasyonu HER LO retune sonrası geçersizdir
(``calibration_required`` bayrağı).

Saf numpy/scipy; Qt/UI bilmez, yan etkisizdir.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import correlate, correlation_lags


def phase_calibration(
    rx0: np.ndarray,
    rx1: np.ndarray,
    max_shift: int | None = None,
) -> tuple[int, float]:
    """İki kanal arası örnek kayması ve sabit faz farkını kestir.

    Önce ``rx1`` ile ``rx0``'ın çapraz korelasyonunun tepe noktasından tam sayı
    örnek kaymasını bulur; ardından kaymayı hizalayıp kalan sabit faz farkını
    ölçer.

    Args:
        rx0: Referans kanal IQ (1B karmaşık).
        rx1: İkinci kanal IQ (rx0 ile aynı uzunluk).
        max_shift: Aranan kaymayı ±bu değerle sınırla (None → tüm aralık).

    Returns:
        (sample_shift, phase_offset):
            sample_shift: rx1'in rx0'a göre gecikmesi (örnek, tam sayı).
            phase_offset: rx1'in rx0'a göre sabit faz farkı (rad).
    """
    if rx0.shape != rx1.shape:
        raise ValueError("rx0 ve rx1 aynı uzunlukta olmalı.")
    if rx0.shape[0] == 0:
        raise ValueError("Boş giriş.")

    corr = correlate(rx1, rx0, mode="full")
    lags = correlation_lags(rx1.shape[0], rx0.shape[0], mode="full")

    mag = np.abs(corr)
    if max_shift is not None:
        # Aranan kaymayı sınırla: izin verilen lag dışını maskele.
        mask = np.abs(lags) <= int(max_shift)
        mag = np.where(mask, mag, -np.inf)

    peak = int(np.argmax(mag))
    sample_shift = int(lags[peak])
    # Tepe noktasındaki karmaşık korelasyon değerinin açısı = sabit faz farkı.
    phase_offset = float(np.angle(corr[peak]))
    return sample_shift, phase_offset


def apply_calibration(
    iq: np.ndarray,
    sample_shift: int,
    phase_offset: float,
) -> np.ndarray:
    """Kalibrasyon düzeltmesini ikinci kanal IQ'suna uygula.

    ``phase_calibration`` ile bulunan kayma ve fazı tersine çevirerek ``rx1``'i
    ``rx0`` ile hizalar:  rx1_düz[n] = rx1[n + k]·exp(−j·φ).

    Not: Kayma ``numpy.roll`` ile dairesel uygulanır (blok kenarında küçük
    sarma); akış işlemede blok boyu kaymaya göre büyük olduğundan etki ihmal
    edilebilir.

    Args:
        iq: Düzeltilecek kanal (genelde rx1) IQ örnekleri.
        sample_shift: ``phase_calibration``'dan dönen örnek kayması.
        phase_offset: ``phase_calibration``'dan dönen faz farkı (rad).

    Returns:
        Hizalanmış IQ (giriş ile aynı uzunluk/tip).
    """
    shifted = np.roll(iq, -int(sample_shift))
    corrected = shifted * np.exp(-1j * float(phase_offset))
    return corrected.astype(iq.dtype)
