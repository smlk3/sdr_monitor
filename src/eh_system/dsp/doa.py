"""Yön bulma (DF): iki kanallı faz interferometri.

İki koherent RX kanalı arasındaki sabit faz farkından, bilinen anten ayrımı
(baseline) ve frekansla geliş açısını (DoA) kestirir.

ÖNEMLİ KISIT — TEK KAYNAK:
    Faz interferometri, bant içinde BASKIN TEK BİR düzlem dalga olduğunu varsayar.
    Birden çok eşzamanlı kaynak varsa ölçülen faz, kaynakların vektörel
    toplamıdır ve sonuç anlamsızdır. Çoklu kaynak ayrımı (ör. MUSIC) bu modülün
    kapsamı dışındadır. Kullanım: önce CFAR ile tek baskın sinyal yalıtılır,
    sonra DoA o bant için hesaplanır.

Tek bir baz çizgisi (baseline) ön/arka (front/back) belirsizliğini çözemez;
dönüş açısı broadside'a (diziye dik eksen) göre [-90, +90] derecedir.

Saf numpy; Qt/UI bilmez, yan etkisizdir.
"""

from __future__ import annotations

import warnings

import numpy as np

# Işık hızı (m/s).
_C = 299_792_458.0


def phase_difference(rx0: np.ndarray, rx1: np.ndarray) -> float:
    """İki kanal arasındaki ortalama (koherent) faz farkını (rad) döndür.

    ``rx1`` referans ``rx0``'a göre ``exp(j·Δφ)`` ile döner kabul edilir. Faz,
    ``sum(conj(rx0)·rx1)`` karmaşık toplamının açısı olarak SNR-ağırlıklı
    biçimde kestirilir (örnek-örnek açı ortalamasından gürültüye daha dayanıklı).

    Args:
        rx0: Referans kanal IQ (1B karmaşık).
        rx1: İkinci kanal IQ (rx0 ile aynı uzunluk).

    Returns:
        Faz farkı (rad), (-π, π].
    """
    if rx0.shape != rx1.shape:
        raise ValueError("rx0 ve rx1 aynı uzunlukta olmalı.")
    if rx0.shape[0] == 0:
        raise ValueError("Boş giriş.")
    cross = np.vdot(rx0, rx1)  # sum(conj(rx0) * rx1)
    return float(np.angle(cross))


def doa_phase_interferometry(
    rx0: np.ndarray,
    rx1: np.ndarray,
    baseline_m: float,
    freq_hz: float,
) -> float:
    """Faz interferometri ile geliş açısını (derece) kestir.

    Δφ = 2π·d·sin(θ)/λ  →  sin(θ) = Δφ·λ / (2π·d) = Δφ·c / (2π·d·f)

    Anten ayrımı ``d`` λ/2'yi aşarsa faz sarması nedeniyle açı BELİRSİZ olur
    (grating/aliasing); bu durumda uyarı verilir. Gürültü nedeniyle |sin θ| > 1
    çıkarsa [-1, 1] aralığına kırpılır.

    Args:
        rx0: Referans kanal IQ (kalibre edilmiş).
        rx1: İkinci kanal IQ (kalibre edilmiş, rx0 ile aynı uzunluk).
        baseline_m: İki anten arası mesafe (m).
        freq_hz: Sinyal frekansı (Hz).

    Returns:
        Geliş açısı (derece), broadside'a göre [-90, +90].
    """
    if baseline_m <= 0.0:
        raise ValueError("baseline_m pozitif olmalı.")
    if freq_hz <= 0.0:
        raise ValueError("freq_hz pozitif olmalı.")

    lam = _C / freq_hz
    if baseline_m > lam / 2.0:
        warnings.warn(
            f"Anten ayrımı d={baseline_m:.3f} m > λ/2={lam / 2.0:.3f} m: faz "
            "sarması nedeniyle açı belirsiz (aliasing). d ≤ λ/2 önerilir.",
            stacklevel=2,
        )

    delta_phi = phase_difference(rx0, rx1)
    sin_theta = delta_phi * lam / (2.0 * np.pi * baseline_m)
    sin_theta = float(np.clip(sin_theta, -1.0, 1.0))
    return float(np.degrees(np.arcsin(sin_theta)))


def is_aliasing(baseline_m: float, freq_hz: float) -> bool:
    """Verilen geometride λ/2 aşılıyor mu (açı belirsiz olur mu)?"""
    if baseline_m <= 0.0 or freq_hz <= 0.0:
        raise ValueError("baseline_m ve freq_hz pozitif olmalı.")
    return baseline_m > (_C / freq_hz) / 2.0
