"""Sentetik IQ üreten MockSDR.

Donanım olmadan tüm UI/DSP zincirinin çalışabilmesi için gerçekçi bir spektrum
üretir. Tespit ve sınıflandırmanın görünür olması için birbirinden ayrık,
belirgin sinyaller konur:
- 3 adet CW taşıyıcı (dar, sabit zarf → "analog"),
- 1 dar bant FM benzeri öbek (sabit zarf, frekans modüleli → "analog"),
- 1 dijital benzeri (QPSK) öbek (genlik geçişli, geniş bant → "dijital").

Frekans ve örnekleme hızı değişimlerine tepki verir (sinyaller bant içinde kayar).
"""

from __future__ import annotations

import time

import numpy as np

from .sdr_base import SDRBase

# Sahte sinyallerin merkez frekansa göre ofsetleri (Hz) ve göreli güçleri.
# Bunlar mutlak frekanslar DEĞİL; her zaman bant içinde görünmeleri için
# örnekleme hızına oranlanır (Nyquist bandının kesri).
_CW_OFFSETS_FRAC = (-0.30, 0.12, 0.35)
_CW_AMPLITUDES = (0.7, 1.0, 0.5)
# Dar bant FM: sinüzoidal (sınırlı) mesaj → sabit zarf, düzgün anlık frekans.
_NBFM_OFFSET_FRAC = -0.08      # merkez ofset (Nyquist kesri)
_NBFM_MSG_FREQ_FRAC = 0.005    # mesaj frekansı / örnekleme hızı (yavaş)
_NBFM_DEV_FRAC = 0.004         # tepe frekans sapması / örnekleme hızı (modindex<1)
_NBFM_AMPLITUDE = 1.0
# Dijital (QPSK) öbek: merkez ofset, sembol hızı (band kesri) ve genliği.
# Sembol başına faz SIÇRAMALARI (düzgün FM'in aksine) → "dijital" sınıfı.
# Kök-yükseltilmiş-kosinüs (RRC) darbe şekillendirme: kompakt spektrum, yan
# loblar bastırılmış (dikdörtgen darbenin geniş bant saçılması olmadan).
_QPSK_OFFSET_FRAC = 0.55
# Dar bir dijital sinyal: CA-CFAR'ın kendini-maskelemesini önlemek için işgal
# edilen bant, eğitim penceresinden dar tutulur; yine de birkaç sembol geçişi
# "dijital" imzayı korur.
_QPSK_SYMRATE_FRAC = 0.01      # sembol hızı / örnekleme hızı
_QPSK_AMPLITUDE = 1.2
_QPSK_RRC_BETA = 0.35          # RRC rolloff faktörü
_QPSK_RRC_SPAN = 8            # RRC filtre uzunluğu (sembol)
_NOISE_SIGMA = 0.05            # AWGN standart sapması (IQ başına)


def _rrc_taps(beta: float, sps: int, span: int) -> np.ndarray:
    """Kök-yükseltilmiş-kosinüs (RRC) filtre katsayıları (enerjiye normalize).

    Args:
        beta: Rolloff faktörü (0..1).
        sps: Sembol başına örnek.
        span: Filtre uzunluğu (sembol cinsinden).
    """
    n_taps = span * sps
    t = (np.arange(n_taps + 1) - n_taps / 2) / sps
    h = np.zeros_like(t)
    for i, x in enumerate(t):
        if abs(x) < 1e-8:
            h[i] = 1.0 - beta + 4.0 * beta / np.pi
        elif beta > 0 and abs(abs(x) - 1.0 / (4.0 * beta)) < 1e-8:
            h[i] = (beta / np.sqrt(2.0)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
            )
        else:
            h[i] = (
                np.sin(np.pi * x * (1.0 - beta))
                + 4.0 * beta * x * np.cos(np.pi * x * (1.0 + beta))
            ) / (np.pi * x * (1.0 - (4.0 * beta * x) ** 2))
    return h / np.sqrt(np.sum(h**2))


class MockSDR(SDRBase):
    """Sentetik IQ üreteci. ``SDRBase`` arayüzünü tam gerçekler."""

    def __init__(
        self,
        frequency: float = 433.0e6,
        sample_rate: float = 10.0e6,
        gain_db: float = 30.0,
    ) -> None:
        self._freq = float(frequency)
        self._rate = float(sample_rate)
        self._gain = float(gain_db)
        self._opened = False
        self._phase = 0.0  # örnek blokları arasında faz sürekliliği
        self._rng = np.random.default_rng()

    # --- Yaşam döngüsü ---

    def open(self) -> None:
        self._opened = True

    def close(self) -> None:
        self._opened = False

    # --- Ayarlar ---

    def set_frequency(self, freq_hz: float) -> None:
        self._freq = float(freq_hz)

    def set_sample_rate(self, rate_hz: float) -> None:
        self._rate = float(rate_hz)

    def set_gain(self, gain_db: float) -> None:
        self._gain = float(gain_db)

    @property
    def frequency(self) -> float:
        return self._freq

    @property
    def sample_rate(self) -> float:
        return self._rate

    # --- Üretim ---

    def read_samples(self, n: int) -> np.ndarray:
        """``n`` örneklik sentetik IQ bloğu üret.

        Gerçek donanım okuma süresini taklit etmek için kısa bir uyku eklenir,
        böylece worker CPU'yu %100 meşgul etmez ve FPS gerçekçi kalır.
        """
        if not self._opened:
            raise RuntimeError("MockSDR açık değil; önce open() çağırın.")

        rate = self._rate
        t = (np.arange(n, dtype=np.float64) + self._phase) / rate

        iq = np.zeros(n, dtype=np.complex128)

        # Kazancı doğrusal bir ölçeğe çevir (görsel olarak kazancın etkisini
        # spektrumda görmek için). Referans 30 dB.
        gain_lin = 10.0 ** ((self._gain - 30.0) / 20.0)

        # CW taşıyıcılar — baseband ofset frekansları örnekleme hızına oranlı.
        nyq = rate / 2.0
        for frac, amp in zip(_CW_OFFSETS_FRAC, _CW_AMPLITUDES, strict=True):
            f_off = frac * nyq
            iq += amp * np.exp(2j * np.pi * f_off * t)

        # Dar bant FM: sinüzoidal mesajla SINIRLI sapmalı, sabit zarflı FM.
        # Faz, sürekli küresel zaman ``t`` üzerinden üretildiğinden bloklar
        # arasında kendiliğinden süreklidir (ayrı faz takibi gerekmez).
        f_center = _NBFM_OFFSET_FRAC * nyq
        f_msg = _NBFM_MSG_FREQ_FRAC * rate
        dev = _NBFM_DEV_FRAC * rate
        # FM faz: integral(2π·dev·cos) = (dev/f_msg)·sin → sınırlı modülasyon.
        nbfm_inst_phase = 2.0 * np.pi * f_center * t + (dev / f_msg) * np.sin(
            2.0 * np.pi * f_msg * t
        )
        iq += _NBFM_AMPLITUDE * np.exp(1j * nbfm_inst_phase)

        # Dijital QPSK öbek: rastgele 4 fazlı semboller, RRC darbe ile
        # şekillendirilir → kompakt bant. Sembol geçişlerindeki ani faz
        # sıçramaları sınıflandırmada "dijital" imzası verir.
        f_qpsk = _QPSK_OFFSET_FRAC * nyq
        sps = max(int(round(1.0 / _QPSK_SYMRATE_FRAC)), 1)  # örnek/sembol
        n_sym = n // sps + _QPSK_RRC_SPAN + 1
        sym = self._rng.integers(0, 4, size=n_sym)
        sym_iq = np.exp(1j * (np.pi / 4 + sym * (np.pi / 2)))  # QPSK takımyıldızı
        upsampled = np.zeros(n_sym * sps, dtype=np.complex128)
        upsampled[::sps] = sym_iq
        taps = _rrc_taps(_QPSK_RRC_BETA, sps, _QPSK_RRC_SPAN)
        shaped = np.convolve(upsampled, taps, mode="same")[:n]
        iq += _QPSK_AMPLITUDE * shaped * np.exp(2j * np.pi * f_qpsk * t)

        # AWGN
        noise = self._rng.standard_normal(n) + 1j * self._rng.standard_normal(n)
        iq += _NOISE_SIGMA * noise

        iq *= gain_lin

        # Faz sürekliliği için örnek sayacını ilerlet.
        self._phase += n

        # Gerçekçi throughput taklidi: blok süresinin bir kısmı kadar bekle.
        # Tam blok süresi beklemek FPS'i düşürür; yarısı yeterli gerçekçilik.
        time.sleep(min(0.5 * n / rate, 0.02))

        return iq.astype(np.complex64)
