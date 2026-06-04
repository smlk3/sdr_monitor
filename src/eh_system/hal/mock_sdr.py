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

# --- İki kanallı (DF) ground-truth varsayılanları ---
# Işık hızı (m/s) — geometrik faz hesabı için.
_C = 299_792_458.0
# İkinci kanalın birinciye göre DONANIMSAL uyumsuzluğu (kalibrasyon hedefi):
# sabit örnek kayması + sabit faz farkı. "Kalibre" bunları ölçüp giderir.
_DF_SAMPLE_SHIFT = 3           # örnek kayması (ADC/kablo gecikmesi taklidi)
_DF_PHASE_OFFSET_RAD = 0.6     # sabit faz farkı (rad)
# Baskın kaynağın GERÇEK geliş açısı (derece) — kalibrasyon sonrası DoA'nın
# geri çıkarması beklenen değer. Geometrik faz = 2π·d·sin(θ)·f/c.
_DF_TRUE_DOA_DEG = 25.0
# İki anten arası varsayılan mesafe (m). 433 MHz'de λ/2 ≈ 0.346 m > 0.15 (güvenli).
_DF_BASELINE_M = 0.15
# Yönlü anten huzme genişliği (Gaussian σ, derece) — tarama açı-güç haritasında
# kaynak yönünde belirgin bir lob oluşması için. Motor yönü kaynaktan uzaklaştıkça
# alınan SİNYAL gücü düşer (gürültü sabit kalır → SNR düşer).
_DF_BEAMWIDTH_SIGMA_DEG = 35.0


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
        df_sample_shift: int = _DF_SAMPLE_SHIFT,
        df_phase_offset_rad: float = _DF_PHASE_OFFSET_RAD,
        df_true_doa_deg: float = _DF_TRUE_DOA_DEG,
        df_baseline_m: float = _DF_BASELINE_M,
    ) -> None:
        self._freq = float(frequency)
        self._rate = float(sample_rate)
        self._gain = float(gain_db)
        self._opened = False
        self._phase = 0.0  # örnek blokları arasında faz sürekliliği
        self._rng = np.random.default_rng()
        # İki kanallı (DF) ground-truth parametreleri.
        self._df_sample_shift = int(df_sample_shift)
        self._df_phase_offset_rad = float(df_phase_offset_rad)
        self._df_true_doa_deg = float(df_true_doa_deg)
        self._df_baseline_m = float(df_baseline_m)
        # Kalibrasyon referans modu: True iken ikinci kanala SADECE donanımsal
        # uyumsuzluk (faz+kayma) konur, geometrik (yön) faz konmaz — broadside
        # referans kaynağı taklidi. "Kalibre" sırasında worker bunu açar.
        self._cal_reference = False
        # Motor (anten) yönelim açısı (derece) — tarama sırasında worker ayarlar.
        # None iken yönlü kazanç uygulanmaz (gücün 1.0; canlı spektrum etkilenmez).
        self._pointing_deg: float | None = None

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

    def set_calibration_reference(self, on: bool) -> None:
        """Kalibrasyon referans modunu aç/kapat (yalnızca MockSDR).

        Açıkken ``read_samples_dual`` ikinci kanala SADECE donanımsal uyumsuzluk
        koyar (geometrik yön fazı yok) — broadside referans kaynağı taklidi.
        Böylece "Kalibre" donanımsal kaymayı/fazı izole edip ölçebilir; gerçek
        ölçümde (kapalı) kalan geometrik faz DoA olarak geri çıkar.
        """
        self._cal_reference = bool(on)

    def set_pointing_angle(self, deg: float | None) -> None:
        """Anten yönelim açısını ayarla (yalnızca MockSDR; tarama için).

        Yön bulma taraması her açıda bunu ayarlar; mock, kaynak yönüne
        (``df_true_doa_deg``) uzaklığa göre sinyal gücünü Gaussian huzme ile
        ölçekler → kutupsal açı-güç haritasında belirgin bir lob. ``None`` iken
        yönlü kazanç uygulanmaz (canlı spektrum etkilenmez).
        """
        self._pointing_deg = None if deg is None else float(deg)

    def _antenna_gain(self) -> float:
        """Yönelim açısına göre doğrusal anten kazancı (0..1)."""
        if self._pointing_deg is None:
            return 1.0
        # Kaynağa açısal uzaklığı [-180, 180]'e sar.
        diff = (self._pointing_deg - self._df_true_doa_deg + 180.0) % 360.0 - 180.0
        return float(np.exp(-0.5 * (diff / _DF_BEAMWIDTH_SIGMA_DEG) ** 2))

    # --- Üretim ---

    def _generate_clean(self, n: int) -> np.ndarray:
        """``n`` örneklik gürültüsüz, kazanç ölçekli sentetik IQ üret.

        Faz sürekliliği için örnek sayacını (``self._phase``) ilerletir.
        """
        rate = self._rate
        t = (np.arange(n, dtype=np.float64) + self._phase) / rate
        iq = np.zeros(n, dtype=np.complex128)

        # Kazancı doğrusal bir ölçeğe çevir (referans 30 dB).
        gain_lin = 10.0 ** ((self._gain - 30.0) / 20.0)

        # CW taşıyıcılar — baseband ofset frekansları örnekleme hızına oranlı.
        nyq = rate / 2.0
        for frac, amp in zip(_CW_OFFSETS_FRAC, _CW_AMPLITUDES, strict=True):
            f_off = frac * nyq
            iq += amp * np.exp(2j * np.pi * f_off * t)

        # Dar bant FM: sinüzoidal mesajla SINIRLI sapmalı, sabit zarflı FM.
        f_center = _NBFM_OFFSET_FRAC * nyq
        f_msg = _NBFM_MSG_FREQ_FRAC * rate
        dev = _NBFM_DEV_FRAC * rate
        nbfm_inst_phase = 2.0 * np.pi * f_center * t + (dev / f_msg) * np.sin(
            2.0 * np.pi * f_msg * t
        )
        iq += _NBFM_AMPLITUDE * np.exp(1j * nbfm_inst_phase)

        # Dijital QPSK öbek: RRC şekilli rastgele 4-PSK semboller.
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

        iq *= gain_lin
        # Yönlü anten kazancı (sinyale uygulanır; gürültü ayrıca eklenir).
        iq *= self._antenna_gain()
        # Faz sürekliliği için örnek sayacını ilerlet.
        self._phase += n
        return iq

    def _noise(self, n: int) -> np.ndarray:
        """``n`` örneklik karmaşık AWGN üret."""
        z = self._rng.standard_normal(n) + 1j * self._rng.standard_normal(n)
        return _NOISE_SIGMA * z

    def read_samples(self, n: int) -> np.ndarray:
        """``n`` örneklik sentetik IQ bloğu üret (tek kanal = kanal 0).

        Gerçek donanım okuma süresini taklit etmek için kısa bir uyku eklenir,
        böylece worker CPU'yu %100 meşgul etmez ve FPS gerçekçi kalır.
        """
        if not self._opened:
            raise RuntimeError("MockSDR açık değil; önce open() çağırın.")

        iq = self._generate_clean(n) + self._noise(n)
        time.sleep(min(0.5 * n / self._rate, 0.02))
        return iq.astype(np.complex64)

    def read_samples_dual(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """İki koherent kanal üret: rx1 = exp(j·Δφ)·rx0(n−k) + bağımsız gürültü.

        Kanal 0 baskın sinyaldir. Kanal 1, bilinen bir örnek kayması (k) ve faz
        farkı (Δφ) ile üretilir; bu fark DONANIMSAL uyumsuzluk + (referans modu
        kapalıyken) GEOMETRİK yön fazını içerir. Bilinen ground-truth sayesinde
        kalibrasyon ve DoA testlerinde beklenen değerler geri çıkarılabilir.

        Her kanala bağımsız AWGN eklenir (korelasyon ground-truth'u bozmaz ama
        gerçekçi gürültü tabanı sağlar).
        """
        if not self._opened:
            raise RuntimeError("MockSDR açık değil; önce open() çağırın.")

        clean = self._generate_clean(n)

        # Toplam faz farkı: donanımsal + (referans değilse) geometrik yön fazı.
        phi = self._df_phase_offset_rad
        if not self._cal_reference:
            lam = _C / self._freq
            sin_theta = np.sin(np.radians(self._df_true_doa_deg))
            phi += 2.0 * np.pi * self._df_baseline_m * sin_theta / lam

        rx0 = clean + self._noise(n)
        rx1 = (
            np.roll(clean, self._df_sample_shift) * np.exp(1j * phi)
            + self._noise(n)
        )
        time.sleep(min(0.5 * n / self._rate, 0.02))
        return rx0.astype(np.complex64), rx1.astype(np.complex64)
