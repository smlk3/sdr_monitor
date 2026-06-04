"""Gerçek LimeSDR donanımı (SoapySDR, driver='lime').

SoapySDR Python bağlaması kurulu olmayabilir; bu modül donanım/sürücü olmadan
da IMPORT EDİLEBİLİR kalır. Eksiklik yalnızca ``open()`` çağrıldığında anlamlı
bir hata olarak ortaya çıkar.

Donanım tuzakları (CLAUDE.md):
- Stream formatı CS16 istenir, içeride float32/complex64'e çevrilir (USB 2x).
- Faz kalibrasyonu LO retune sonrası geçersizdir (bu iskelette DF yok).
"""

from __future__ import annotations

import numpy as np

from .sdr_base import SDRBase

# SoapySDR'ı yumuşak (soft) içe aktar: yoksa modül yine de yüklenir.
try:
    import SoapySDR  # type: ignore
    from SoapySDR import SOAPY_SDR_CS16, SOAPY_SDR_RX  # type: ignore

    _SOAPY_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - ortam bağımlı
    SoapySDR = None  # type: ignore
    SOAPY_SDR_CS16 = None  # type: ignore
    SOAPY_SDR_RX = None  # type: ignore
    _SOAPY_IMPORT_ERROR = exc

# CS16 tam ölçek: int16 aralığı [-32768, 32767]. Normalize için 2^15.
_CS16_FULL_SCALE = 32768.0


class LimeSDR(SDRBase):
    """SoapySDR üzerinden LimeSDR RX akışı."""

    def __init__(
        self,
        frequency: float = 433.0e6,
        sample_rate: float = 10.0e6,
        gain_db: float = 30.0,
        channel: int = 0,
    ) -> None:
        self._freq = float(frequency)
        self._rate = float(sample_rate)
        self._gain = float(gain_db)
        self._channel = int(channel)
        self._dev = None
        self._stream = None
        # CS16 ara tampon: kanal başına 2 int16 (I, Q) → interleaved int16.
        self._buf: np.ndarray | None = None

    # --- Yaşam döngüsü ---

    def open(self) -> None:
        """Cihazı bul, ayarla ve RX akışını başlat.

        SoapySDR yoksa veya cihaz bulunamazsa anlamlı bir istisna fırlatır.
        """
        if SoapySDR is None:
            raise RuntimeError(
                "SoapySDR Python bağlaması bulunamadı. LimeSDR kullanmak için "
                "LimeSuite + SoapySDR (python3-soapysdr) kurun veya --mock ile "
                f"çalıştırın. Orijinal hata: {_SOAPY_IMPORT_ERROR!r}"
            )
        try:
            self._dev = SoapySDR.Device({"driver": "lime"})
        except Exception as exc:
            raise RuntimeError(
                f"LimeSDR açılamadı (driver='lime'). Cihaz bağlı mı? Hata: {exc}"
            ) from exc

        ch = self._channel
        self._dev.setSampleRate(SOAPY_SDR_RX, ch, self._rate)
        self._dev.setFrequency(SOAPY_SDR_RX, ch, self._freq)
        self._dev.setGain(SOAPY_SDR_RX, ch, self._gain)

        # CS16 stream iste (USB throughput için), kanal listesi [ch].
        self._stream = self._dev.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CS16, [ch])
        self._dev.activateStream(self._stream)

    def close(self) -> None:
        """Akışı durdur ve cihazı serbest bırak (idempotent)."""
        try:
            if self._dev is not None and self._stream is not None:
                self._dev.deactivateStream(self._stream)
                self._dev.closeStream(self._stream)
        finally:
            self._stream = None
            self._dev = None
            self._buf = None

    # --- Ayarlar ---

    def set_frequency(self, freq_hz: float) -> None:
        self._freq = float(freq_hz)
        if self._dev is not None:
            self._dev.setFrequency(SOAPY_SDR_RX, self._channel, self._freq)

    def set_sample_rate(self, rate_hz: float) -> None:
        self._rate = float(rate_hz)
        if self._dev is not None:
            self._dev.setSampleRate(SOAPY_SDR_RX, self._channel, self._rate)

    def set_gain(self, gain_db: float) -> None:
        self._gain = float(gain_db)
        if self._dev is not None:
            self._dev.setGain(SOAPY_SDR_RX, self._channel, self._gain)

    @property
    def frequency(self) -> float:
        return self._freq

    @property
    def sample_rate(self) -> float:
        return self._rate

    # --- Üretim ---

    def read_samples(self, n: int) -> np.ndarray:
        """``n`` IQ örneği oku, CS16'dan complex64'e çevirip döndür.

        readStream interleaved CS16 (I0,Q0,I1,Q1,...) doldurur; kısmi okumalar
        döngüde tamamlanır.
        """
        if self._dev is None or self._stream is None:
            raise RuntimeError("LimeSDR açık değil; önce open() çağırın.")

        if self._buf is None or self._buf.shape[0] != 2 * n:
            # interleaved int16 tampon: 2*n eleman (I/Q çiftleri)
            self._buf = np.empty(2 * n, dtype=np.int16)

        got = 0
        # readStream kanal başına bir tampon listesi bekler.
        while got < n:
            view = self._buf[2 * got : 2 * n]
            sr = self._dev.readStream(self._stream, [view], n - got, timeoutUs=int(1e6))
            if sr.ret > 0:
                got += sr.ret
            elif sr.ret < 0:
                # Negatif: timeout/overflow vb. Bu iskelette sıfırla doldur.
                break

        # CS16 interleaved → complex64, [-1, 1) aralığına normalize.
        i = self._buf[0 : 2 * n : 2].astype(np.float32)
        q = self._buf[1 : 2 * n : 2].astype(np.float32)
        iq = (i + 1j * q) / _CS16_FULL_SCALE
        return iq.astype(np.complex64)
