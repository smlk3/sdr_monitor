"""Soyut SDR arayüzü.

Tüm SDR uygulamaları (MockSDR, LimeSDR) bu arayüzü gerçekler. UI ve worker
katmanları yalnızca bu soyut tipe bağımlıdır; somut donanımı bilmezler.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class SDRBase(ABC):
    """SDR cihazları için soyut taban sınıf.

    Örnek formatı her zaman ``numpy.complex64`` (IQ). Alt sınıflar donanım
    formatını (ör. CS16) içeride float32'ye çevirmekle yükümlüdür.
    """

    @abstractmethod
    def open(self) -> None:
        """Cihazı aç ve akışı başlatmaya hazır hale getir.

        Donanım yoksa veya açılamazsa anlamlı bir istisna fırlatır.
        """

    @abstractmethod
    def close(self) -> None:
        """Cihazı kapat ve kaynakları serbest bırak (idempotent)."""

    @abstractmethod
    def set_frequency(self, freq_hz: float) -> None:
        """Merkez frekansını (Hz) ayarla."""

    @abstractmethod
    def set_sample_rate(self, rate_hz: float) -> None:
        """Örnekleme hızını (Hz) ayarla."""

    @abstractmethod
    def set_gain(self, gain_db: float) -> None:
        """RX kazancını (dB) ayarla."""

    @abstractmethod
    def read_samples(self, n: int) -> np.ndarray:
        """``n`` adet IQ örneği oku.

        Dönüş: ``numpy.complex64`` türünde, uzunluğu ``n`` olan dizi.
        """

    def read_samples_dual(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """İki RX kanalından EŞ ZAMANLI ``n`` IQ örneği oku (koherent).

        Yön bulma (DF) faz interferometrisi için iki kanal aynı anda
        örneklenmelidir. Varsayılan uygulama desteklenmediğini bildirir; çift
        kanal sunan alt sınıflar (MockSDR, iki kanallı LimeSDR) bunu gerçekler.

        Dönüş: ``(rx0, rx1)`` — her biri ``complex64``, uzunluğu ``n``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} çift kanal (DF) okumayı desteklemiyor."
        )

    # --- TX (Elektronik Taarruz) ---
    # NOT (emniyet): TX yalnız RF switch sıralaması (RX durdur → switch → guard)
    # tamamlandıktan sonra ve yazılımsal güç limiti uygulanarak çağrılmalıdır.
    # Bu metotlar donanımı yalnız sürer; emniyet mantığı rf_switch/tx_worker'da.

    def start_tx(self, freq_hz: float, sample_rate: float, gain_db: float) -> None:
        """TX akışını başlat. Çift kanal gibi, desteklemeyen SDR hata verir."""
        raise NotImplementedError(
            f"{type(self).__name__} TX'i desteklemiyor."
        )

    def write_samples(self, iq: np.ndarray) -> int:
        """``iq`` (complex64) bloğunu TX akışına yaz; yazılan örnek sayısı döner."""
        raise NotImplementedError(
            f"{type(self).__name__} TX'i desteklemiyor."
        )

    def stop_tx(self) -> None:
        """TX akışını durdur (idempotent)."""
        raise NotImplementedError(
            f"{type(self).__name__} TX'i desteklemiyor."
        )

    # --- Bilgi erişimcileri (alt sınıflar geçerli değerleri tutar) ---

    @property
    @abstractmethod
    def frequency(self) -> float:
        """Geçerli merkez frekansı (Hz)."""

    @property
    @abstractmethod
    def sample_rate(self) -> float:
        """Geçerli örnekleme hızı (Hz)."""

    # --- Bağlam yöneticisi kolaylığı ---

    def __enter__(self) -> SDRBase:
        self.open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
