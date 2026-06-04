"""SDR RX worker: QThread içinde read_samples -> compute_psd döngüsü.

CLAUDE.md kuralları:
- Ağır iş (SDR I/O, FFT) ASLA GUI thread'inde çalışmaz.
- Worker -> UI yalnızca Qt signal/slot ile; numpy array'ler .copy() ile geçer.
- UI -> Worker thread-safe (queue.Queue) komut kanalı ile.
- Her donanım komutu try/except ile sarılır; hata signal ile bildirilir, çökme yok.
"""

from __future__ import annotations

import queue
import time
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QThread, Signal

from ..dsp.cfar import ca_cfar
from ..dsp.params import extract_signals
from ..dsp.spectrum import compute_psd
from ..hal.sdr_base import SDRBase
from .signal_tracker import SignalTracker

# FPS ölçümü için kayan ortalama penceresi (kare sayısı).
_FPS_WINDOW = 15

# CFAR varsayılanları (UI'dan ayarlanabilir).
_DEFAULT_NUM_TRAIN = 16
_DEFAULT_NUM_GUARD = 4
_DEFAULT_PFA = 1e-4
# Tek sinyalin küçük güç düşüşleriyle parçalanan adalarını birleştirme yarıçapı.
_DETECTION_CLOSE_GAP = 2


@dataclass(frozen=True)
class _Command:
    """UI'dan worker'a thread-safe iletilen komut."""

    # "freq" | "gain" | "sample_rate" | "fft_size"
    # | "num_train" | "num_guard" | "pfa"
    kind: str
    value: float


class SDRWorker(QThread):
    """RX akışını okuyup PSD hesaplayan ve UI'ya emit eden worker thread'i."""

    # (freqs, psd_db) — her ikisi de bağımsız kopyalardır.
    spectrum_ready = Signal(np.ndarray, np.ndarray)
    # CFAR eşiği (threshold_db) — bağımsız kopya.
    cfar_ready = Signal(np.ndarray)
    # İzlenen sinyal listesi: list[TrackedSignal] (her kare yeniden üretilir).
    signals_ready = Signal(object)
    # Anlık FPS (float).
    fps_updated = Signal(float)
    # Kümülatif overflow/okuma-hatası sayacı (int).
    overflow_updated = Signal(int)
    # Bağlantı durumu metni.
    status_changed = Signal(str)
    # Hata mesajı (çökme yerine).
    error_occurred = Signal(str)

    def __init__(
        self,
        sdr: SDRBase,
        fft_size: int = 1024,
        read_size: int = 4096,
        window: str = "hann",
        num_train: int = _DEFAULT_NUM_TRAIN,
        num_guard: int = _DEFAULT_NUM_GUARD,
        pfa: float = _DEFAULT_PFA,
        cal_offset_db: float = 0.0,
        parent: object | None = None,
    ) -> None:
        super().__init__(parent)
        self._sdr = sdr
        self._fft_size = int(fft_size)
        self._read_size = max(int(read_size), self._fft_size)
        self._window = window
        self._cmd_queue: queue.Queue[_Command] = queue.Queue()
        self._running = False
        self._overflow_count = 0
        # CFAR + sinyal çıkarımı durumu.
        self._num_train = int(num_train)
        self._num_guard = int(num_guard)
        self._pfa = float(pfa)
        self._cal_offset_db = float(cal_offset_db)
        self._tracker = SignalTracker()

    # --- UI'dan çağrılan thread-safe ayar metotları (queue'ya yazar) ---

    def request_frequency(self, freq_hz: float) -> None:
        """Merkez frekans değişimini worker'a güvenle ilet."""
        self._cmd_queue.put(_Command("freq", float(freq_hz)))

    def request_gain(self, gain_db: float) -> None:
        """Kazanç değişimini worker'a güvenle ilet."""
        self._cmd_queue.put(_Command("gain", float(gain_db)))

    def request_sample_rate(self, rate_hz: float) -> None:
        """Örnekleme hızı değişimini worker'a güvenle ilet."""
        self._cmd_queue.put(_Command("sample_rate", float(rate_hz)))

    def request_fft_size(self, fft_size: int) -> None:
        """FFT boyutu değişimini worker'a güvenle ilet."""
        self._cmd_queue.put(_Command("fft_size", float(fft_size)))

    def request_num_train(self, num_train: int) -> None:
        """CFAR eğitim hücresi sayısını worker'a güvenle ilet."""
        self._cmd_queue.put(_Command("num_train", float(num_train)))

    def request_num_guard(self, num_guard: int) -> None:
        """CFAR koruma hücresi sayısını worker'a güvenle ilet."""
        self._cmd_queue.put(_Command("num_guard", float(num_guard)))

    def request_pfa(self, pfa: float) -> None:
        """CFAR yanlış alarm olasılığını worker'a güvenle ilet."""
        self._cmd_queue.put(_Command("pfa", float(pfa)))

    def stop(self) -> None:
        """Döngüyü temiz biçimde durdur ve thread'in bitmesini bekle."""
        self._running = False
        self.wait(2000)

    # --- Komut işleme ---

    def _drain_commands(self) -> None:
        """Bekleyen tüm komutları uygula. Yalnızca worker thread'inde çağrılır."""
        while True:
            try:
                cmd = self._cmd_queue.get_nowait()
            except queue.Empty:
                return
            try:
                if cmd.kind == "freq":
                    self._sdr.set_frequency(cmd.value)
                    # Frekans değişti → eski izler artık geçerli değil.
                    self._tracker.reset()
                    self.status_changed.emit(f"Frekans: {cmd.value / 1e6:.3f} MHz")
                elif cmd.kind == "gain":
                    self._sdr.set_gain(cmd.value)
                elif cmd.kind == "sample_rate":
                    self._sdr.set_sample_rate(cmd.value)
                    # Bant genişliği değişti → izleri sıfırla.
                    self._tracker.reset()
                elif cmd.kind == "fft_size":
                    new_size = int(cmd.value)
                    self._fft_size = new_size
                    self._read_size = max(self._read_size, new_size)
                elif cmd.kind == "num_train":
                    self._num_train = max(1, int(cmd.value))
                elif cmd.kind == "num_guard":
                    self._num_guard = max(0, int(cmd.value))
                elif cmd.kind == "pfa":
                    self._pfa = float(cmd.value)
            except Exception as exc:  # donanım komutu — çökme yok
                self.error_occurred.emit(f"Ayar hatası ({cmd.kind}): {exc}")

    # --- Ana döngü ---

    def run(self) -> None:
        """Thread giriş noktası: aç, döngüde oku+işle+emit, kapat."""
        self._running = True
        try:
            self._sdr.open()
            self.status_changed.emit("Bağlı")
        except Exception as exc:
            self.error_occurred.emit(f"SDR açılamadı: {exc}")
            self.status_changed.emit("Bağlantı yok")
            return

        frame_times: list[float] = []
        try:
            while self._running:
                self._drain_commands()

                t0 = time.perf_counter()
                try:
                    iq = self._sdr.read_samples(self._read_size)
                except Exception as exc:
                    self._overflow_count += 1
                    self.overflow_updated.emit(self._overflow_count)
                    self.error_occurred.emit(f"Okuma hatası: {exc}")
                    continue

                if iq.shape[0] < self._fft_size:
                    self._overflow_count += 1
                    self.overflow_updated.emit(self._overflow_count)
                    continue

                try:
                    freqs, psd_db = compute_psd(
                        iq,
                        self._fft_size,
                        sample_rate=self._sdr.sample_rate,
                        center_freq=self._sdr.frequency,
                        window=self._window,
                    )
                except Exception as exc:
                    self.error_occurred.emit(f"PSD hatası: {exc}")
                    continue

                # Array'ler kopyalanarak geçer (paylaşılan bellek yok).
                self.spectrum_ready.emit(freqs.copy(), psd_db.copy())

                # --- CFAR tespiti + sinyal parametre çıkarımı ---
                try:
                    detections, threshold_db = ca_cfar(
                        psd_db, self._num_train, self._num_guard, self._pfa
                    )
                    self.cfar_ready.emit(threshold_db.copy())

                    signals = extract_signals(
                        freqs,
                        psd_db,
                        detections,
                        iq=iq[: self._fft_size],
                        sample_rate=self._sdr.sample_rate,
                        center_freq=self._sdr.frequency,
                        cal_offset_db=self._cal_offset_db,
                        close_gap=_DETECTION_CLOSE_GAP,
                    )
                    tracked = self._tracker.update(signals, time.time())
                    self.signals_ready.emit(tracked)
                except Exception as exc:
                    self.error_occurred.emit(f"CFAR/parametre hatası: {exc}")

                # FPS kayan ortalama.
                dt = time.perf_counter() - t0
                frame_times.append(dt)
                if len(frame_times) > _FPS_WINDOW:
                    frame_times.pop(0)
                mean_dt = sum(frame_times) / len(frame_times)
                if mean_dt > 0:
                    self.fps_updated.emit(1.0 / mean_dt)
        finally:
            try:
                self._sdr.close()
            except Exception as exc:
                self.error_occurred.emit(f"Kapatma hatası: {exc}")
            self.status_changed.emit("Durduruldu")
