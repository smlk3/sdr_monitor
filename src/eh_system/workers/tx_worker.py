"""TX worker: ring buffer'dan sürekli yayın (QThread) — EMNİYET KRİTİK.

CLAUDE.md emniyet kuralları (KATI):
- TX gücü yazılımsal üst limitle sınırlı; limit AŞILAMAZ (burada kırpılır).
- RF switch sıralaması ZORUNLU (rf_switch ile dayatılır): RX durdur → switch →
  ≥10 µs guard → TX. RX portuna TX yönlendirme engellenir.
- Acil durdur (emergency_stop): tek çağrıda TX kes (PA disable) < 500 ms.

Mimari:
- Ağır iş (TX I/O) GUI thread'inde ASLA çalışmaz; burada QThread.
- UI → worker thread-safe: ``set_waveform`` / ``set_power_db`` kilitle korunur.
- worker → UI yalnız Qt signal/slot.

Sürekli yayın bir dalga formu ring buffer'ından döngüyle beslenir (karıştırma
bloğu veya modüle edilmiş ses). Dalga formu yoksa sessizlik (sıfır) yazılır.
"""

from __future__ import annotations

import threading

import numpy as np
from PySide6.QtCore import QThread, Signal

from ..hal.rf_switch import RFSwitch, SwitchState
from ..hal.sdr_base import SDRBase

# Yayın blok boyutu (örnek).
_TX_BLOCK = 4096


class TXWorker(QThread):
    """Ring buffer'dan sürekli, güç-limitli TX yapan worker thread'i."""

    # TX durumu değişti (yayında mı?).
    tx_state_changed = Signal(bool)
    # Durum metni / hata.
    status_changed = Signal(str)
    error_occurred = Signal(str)
    # İstenen güç limiti aştı → kırpıldı (kırpılan dB değeri).
    power_capped = Signal(float)

    def __init__(
        self,
        sdr: SDRBase,
        rf_switch: RFSwitch,
        tx_freq: float,
        sample_rate: float,
        tx_gain_db: float,
        power_limit_db: float,
        block_size: int = _TX_BLOCK,
        parent: object | None = None,
    ) -> None:
        super().__init__(parent)
        self._sdr = sdr
        self._switch = rf_switch
        self._tx_freq = float(tx_freq)
        self._sample_rate = float(sample_rate)
        self._tx_gain_db = float(tx_gain_db)
        # Yazılımsal güç üst limiti — ASLA aşılamaz (emniyet).
        self._power_limit_db = float(power_limit_db)
        self._block_size = int(block_size)

        self._running = False
        self._emergency = False
        # Dalga formu (ring buffer) ve çıkış genliği kilit altında paylaşılır.
        self._lock = threading.Lock()
        self._waveform: np.ndarray | None = None
        self._ring_pos = 0
        # Çıkış genliği: 1.0 = güç limiti (tam ölçek). Varsayılan limitin altında.
        self._amplitude = 10.0 ** ((-10.0) / 20.0)  # limitin 10 dB altı

    # --- UI'dan çağrılan thread-safe ayarlar ---

    def set_waveform(self, iq: np.ndarray | None) -> None:
        """Yayınlanacak dalga formunu (ring buffer) ayarla (None → sessizlik)."""
        with self._lock:
            self._waveform = (
                None if iq is None else np.asarray(iq, dtype=np.complex64).copy()
            )
            self._ring_pos = 0

    def set_power_db(self, power_db: float) -> None:
        """Çıkış gücünü ayarla; yazılımsal limiti AŞAMAZ (kırpılır).

        Genlik = 10^((power_db − limit)/20), [0, 1]'e kırpılır → limit = tam
        ölçek. İstenen güç limiti aşarsa kırpılır ve ``power_capped`` yayılır.
        """
        capped = power_db
        if power_db > self._power_limit_db:
            capped = self._power_limit_db
            self.power_capped.emit(capped)
        amp = 10.0 ** ((capped - self._power_limit_db) / 20.0)
        with self._lock:
            self._amplitude = float(min(amp, 1.0))

    def stop(self) -> None:
        """Normal durdur: döngüyü bitir, güvenli RX'e dön, thread'i bekle."""
        self._running = False
        self.wait(3000)

    def emergency_stop(self) -> None:
        """ACİL DURDUR: PA'yı derhal kes (<500 ms), güvenli RX yoluna geç.

        rf_switch.emergency_disable() PA disable callback'ini SENKRON çağırır →
        RF çıkışı anında ölür (akış kapanmasını beklemez). Worker döngüsü
        bayrakla durur ve TX akışını kendi thread'inde kapatır.
        """
        self._emergency = True
        self._running = False
        # Donanım kesimi anında ve senkron (emniyet önceliği).
        try:
            self._switch.emergency_disable()
        except Exception as exc:  # acil yolda asla çökme
            self.error_occurred.emit(f"Acil durdur switch hatası: {exc}")
        self.status_changed.emit("ACİL DURDUR — TX kesildi")
        self.tx_state_changed.emit(False)

    # --- Yardımcı ---

    def _next_block(self) -> np.ndarray:
        """Ring buffer'dan sıradaki bloğu (genlik ölçekli) döndür."""
        with self._lock:
            wf = self._waveform
            amp = self._amplitude
            if wf is None or wf.shape[0] == 0:
                return np.zeros(self._block_size, dtype=np.complex64)
            n, pos, total = self._block_size, self._ring_pos, wf.shape[0]
            idx = (pos + np.arange(n)) % total
            self._ring_pos = (pos + n) % total
        return (wf[idx] * amp).astype(np.complex64)

    # --- Ana döngü ---

    def run(self) -> None:
        """Thread giriş: güvenli TX'e geç, ring buffer'ı sürekli yaz, kapat."""
        self._running = True
        self._emergency = False
        try:
            # ZORUNLU sıralama: RX durdur → switch → guard → TX (rf_switch dayatır).
            self._switch.engage_tx()
            self._sdr.start_tx(self._tx_freq, self._sample_rate, self._tx_gain_db)
            self.tx_state_changed.emit(True)
            self.status_changed.emit("TX yayında")
        except Exception as exc:
            self.error_occurred.emit(f"TX başlatılamadı: {exc}")
            self._safe_shutdown()
            return

        try:
            while self._running and not self._emergency:
                # Emniyet: yalnız switch TX_ACTIVE iken yaz (yanlış yola yayın yok).
                if self._switch.state != SwitchState.TX_ACTIVE:
                    self.error_occurred.emit("Switch TX değil; yayın durduruldu.")
                    break
                try:
                    self._sdr.write_samples(self._next_block())
                except Exception as exc:
                    self.error_occurred.emit(f"TX yazma hatası: {exc}")
                    break
        finally:
            self._safe_shutdown()

    def _safe_shutdown(self) -> None:
        """TX akışını kapat ve (acil değilse) güvenli RX yoluna dön."""
        try:
            self._sdr.stop_tx()
        except Exception as exc:
            self.error_occurred.emit(f"TX durdurma hatası: {exc}")
        # Acil durumda switch zaten emergency_disable ile güvenli yola alındı;
        # normal durdurmada düzgün TX→RX dizisini uygula.
        if not self._emergency:
            try:
                if self._switch.state == SwitchState.TX_ACTIVE:
                    self._switch.release_tx()
            except Exception as exc:
                self.error_occurred.emit(f"Switch RX'e dönüş hatası: {exc}")
        self.tx_state_changed.emit(False)
        self.status_changed.emit("TX durduruldu")
