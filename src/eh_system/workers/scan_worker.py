"""DF tarama worker'ı: motorlu stop-and-measure tarama + faz interferometri.

CLAUDE.md kuralları:
- Ağır iş (SDR I/O, FFT, DOA) GUI thread'inde ASLA çalışmaz; burada QThread.
- Worker -> UI yalnızca Qt signal/slot; numpy array'ler kopyalanarak geçer.
- DF taraması STOP-AND-MEASURE: dön → dur → bekle (settle) → ölç. Hareket
  halinde ölçüm yapılmaz.
- Faz kalibrasyonu LO retune sonrası geçersizdir (calibration_required); tarama
  öncesi "Kalibre" ile (referans modunda) ölçülen kayma/faz uygulanır.

İki çalışma kipi (``mode``):
- ``"calibrate"``: referans (broadside) kaynağa karşı kanal kayması + sabit
  faz farkını ölçer; ``calibration_done`` ile döndürür.
- ``"scan"``: her açıda durup ölçer; açı-güç haritası + hedef yön (bearing)
  üretir. Her açıda kalibre edilmiş kanallardan tek-kaynak DoA hesaplanır.

ÖNEMLİ: Faz interferometri TEK KAYNAK içindir (bkz. dsp.doa). Açı-güç
haritasının tepe noktası hedef yönü verir; DoA o yöndeki ince kestirimdir.
"""

from __future__ import annotations

import time

import numpy as np
from PySide6.QtCore import QThread, Signal

from ..dsp.calibration import apply_calibration, phase_calibration
from ..dsp.doa import doa_phase_interferometry
from ..dsp.spectrum import compute_psd
from ..hal.motor import MotorBase
from ..hal.sdr_base import SDRBase

# Varsayılan tarama parametreleri.
_DEFAULT_START_DEG = 0.0
_DEFAULT_STOP_DEG = 180.0
_DEFAULT_STEP_DEG = 10.0
# Stop-and-measure bekleme süresi (s): dön → dur → 50 ms bekle → ölç.
_DEFAULT_SETTLE_S = 0.05
# Kalibrasyonun yapıldığı referans (broadside) açısı.
_CAL_REFERENCE_DEG = 0.0


class ScanWorker(QThread):
    """Motorlu DF taraması ve kanal kalibrasyonu yapan worker thread'i."""

    # Her açıdaki ölçüm: (motor_açısı_derece, güç_dB, doa_derece).
    measurement_ready = Signal(float, float, float)
    # Tarama bitti: açı-güç haritası list[tuple[float, float]] (kopya).
    scan_complete = Signal(object)
    # Hedef yön: (bearing_derece, tepe_güç_dB, tepe_yöndeki_doa_derece).
    # DoA, en güçlü (en güvenilir SNR) açıdaki faz interferometri kestirimidir.
    bearing_ready = Signal(float, float, float)
    # Kalibrasyon tamam: (örnek_kayması, faz_farkı_rad).
    calibration_done = Signal(int, float)
    # Durum metni / hata.
    status_changed = Signal(str)
    error_occurred = Signal(str)

    def __init__(
        self,
        sdr: SDRBase,
        motor: MotorBase,
        baseline_m: float,
        fft_size: int = 1024,
        read_size: int = 4096,
        window: str = "hann",
        num_train: int = 16,
        num_guard: int = 4,
        pfa: float = 1e-4,
        mode: str = "scan",
        start_deg: float = _DEFAULT_START_DEG,
        stop_deg: float = _DEFAULT_STOP_DEG,
        step_deg: float = _DEFAULT_STEP_DEG,
        settle_s: float = _DEFAULT_SETTLE_S,
        cal_shift: int = 0,
        cal_phase: float = 0.0,
        parent: object | None = None,
    ) -> None:
        super().__init__(parent)
        self._sdr = sdr
        self._motor = motor
        self._baseline_m = float(baseline_m)
        self._fft_size = int(fft_size)
        self._read_size = max(int(read_size), self._fft_size)
        self._window = window
        self._num_train = int(num_train)
        self._num_guard = int(num_guard)
        self._pfa = float(pfa)
        self._mode = str(mode)
        self._start_deg = float(start_deg)
        self._stop_deg = float(stop_deg)
        self._step_deg = float(step_deg)
        self._settle_s = float(settle_s)
        self._cal_shift = int(cal_shift)
        self._cal_phase = float(cal_phase)
        self._running = False

    def stop(self) -> None:
        """Taramayı temiz biçimde durdur ve thread'in bitmesini bekle."""
        self._running = False
        self.wait(5000)

    # --- Ana giriş ---

    def run(self) -> None:
        """Thread giriş noktası: aç, kipe göre kalibre et veya tara, kapat."""
        self._running = True
        opened_sdr = False
        opened_motor = False
        try:
            self._sdr.open()
            opened_sdr = True
            self._motor.open()
            opened_motor = True
        except Exception as exc:
            self.error_occurred.emit(f"DF cihaz açılamadı: {exc}")
            self.status_changed.emit("Bağlantı yok")
            self._safe_close(opened_sdr, opened_motor)
            return

        try:
            if self._mode == "calibrate":
                self._run_calibration()
            else:
                self._run_scan()
        except NotImplementedError as exc:
            self.error_occurred.emit(f"Çift kanal desteklenmiyor: {exc}")
        except Exception as exc:  # çökme yok
            self.error_occurred.emit(f"DF hatası: {exc}")
        finally:
            self._safe_close(opened_sdr, opened_motor)

    def _safe_close(self, sdr_open: bool, motor_open: bool) -> None:
        """SDR ve motoru güvenle kapat (hata yutulur)."""
        if sdr_open:
            try:
                self._sdr.close()
            except Exception as exc:
                self.error_occurred.emit(f"SDR kapatma hatası: {exc}")
        if motor_open:
            try:
                self._motor.close()
            except Exception as exc:
                self.error_occurred.emit(f"Motor kapatma hatası: {exc}")

    # --- Kalibrasyon kipi ---

    def _run_calibration(self) -> None:
        """Referans (broadside) kaynağa karşı kanal kayması + fazını ölç."""
        self.status_changed.emit("Kalibrasyon: referansa yöneliniyor...")
        try:
            self._motor.move_absolute(_CAL_REFERENCE_DEG)
        except Exception as exc:
            self.error_occurred.emit(f"Motor hatası: {exc}")
        # Kalibrasyon tam sinyalle yapılır (yönlü kazanç devre dışı).
        set_point = getattr(self._sdr, "set_pointing_angle", None)
        if callable(set_point):
            set_point(None)
        time.sleep(self._settle_s)

        # MockSDR referans modu: yalnız donanımsal uyumsuzluk üretilir.
        set_ref = getattr(self._sdr, "set_calibration_reference", None)
        if callable(set_ref):
            set_ref(True)
        try:
            rx0, rx1 = self._sdr.read_samples_dual(self._read_size)
            shift, phase = phase_calibration(rx0, rx1)
            self._cal_shift, self._cal_phase = shift, phase
            self.calibration_done.emit(int(shift), float(phase))
            self.status_changed.emit(
                f"Kalibrasyon tamam: kayma={shift} örnek, faz={phase:.3f} rad"
            )
        finally:
            if callable(set_ref):
                set_ref(False)

    # --- Tarama kipi ---

    def _angles(self) -> np.ndarray:
        """Tarama açıları dizisi (start..stop, step adımlı, stop dahil)."""
        if self._step_deg <= 0.0:
            raise ValueError("step_deg pozitif olmalı.")
        n = int(np.floor((self._stop_deg - self._start_deg) / self._step_deg)) + 1
        return self._start_deg + self._step_deg * np.arange(max(n, 1))

    def _run_scan(self) -> None:
        """Stop-and-measure tarama: her açıda dur, bekle, ölç, DoA hesapla."""
        freq = self._sdr.frequency
        angle_power: list[tuple[float, float]] = []
        best_angle, best_power, best_doa = 0.0, -np.inf, 0.0

        for angle in self._angles():
            if not self._running:
                break
            # Dön → dur → bekle (settle) → ölç.
            try:
                self._motor.move_absolute(float(angle))
            except Exception as exc:
                self.error_occurred.emit(f"Motor hatası: {exc}")
            # MockSDR: anten yönelimini bildir (yönlü kazanç → açı-güç lobu).
            set_point = getattr(self._sdr, "set_pointing_angle", None)
            if callable(set_point):
                set_point(float(angle))
            time.sleep(self._settle_s)

            try:
                rx0, rx1 = self._sdr.read_samples_dual(self._read_size)
            except Exception as exc:
                self.error_occurred.emit(f"Okuma hatası: {exc}")
                continue

            try:
                _freqs, psd_db = compute_psd(
                    rx0,
                    self._fft_size,
                    sample_rate=self._sdr.sample_rate,
                    center_freq=freq,
                    window=self._window,
                )
                power_db = float(np.max(psd_db))
                rx1_corr = apply_calibration(rx1, self._cal_shift, self._cal_phase)
                doa_deg = doa_phase_interferometry(
                    rx0, rx1_corr, self._baseline_m, freq
                )
            except Exception as exc:
                self.error_occurred.emit(f"DOA/PSD hatası: {exc}")
                continue

            self.measurement_ready.emit(float(angle), power_db, float(doa_deg))
            angle_power.append((float(angle), power_db))
            if power_db > best_power:
                best_power = power_db
                best_angle = float(angle)
                best_doa = float(doa_deg)

        # Yönlü kazancı sıfırla (canlı spektrum yön bağımsız kalsın).
        set_point = getattr(self._sdr, "set_pointing_angle", None)
        if callable(set_point):
            set_point(None)

        if angle_power:
            self.bearing_ready.emit(best_angle, best_power, best_doa)
        self.scan_complete.emit(list(angle_power))
        self.status_changed.emit("Tarama tamam")
