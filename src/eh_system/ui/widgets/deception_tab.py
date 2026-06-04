"""Aldatma (deception) sekmesi: analog telsiz NBFM aldatma yayını.

Hedef frekans, ses kaynağı (test tonu / WAV dosyası), CTCSS tonu, frekans
sapması seçilir; ses zinciri (bandpass → pre-emphasis → CTCSS → NBFM) ile
modüle edilip yayınlanır.

Emniyet (CLAUDE.md):
- TX güç yazılımsal limitle sınırlı (limit aşılamaz).
- Yayın başlatmadan önce operatör onayı + yasal uyarı.
- Acil Durdur her zaman erişilebilir.

Bu widget SDR/worker'a dokunmaz; isteği Qt signal ile MainWindow'a iletir.
NBFM sesi dar bant olduğundan ayrı bir ses-baseband örnekleme hızında üretilir.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from scipy.signal import resample

from ...dsp.modulator import CTCSS_TONES, DEFAULT_DEVIATION_HZ, process_voice

# Ses-baseband örnekleme hızı (NBFM dar bant; TX bunu kullanır).
VOICE_SR = 48_000.0
# Test tonu süresi (s) ve frekansı (Hz).
_TEST_TONE_SEC = 2.0
_TEST_TONE_HZ = 1000.0
_SRC_TEST = "Test tonu (1 kHz)"
_SRC_WAV = "WAV dosyası"
_CTCSS_NONE = "Yok"

_LEGAL_WARNING = (
    "YASAL UYARI — TELSİZ ALDATMA\n\n"
    "RF yayını yalnızca yetkili test ortamında (Faraday kafesi veya yasal "
    "izinli düşük güç) yapılmalıdır. İzinsiz yayın yasa dışıdır.\n\n"
    "Aldatma yayınını başlatmayı onaylıyor musunuz?"
)


class DeceptionTab(QWidget):
    """Analog telsiz NBFM aldatma kontrol sekmesi."""

    # (tx_freq_hz, waveform_iq, power_db, sample_rate) — yayını başlat.
    tx_start_requested = Signal(float, object, float, float)
    tx_stop_requested = Signal()
    emergency_requested = Signal()

    def __init__(self, power_limit_db: float, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._power_limit_db = float(power_limit_db)
        self._wav_path: str | None = None
        self._tx_on = False
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Hedef f (MHz):"))
        self._freq_spin = QDoubleSpinBox()
        self._freq_spin.setRange(1.0, 6000.0)
        self._freq_spin.setDecimals(3)
        self._freq_spin.setValue(446.0)
        row1.addWidget(self._freq_spin)

        row1.addWidget(QLabel("Ses kaynağı:"))
        self._src_combo = QComboBox()
        self._src_combo.addItems([_SRC_TEST, _SRC_WAV])
        self._src_combo.currentTextChanged.connect(self._on_src_changed)
        row1.addWidget(self._src_combo)
        self._wav_btn = QPushButton("WAV Seç")
        self._wav_btn.clicked.connect(self._on_pick_wav)
        self._wav_btn.setEnabled(False)
        row1.addWidget(self._wav_btn)
        row1.addStretch(1)
        root.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("CTCSS:"))
        self._ctcss_combo = QComboBox()
        self._ctcss_combo.addItem(_CTCSS_NONE)
        for tone in CTCSS_TONES:
            self._ctcss_combo.addItem(f"{tone:.1f} Hz", tone)
        row2.addWidget(self._ctcss_combo)

        row2.addWidget(QLabel("Sapma (Hz):"))
        self._dev_spin = QDoubleSpinBox()
        self._dev_spin.setRange(500.0, 75_000.0)
        self._dev_spin.setValue(DEFAULT_DEVIATION_HZ)
        row2.addWidget(self._dev_spin)

        row2.addWidget(QLabel("Güç (dB):"))
        self._power_spin = QDoubleSpinBox()
        self._power_spin.setRange(-60.0, self._power_limit_db)  # limit aşılamaz
        self._power_spin.setValue(min(-10.0, self._power_limit_db))
        row2.addWidget(self._power_spin)
        row2.addWidget(QLabel(f"(limit {self._power_limit_db:.0f} dB)"))
        row2.addStretch(1)
        root.addLayout(row2)

        row3 = QHBoxLayout()
        self._start_btn = QPushButton("Yayın Başlat")
        self._start_btn.clicked.connect(self._on_start)
        row3.addWidget(self._start_btn)
        self._stop_btn = QPushButton("Yayın Durdur")
        self._stop_btn.clicked.connect(self._on_stop)
        self._stop_btn.setEnabled(False)
        row3.addWidget(self._stop_btn)
        self._emergency_btn = QPushButton("ACİL DURDUR")
        self._emergency_btn.setStyleSheet(
            "background-color: #b00000; color: white; font-weight: bold;"
        )
        self._emergency_btn.clicked.connect(self._on_emergency)
        row3.addWidget(self._emergency_btn)
        row3.addStretch(1)
        root.addLayout(row3)

        self._status_label = QLabel("Yayın kapalı.")
        root.addWidget(self._status_label)
        root.addStretch(1)

    def _on_src_changed(self, text: str) -> None:
        self._wav_btn.setEnabled(text == _SRC_WAV)

    def _on_pick_wav(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "WAV seç", "", "WAV (*.wav)")
        if path:
            self._wav_path = path
            self._status_label.setText(f"WAV: {path}")

    def _load_audio(self) -> np.ndarray:
        """Seçili kaynaktan ses örneklerini VOICE_SR'de (mono float) döndür."""
        if self._src_combo.currentText() == _SRC_WAV:
            if not self._wav_path:
                raise ValueError("WAV dosyası seçilmedi.")
            from scipy.io import wavfile

            sr, data = wavfile.read(self._wav_path)
            audio = data.astype(np.float64)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)  # mono'ya indir
            # Tam ölçeğe normalize (int PCM ise).
            if np.issubdtype(data.dtype, np.integer):
                audio /= np.iinfo(data.dtype).max
            if sr != VOICE_SR:
                n_new = int(round(audio.shape[0] * VOICE_SR / sr))
                audio = resample(audio, n_new)
            return audio
        # Test tonu.
        t = np.arange(int(_TEST_TONE_SEC * VOICE_SR)) / VOICE_SR
        return np.sin(2.0 * np.pi * _TEST_TONE_HZ * t)

    def _ctcss_freq(self) -> float | None:
        data = self._ctcss_combo.currentData()
        return float(data) if data is not None else None

    def _on_start(self) -> None:
        try:
            audio = self._load_audio()
        except Exception as exc:
            QMessageBox.warning(self, "Ses hatası", str(exc))
            return
        resp = QMessageBox.warning(
            self,
            "Yayın Onayı",
            _LEGAL_WARNING,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if resp != QMessageBox.StandardButton.Yes:
            return
        waveform = process_voice(
            audio, VOICE_SR, ctcss_freq_hz=self._ctcss_freq(),
            deviation=self._dev_spin.value(),
        )
        self._set_tx_on(True)
        self.tx_start_requested.emit(
            self._freq_spin.value() * 1e6, waveform, self._power_spin.value(), VOICE_SR
        )

    def _on_stop(self) -> None:
        self._set_tx_on(False)
        self.tx_stop_requested.emit()

    def _on_emergency(self) -> None:
        self._set_tx_on(False)
        self._status_label.setText("ACİL DURDUR — yayın kesildi.")
        self.emergency_requested.emit()

    def _set_tx_on(self, on: bool) -> None:
        self._tx_on = on
        self._start_btn.setEnabled(not on)
        self._stop_btn.setEnabled(on)
        self._status_label.setText("YAYINDA" if on else "Yayın kapalı.")

    def notify_tx_stopped(self) -> None:
        """MainWindow TX'in durduğunu bildirince UI'yı senkronla."""
        if self._tx_on:
            self._set_tx_on(False)
