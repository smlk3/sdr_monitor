"""Karıştırma (jamming) sekmesi: hedef seçimi, tür/BW/güç, eş zamanlı limit.

Emniyet (CLAUDE.md):
- TX güç slider'ı yazılımsal üst limitle sınırlıdır (limit aşılamaz).
- Master TX ON, operatör onayı + yasal uyarı dialogu olmadan yayını başlatmaz.
- Eş zamanlı hedef sınırı: en çok 5 DAR veya 2 GENİŞ hedef.
- Acil Durdur düğmesi her zaman erişilebilir.

Bu widget yalnız UI ve dalga formu kompozisyonu yapar; SDR/worker'a dokunmaz.
Yayın isteğini Qt signal ile MainWindow'a iletir (orada RF switch sıralaması +
TX worker yönetilir).
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...dsp.jammer import barrage_jam, multi_tone_jam, sweep_jam

# Karıştırma türleri.
JAM_BARRAGE = "Baraj (AWGN)"
JAM_MULTITONE = "Çoklu-ton"
JAM_SWEEP = "Sweep (chirp)"
_JAM_TYPES = (JAM_BARRAGE, JAM_MULTITONE, JAM_SWEEP)

# Eş zamanlı hedef sınırları ve dar/geniş eşiği.
_NARROW_BW_THRESHOLD_HZ = 200.0e3  # ≤ bu → "dar"
_MAX_NARROW = 5
_MAX_WIDE = 2
# Kompoze edilen dalga formu uzunluğu (ring buffer'da döngüye girer).
_WAVEFORM_LEN = 1 << 16
# Çoklu-ton: hedef bandını dolduran ton sayısı.
_MULTITONE_COUNT = 5

# Yasal uyarı metni (master TX onay dialogunda gösterilir).
_LEGAL_WARNING = (
    "YASAL UYARI — ELEKTRONİK TAARRUZ\n\n"
    "RF yayını yalnızca yetkili test ortamında (Faraday kafesi veya yasal "
    "izinli düşük güç) yapılmalıdır. İzinsiz karıştırma yasa dışıdır ve "
    "yaşamsal haberleşmeyi tehlikeye atabilir.\n\n"
    "TX gücü yazılımsal üst limitle sınırlıdır. Yayını başlatmak için onaylıyor "
    "musunuz?"
)


class JammingTab(QWidget):
    """Karıştırma kontrol sekmesi."""

    # (tx_center_freq_hz, waveform_iq, power_db, sample_rate) — yayını başlat.
    tx_start_requested = Signal(float, object, float, float)
    # Yayını durdur.
    tx_stop_requested = Signal()
    # Acil durdur.
    emergency_requested = Signal()

    def __init__(
        self,
        sample_rate: float,
        power_limit_db: float,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._sample_rate = float(sample_rate)
        self._power_limit_db = float(power_limit_db)
        # Hedef listesi: her biri (freq_hz, bw_hz, tür).
        self._targets: list[tuple[float, float, str]] = []
        self._tx_on = False
        self._build_ui()

    # --- UI ---

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        # Hedef ekleme çubuğu.
        entry = QHBoxLayout()
        entry.addWidget(QLabel("Hedef f (MHz):"))
        self._freq_spin = QDoubleSpinBox()
        self._freq_spin.setRange(1.0, 6000.0)
        self._freq_spin.setDecimals(3)
        self._freq_spin.setValue(433.0)
        entry.addWidget(self._freq_spin)

        entry.addWidget(QLabel("BW (kHz):"))
        self._bw_spin = QDoubleSpinBox()
        self._bw_spin.setRange(1.0, 5000.0)
        self._bw_spin.setValue(100.0)
        entry.addWidget(self._bw_spin)

        entry.addWidget(QLabel("Tür:"))
        self._type_combo = QComboBox()
        for t in _JAM_TYPES:
            self._type_combo.addItem(t)
        entry.addWidget(self._type_combo)

        self._add_btn = QPushButton("Hedef Ekle")
        self._add_btn.clicked.connect(self._on_add_target)
        entry.addWidget(self._add_btn)
        self._del_btn = QPushButton("Seçiliyi Sil")
        self._del_btn.clicked.connect(self._on_del_target)
        entry.addWidget(self._del_btn)
        entry.addStretch(1)
        root.addLayout(entry)

        # Hedef tablosu.
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(
            ["Frekans (MHz)", "BW (kHz)", "Tür", "Sınıf"]
        )
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        root.addWidget(self._table)

        self._limit_label = QLabel()
        root.addWidget(self._limit_label)

        # TX kontrol çubuğu.
        txbar = QHBoxLayout()
        txbar.addWidget(QLabel("TX merkez f (MHz):"))
        self._tx_freq_spin = QDoubleSpinBox()
        self._tx_freq_spin.setRange(1.0, 6000.0)
        self._tx_freq_spin.setDecimals(3)
        self._tx_freq_spin.setValue(433.0)
        txbar.addWidget(self._tx_freq_spin)

        txbar.addWidget(QLabel("Güç (dB):"))
        self._power_spin = QDoubleSpinBox()
        # Slider/spin yazılımsal limiti AŞAMAZ.
        self._power_spin.setRange(-60.0, self._power_limit_db)
        self._power_spin.setValue(min(-10.0, self._power_limit_db))
        txbar.addWidget(self._power_spin)
        txbar.addWidget(QLabel(f"(limit {self._power_limit_db:.0f} dB)"))

        txbar.addStretch(1)
        self._master_btn = QPushButton("Master TX: KAPALI")
        self._master_btn.setCheckable(True)
        self._master_btn.clicked.connect(self._on_master_toggled)
        txbar.addWidget(self._master_btn)

        self._emergency_btn = QPushButton("ACİL DURDUR")
        self._emergency_btn.setStyleSheet(
            "background-color: #b00000; color: white; font-weight: bold;"
        )
        self._emergency_btn.clicked.connect(self._on_emergency)
        txbar.addWidget(self._emergency_btn)
        root.addLayout(txbar)

        self._status_label = QLabel("TX kapalı.")
        root.addWidget(self._status_label)

        self._refresh_limit_label()

    # --- Hedef yönetimi + eş zamanlı limit ---

    def _counts(self) -> tuple[int, int]:
        """(dar, geniş) hedef sayılarını döndür."""
        narrow = sum(1 for _f, bw, _t in self._targets if bw <= _NARROW_BW_THRESHOLD_HZ)
        return narrow, len(self._targets) - narrow

    def _refresh_limit_label(self) -> None:
        narrow, wide = self._counts()
        self._limit_label.setText(
            f"Eş zamanlı hedef — Dar: {narrow}/{_MAX_NARROW}   "
            f"Geniş: {wide}/{_MAX_WIDE}   (dar ≤ {_NARROW_BW_THRESHOLD_HZ / 1e3:.0f} kHz)"
        )

    def _on_add_target(self) -> None:
        """Hedef ekle; eş zamanlı sınırı aşacaksa reddet (emniyet)."""
        freq = self._freq_spin.value() * 1e6
        bw = self._bw_spin.value() * 1e3
        jam_type = self._type_combo.currentText()
        is_narrow = bw <= _NARROW_BW_THRESHOLD_HZ
        narrow, wide = self._counts()
        if is_narrow and narrow >= _MAX_NARROW:
            QMessageBox.warning(self, "Sınır", f"En çok {_MAX_NARROW} dar hedef.")
            return
        if not is_narrow and wide >= _MAX_WIDE:
            QMessageBox.warning(self, "Sınır", f"En çok {_MAX_WIDE} geniş hedef.")
            return
        self._targets.append((freq, bw, jam_type))
        self._refresh_table()

    def _on_del_target(self) -> None:
        row = self._table.currentRow()
        if 0 <= row < len(self._targets):
            self._targets.pop(row)
            self._refresh_table()

    def _refresh_table(self) -> None:
        self._table.setRowCount(len(self._targets))
        for row, (freq, bw, jam_type) in enumerate(self._targets):
            klass = "dar" if bw <= _NARROW_BW_THRESHOLD_HZ else "geniş"
            cells = (f"{freq / 1e6:.3f}", f"{bw / 1e3:.1f}", jam_type, klass)
            for col, text in enumerate(cells):
                self._table.setItem(row, col, QTableWidgetItem(text))
        self._refresh_limit_label()

    # --- Dalga formu kompozisyonu ---

    def _compose_waveform(self, tx_center: float) -> np.ndarray:
        """Tüm hedeflerin karıştırma dalga formunu baseband'de topla."""
        n = _WAVEFORM_LEN
        t = np.arange(n, dtype=np.float64) / self._sample_rate
        total = np.zeros(n, dtype=np.complex128)
        for freq, bw, jam_type in self._targets:
            offset = freq - tx_center  # baseband ofset
            if jam_type == JAM_BARRAGE:
                base = barrage_jam(n, self._sample_rate, max(bw, 1.0), amplitude=1.0)
                comp = base * np.exp(2j * np.pi * offset * t)
            elif jam_type == JAM_MULTITONE:
                tones = offset + np.linspace(-bw / 2, bw / 2, _MULTITONE_COUNT)
                comp = multi_tone_jam(n, self._sample_rate, tones, amplitude=1.0)
            else:  # JAM_SWEEP
                comp = sweep_jam(
                    n, self._sample_rate, offset - bw / 2, offset + bw / 2,
                    sweep_time_s=n / self._sample_rate, amplitude=1.0,
                )
            total += comp
        peak = float(np.max(np.abs(total))) if total.size else 0.0
        if peak > 1e-12:
            total /= peak  # [0,1]'e normalize; gerçek güç tx_worker'da ölçeklenir
        return total.astype(np.complex64)

    # --- TX kontrol ---

    def _on_master_toggled(self, checked: bool) -> None:
        if checked:
            if not self._targets:
                QMessageBox.information(self, "Hedef yok", "Önce hedef ekleyin.")
                self._master_btn.setChecked(False)
                return
            # Operatör onayı + yasal uyarı (emniyet).
            resp = QMessageBox.warning(
                self,
                "TX Onayı",
                _LEGAL_WARNING,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if resp != QMessageBox.StandardButton.Yes:
                self._master_btn.setChecked(False)
                return
            tx_center = self._tx_freq_spin.value() * 1e6
            waveform = self._compose_waveform(tx_center)
            self._set_tx_on(True)
            self.tx_start_requested.emit(
                tx_center, waveform, self._power_spin.value(), self._sample_rate
            )
        else:
            self._set_tx_on(False)
            self.tx_stop_requested.emit()

    def _set_tx_on(self, on: bool) -> None:
        self._tx_on = on
        self._master_btn.setChecked(on)
        self._master_btn.setText(f"Master TX: {'AÇIK' if on else 'KAPALI'}")
        self._master_btn.setStyleSheet(
            "background-color: #208020; color: white;" if on else ""
        )
        self._status_label.setText("TX YAYINDA" if on else "TX kapalı.")

    def _on_emergency(self) -> None:
        """Acil durdur: TX'i kes ve UI'yı kapalı duruma al."""
        self._set_tx_on(False)
        self._status_label.setText("ACİL DURDUR — TX kesildi.")
        self.emergency_requested.emit()

    def notify_tx_stopped(self) -> None:
        """MainWindow TX'in durduğunu bildirince UI'yı senkronla."""
        if self._tx_on:
            self._set_tx_on(False)
