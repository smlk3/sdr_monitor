"""Ana pencere: gerçek zamanlı spektrum + kayan waterfall + kontroller.

CLAUDE.md kuralları:
- UI, DSP'yi doğrudan çağırmaz; tüm ağır iş SDRWorker (QThread) üzerindedir.
- Worker -> UI yalnızca signal/slot. UI -> Worker thread-safe request_* metotları.
"""

from __future__ import annotations

import time

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..hal.sdr_base import SDRBase
from ..workers.sdr_worker import SDRWorker

# Seçilebilir örnekleme hızları (Hz). Sürdürülebilir IBW ~30 MHz (CLAUDE.md).
_SAMPLE_RATES = (2.0e6, 5.0e6, 10.0e6, 20.0e6, 30.0e6)
# Seçilebilir FFT boyutları.
_FFT_SIZES = (512, 1024, 2048, 4096)
# Kazanç slider aralığı (dB).
_GAIN_MIN, _GAIN_MAX = 0, 70
# Waterfall renk ölçeği için varsayılan dB aralığı.
_WF_DB_MIN, _WF_DB_MAX = -120.0, -10.0
# CFAR varsayılan UI değerleri ve sınırları.
_CFAR_NUM_TRAIN_DEF, _CFAR_NUM_TRAIN_MIN, _CFAR_NUM_TRAIN_MAX = 16, 1, 256
_CFAR_NUM_GUARD_DEF, _CFAR_NUM_GUARD_MIN, _CFAR_NUM_GUARD_MAX = 4, 0, 64
# Pfa = 10^(-exp); slider/spin tam sayı üs ile çalışır.
_CFAR_PFA_EXP_DEF, _CFAR_PFA_EXP_MIN, _CFAR_PFA_EXP_MAX = 4, 1, 9
# Sinyal tablosu sütun başlıkları.
_TABLE_HEADERS = (
    "ID", "Merkez (MHz)", "BW-10dB (kHz)", "BW-20dB (kHz)",
    "Güç (dB)", "Tip", "İlk (s)", "Son (s)",
)
# Waterfall'da tespit bölgelerini işaretleyen LinearRegionItem havuzu boyutu.
_MAX_REGIONS = 32


class MainWindow(QMainWindow):
    """Spektrum analizör ana penceresi."""

    def __init__(
        self,
        sdr: SDRBase,
        fft_size: int = 1024,
        read_size: int = 4096,
        sample_rate: float = 10.0e6,
        frequency: float = 433.0e6,
        gain_db: float = 30.0,
        waterfall_rows: int = 200,
        window: str = "hann",
        num_train: int = _CFAR_NUM_TRAIN_DEF,
        num_guard: int = _CFAR_NUM_GUARD_DEF,
        pfa_exp: int = _CFAR_PFA_EXP_DEF,
    ) -> None:
        super().__init__()
        self.setWindowTitle("EH Sistemi — Spektrum Analizör")
        self._sdr = sdr
        self._fft_size = fft_size
        self._waterfall_rows = waterfall_rows
        self._wf_buffer: np.ndarray | None = None
        self._wf_freqs: np.ndarray | None = None
        # En son alınan frekans ekseni (CFAR eşik eğrisini hizalamak için).
        self._last_freqs: np.ndarray | None = None

        # --- Worker ---
        self._worker = SDRWorker(
            sdr,
            fft_size=fft_size,
            read_size=read_size,
            window=window,
            num_train=num_train,
            num_guard=num_guard,
            pfa=10.0 ** (-pfa_exp),
        )
        self._worker.spectrum_ready.connect(self._on_spectrum)
        self._worker.cfar_ready.connect(self._on_cfar)
        self._worker.signals_ready.connect(self._on_signals)
        self._worker.fps_updated.connect(self._on_fps)
        self._worker.overflow_updated.connect(self._on_overflow)
        self._worker.status_changed.connect(self._on_status)
        self._worker.error_occurred.connect(self._on_error)

        self._build_ui(sample_rate, frequency, gain_db, num_train, num_guard, pfa_exp)
        self._init_waterfall(fft_size)

    # --- UI kurulumu ---

    def _build_ui(
        self,
        sample_rate: float,
        frequency: float,
        gain_db: float,
        num_train: int,
        num_guard: int,
        pfa_exp: int,
    ) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # Kontrol çubuğu
        controls = QHBoxLayout()

        controls.addWidget(QLabel("Frekans (MHz):"))
        self._freq_spin = QDoubleSpinBox()
        self._freq_spin.setRange(1.0, 6000.0)
        self._freq_spin.setDecimals(3)
        self._freq_spin.setValue(frequency / 1e6)
        self._freq_spin.setSingleStep(0.5)
        self._freq_spin.editingFinished.connect(self._on_freq_changed)
        controls.addWidget(self._freq_spin)

        controls.addWidget(QLabel("SR:"))
        self._sr_combo = QComboBox()
        for r in _SAMPLE_RATES:
            self._sr_combo.addItem(f"{r / 1e6:g} MHz", r)
        self._sr_combo.setCurrentIndex(_SAMPLE_RATES.index(sample_rate)
                                       if sample_rate in _SAMPLE_RATES else 2)
        self._sr_combo.currentIndexChanged.connect(self._on_sr_changed)
        controls.addWidget(self._sr_combo)

        controls.addWidget(QLabel("FFT:"))
        self._fft_combo = QComboBox()
        for s in _FFT_SIZES:
            self._fft_combo.addItem(str(s), s)
        self._fft_combo.setCurrentIndex(_FFT_SIZES.index(self._fft_size)
                                        if self._fft_size in _FFT_SIZES else 1)
        self._fft_combo.currentIndexChanged.connect(self._on_fft_changed)
        controls.addWidget(self._fft_combo)

        controls.addWidget(QLabel("Kazanç:"))
        self._gain_slider = QSlider(Qt.Orientation.Horizontal)
        self._gain_slider.setRange(_GAIN_MIN, _GAIN_MAX)
        self._gain_slider.setValue(int(gain_db))
        self._gain_slider.setFixedWidth(140)
        self._gain_slider.valueChanged.connect(self._on_gain_changed)
        controls.addWidget(self._gain_slider)
        self._gain_label = QLabel(f"{int(gain_db)} dB")
        controls.addWidget(self._gain_label)

        controls.addStretch(1)

        self._start_btn = QPushButton("Başlat")
        self._start_btn.clicked.connect(self.start)
        controls.addWidget(self._start_btn)
        self._stop_btn = QPushButton("Durdur")
        self._stop_btn.clicked.connect(self.stop)
        self._stop_btn.setEnabled(False)
        controls.addWidget(self._stop_btn)

        root.addLayout(controls)

        # CFAR kontrol çubuğu (ikinci satır)
        cfar_bar = QHBoxLayout()
        cfar_bar.addWidget(QLabel("CFAR — Eğitim:"))
        self._train_spin = QSpinBox()
        self._train_spin.setRange(_CFAR_NUM_TRAIN_MIN, _CFAR_NUM_TRAIN_MAX)
        self._train_spin.setValue(num_train)
        self._train_spin.valueChanged.connect(self._on_num_train_changed)
        cfar_bar.addWidget(self._train_spin)

        cfar_bar.addWidget(QLabel("Koruma:"))
        self._guard_spin = QSpinBox()
        self._guard_spin.setRange(_CFAR_NUM_GUARD_MIN, _CFAR_NUM_GUARD_MAX)
        self._guard_spin.setValue(num_guard)
        self._guard_spin.valueChanged.connect(self._on_num_guard_changed)
        cfar_bar.addWidget(self._guard_spin)

        cfar_bar.addWidget(QLabel("Pfa = 1e-"))
        self._pfa_spin = QSpinBox()
        self._pfa_spin.setRange(_CFAR_PFA_EXP_MIN, _CFAR_PFA_EXP_MAX)
        self._pfa_spin.setValue(pfa_exp)
        self._pfa_spin.valueChanged.connect(self._on_pfa_changed)
        cfar_bar.addWidget(self._pfa_spin)

        cfar_bar.addStretch(1)
        self._det_count_label = QLabel("Tespit: 0")
        cfar_bar.addWidget(self._det_count_label)
        root.addLayout(cfar_bar)

        # Spektrum grafiği
        self._spectrum_plot = pg.PlotWidget()
        self._spectrum_plot.setLabel("bottom", "Frekans", units="Hz")
        self._spectrum_plot.setLabel("left", "Güç", units="dB")
        self._spectrum_plot.showGrid(x=True, y=True, alpha=0.3)
        self._spectrum_plot.setYRange(_WF_DB_MIN, _WF_DB_MAX)
        self._spectrum_curve = self._spectrum_plot.plot(pen=pg.mkPen("y", width=1))
        # CFAR eşik eğrisi (kırmızı kesik çizgi).
        self._threshold_curve = self._spectrum_plot.plot(
            pen=pg.mkPen("r", width=1, style=Qt.PenStyle.DashLine)
        )
        root.addWidget(self._spectrum_plot, stretch=2)

        # Waterfall
        self._wf_plot = pg.PlotWidget()
        self._wf_plot.setLabel("bottom", "Frekans", units="Hz")
        self._wf_plot.setLabel("left", "Zaman (kare)")
        self._wf_img = pg.ImageItem()
        self._wf_plot.addItem(self._wf_img)
        # Renk haritası (viridis benzeri).
        cmap = pg.colormap.get("viridis")
        self._wf_img.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))
        self._wf_img.setLevels((_WF_DB_MIN, _WF_DB_MAX))

        # Tespit edilen sinyalleri waterfall üzerinde işaretlemek için dikey
        # LinearRegionItem havuzu (her kare yeniden oluşturmak yerine yeniden
        # kullanılır; gereksizler gizlenir).
        self._regions: list[pg.LinearRegionItem] = []
        for _ in range(_MAX_REGIONS):
            region = pg.LinearRegionItem(
                orientation="vertical",
                movable=False,
                brush=pg.mkBrush(255, 0, 0, 40),
                pen=pg.mkPen(255, 80, 80, 160),
            )
            region.setZValue(10)
            region.hide()
            self._wf_plot.addItem(region)
            self._regions.append(region)

        root.addWidget(self._wf_plot, stretch=3)

        # Sinyal listesi tablosu
        self._table = QTableWidget(0, len(_TABLE_HEADERS))
        self._table.setHorizontalHeaderLabels(list(_TABLE_HEADERS))
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        root.addWidget(self._table, stretch=2)

        # Durum çubuğu
        self._status = self.statusBar()
        self._conn_label = QLabel("Bağlantı yok")
        self._fps_label = QLabel("FPS: 0.0")
        self._ovf_label = QLabel("Overflow: 0")
        self._status.addPermanentWidget(self._conn_label)
        self._status.addPermanentWidget(self._fps_label)
        self._status.addPermanentWidget(self._ovf_label)

        self.resize(1000, 720)

    def _init_waterfall(self, fft_size: int) -> None:
        """Dairesel waterfall tamponunu (satır x fft) -120 dB ile başlat."""
        self._fft_size = fft_size
        self._wf_buffer = np.full(
            (self._waterfall_rows, fft_size), _WF_DB_MIN, dtype=np.float32
        )
        self._wf_freqs = None

    # --- Kontrol slot'ları (UI -> Worker, thread-safe) ---

    def _on_freq_changed(self) -> None:
        self._worker.request_frequency(self._freq_spin.value() * 1e6)

    def _on_sr_changed(self) -> None:
        self._worker.request_sample_rate(self._sr_combo.currentData())

    def _on_fft_changed(self) -> None:
        size = int(self._fft_combo.currentData())
        self._init_waterfall(size)
        self._worker.request_fft_size(size)

    def _on_gain_changed(self, value: int) -> None:
        self._gain_label.setText(f"{value} dB")
        self._worker.request_gain(float(value))

    def _on_num_train_changed(self, value: int) -> None:
        self._worker.request_num_train(value)

    def _on_num_guard_changed(self, value: int) -> None:
        self._worker.request_num_guard(value)

    def _on_pfa_changed(self, exp: int) -> None:
        self._worker.request_pfa(10.0 ** (-exp))

    # --- Yaşam döngüsü ---

    def start(self) -> None:
        """Worker thread'ini başlat."""
        if not self._worker.isRunning():
            self._worker.start()
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)

    def stop(self) -> None:
        """Worker thread'ini temiz durdur."""
        self._worker.stop()
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)

    # --- Worker -> UI slot'ları ---

    def _on_spectrum(self, freqs: np.ndarray, psd_db: np.ndarray) -> None:
        """Yeni spektrum karesini çiz ve waterfall'a it."""
        # FFT boyutu değiştiyse tamponu yeniden boyutlandır.
        if self._wf_buffer is None or self._wf_buffer.shape[1] != psd_db.shape[0]:
            self._init_waterfall(psd_db.shape[0])

        self._last_freqs = freqs
        self._spectrum_curve.setData(freqs, psd_db)

        # Dairesel kaydırma: en üstte yeni satır görünür şekilde aşağı kaydır.
        buf = self._wf_buffer
        buf[1:, :] = buf[:-1, :]
        buf[0, :] = psd_db.astype(np.float32)

        # ImageItem: satır=zaman, sütun=frekans. Frekans eksenine oturt.
        f0, f1 = float(freqs[0]), float(freqs[-1])
        self._wf_img.setImage(
            buf.T, autoLevels=False, levels=(_WF_DB_MIN, _WF_DB_MAX)
        )
        # Görüntüyü frekans eksenine ölçekle/konumla.
        self._wf_img.setRect(pg.QtCore.QRectF(f0, 0, f1 - f0, self._waterfall_rows))

    def _on_cfar(self, threshold_db: np.ndarray) -> None:
        """CFAR eşiğini spektrum grafiğine kırmızı kesik çizgi olarak çiz."""
        freqs = self._last_freqs
        if freqs is None or freqs.shape[0] != threshold_db.shape[0]:
            return
        self._threshold_curve.setData(freqs, threshold_db)

    def _on_signals(self, signals: list) -> None:
        """İzlenen sinyalleri tabloya ve waterfall bölgelerine yansıt."""
        self._det_count_label.setText(f"Tespit: {len(signals)}")
        self._update_table(signals)
        self._update_regions(signals)

    def _update_table(self, signals: list) -> None:
        """Sinyal listesini QTableWidget'a yaz (ID'ye göre kararlı sıra)."""
        ordered = sorted(signals, key=lambda s: s.id)
        now = time.time()
        self._table.setRowCount(len(ordered))
        for row, sig in enumerate(ordered):
            p = sig.params
            # "İlk/Son": kaç saniye önce ilk/son görüldü (yaş).
            first_age = max(0.0, now - sig.first_seen)
            last_age = max(0.0, now - sig.last_seen)
            cells = (
                str(sig.id),
                f"{p.center_freq / 1e6:.4f}",
                f"{p.bw_10db / 1e3:.1f}",
                f"{p.bw_20db / 1e3:.1f}",
                f"{p.power_dbm:.1f}",
                p.signal_type,
                f"{first_age:.1f}",
                f"{last_age:.1f}",
            )
            for col, text in enumerate(cells):
                self._table.setItem(row, col, QTableWidgetItem(text))

    def _update_regions(self, signals: list) -> None:
        """Tespit bantlarını waterfall üzerinde LinearRegionItem ile işaretle."""
        for i, region in enumerate(self._regions):
            if i < len(signals):
                p = signals[i].params
                region.setRegion((p.f_lo, p.f_hi))
                region.show()
            else:
                region.hide()

    def _on_fps(self, fps: float) -> None:
        self._fps_label.setText(f"FPS: {fps:.1f}")

    def _on_overflow(self, count: int) -> None:
        self._ovf_label.setText(f"Overflow: {count}")

    def _on_status(self, text: str) -> None:
        self._conn_label.setText(text)

    def _on_error(self, message: str) -> None:
        self._status.showMessage(message, 5000)

    # --- Kapanış ---

    def closeEvent(self, event: object) -> None:  # noqa: N802 (Qt API)
        """Pencere kapanırken worker'ı temiz durdur."""
        self._worker.stop()
        super().closeEvent(event)  # type: ignore[arg-type]
