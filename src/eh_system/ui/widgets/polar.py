"""Kutupsal (polar) açı-güç grafiği ve hedef yön ibresi.

pyqtgraph'ta hazır bir PolarPlotItem yoktur; kutupsal ızgara (eş merkezli
çemberler + açısal ışınlar) elle çizilir ve veriler kartezyene çevrilerek
gösterilir. Açı düzeni PUSULA benzeridir: 0° yukarı (ileri), saat yönünde artar
(x = r·sin θ, y = r·cos θ).

Saf görselleştirme; DSP yapmaz. Güç (dB), ``[vmin, vmax]`` aralığından [0, 1]
yarıçapına ölçeklenir.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg

# Izgara çemberlerinin (normalize) yarıçapları ve açısal ışın adımı (derece).
_GRID_RADII = (0.25, 0.5, 0.75, 1.0)
_SPOKE_STEP_DEG = 30
_LABEL_RADIUS = 1.12


def _polar_to_xy(angle_deg: np.ndarray, radius: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pusula açısı (0°=yukarı, saat yönü) + yarıçapı kartezyene çevir."""
    a = np.radians(angle_deg)
    return radius * np.sin(a), radius * np.cos(a)


class PolarPlot(pg.PlotWidget):
    """Açı-güç kutupsal grafiği + hedef yön ibresi.

    Args:
        vmin: 0 yarıçapına denk gelen güç (dB).
        vmax: 1 yarıçapına denk gelen güç (dB).
    """

    def __init__(
        self,
        vmin: float = -90.0,
        vmax: float = -10.0,
        parent: object | None = None,
    ) -> None:
        super().__init__(parent)
        self._vmin = float(vmin)
        self._vmax = float(vmax)

        self.setAspectLocked(True)
        self.hideAxis("bottom")
        self.hideAxis("left")
        self.setMouseEnabled(x=False, y=False)
        self.setMenuEnabled(False)
        self.setRange(xRange=(-1.25, 1.25), yRange=(-1.25, 1.25))

        self._draw_grid()

        # Veri eğrisi (açı-güç) ve ibre (bearing) — yeniden kullanılır.
        self._curve = self.plot(pen=pg.mkPen("c", width=2), symbol="o", symbolSize=5)
        self._needle = self.plot(pen=pg.mkPen("r", width=3))

    def _draw_grid(self) -> None:
        """Eş merkezli çemberleri, açısal ışınları ve etiketleri çiz (bir kez)."""
        grid_pen = pg.mkPen(120, 120, 120, 160, width=1)
        theta = np.linspace(0.0, 360.0, 181)
        for r in _GRID_RADII:
            x, y = _polar_to_xy(theta, np.full_like(theta, r))
            self.addItem(pg.PlotDataItem(x, y, pen=grid_pen))

        for ang in range(0, 360, _SPOKE_STEP_DEG):
            x, y = _polar_to_xy(
                np.array([ang, ang], dtype=float), np.array([0.0, 1.0])
            )
            self.addItem(pg.PlotDataItem(x, y, pen=grid_pen))
            lx, ly = _polar_to_xy(np.array([float(ang)]), np.array([_LABEL_RADIUS]))
            label = pg.TextItem(f"{ang}°", color=(180, 180, 180), anchor=(0.5, 0.5))
            label.setPos(float(lx[0]), float(ly[0]))
            self.addItem(label)

    def _radius(self, power_db: np.ndarray) -> np.ndarray:
        """Güç (dB) → [0, 1] yarıçapı (vmin/vmax aralığına göre kırpılır)."""
        span = max(self._vmax - self._vmin, 1e-9)
        return np.clip((power_db - self._vmin) / span, 0.0, 1.0)

    def set_data(self, angles_deg: np.ndarray, powers_db: np.ndarray) -> None:
        """Açı-güç haritasını güncelle (kutupsal eğri)."""
        angles = np.asarray(angles_deg, dtype=float)
        powers = np.asarray(powers_db, dtype=float)
        if angles.size == 0:
            self._curve.setData([], [])
            return
        x, y = _polar_to_xy(angles, self._radius(powers))
        self._curve.setData(x, y)

    def set_bearing(self, angle_deg: float) -> None:
        """Hedef yön ibresini (merkezden kenara çizgi) ayarla."""
        x, y = _polar_to_xy(
            np.array([float(angle_deg)]), np.array([1.0])
        )
        self._needle.setData([0.0, float(x[0])], [0.0, float(y[0])])

    def clear_data(self) -> None:
        """Eğri ve ibreyi temizle."""
        self._curve.setData([], [])
        self._needle.setData([], [])
