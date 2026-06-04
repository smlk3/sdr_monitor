"""Motorlu tripod kontrolü (360° tarama) — seri port veya mock.

Komut protokolü (basit ASCII, satır sonu '\\n'):
- ``M<derece>``  → mutlak konuma git (move_absolute)
- ``R<derece>``  → göreli dön (move_relative)
- ``S``          → dur (stop)
- ``?``          → açı sorgula; kontrolcü ``A<derece>`` ile yanıtlar

pyserial kurulu olmayabilir veya kontrolcü bağlı olmayabilir; ``SerialMotor``
yalnızca ``open()`` çağrılınca anlamlı hata verir. Donanımsız geliştirme için
``MockMotor`` aynı arayüzü taklit eder (``--mock``).

UI bilmez; worker üzerinden kullanılır. Her komut try/except ile sarılmalıdır
(çağıran tarafta), çökme olmamalıdır.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

# pyserial'i yumuşak içe aktar: yoksa modül yine de yüklenir.
try:
    import serial  # type: ignore

    _SERIAL_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - ortam bağımlı
    serial = None  # type: ignore
    _SERIAL_IMPORT_ERROR = exc

# Tam tur (derece) — açı normalizasyonu için.
_FULL_TURN_DEG = 360.0


def _wrap_deg(deg: float) -> float:
    """Açıyı [0, 360) aralığına sar."""
    return float(deg) % _FULL_TURN_DEG


class MotorBase(ABC):
    """Motor kontrolcüsü için soyut arayüz (seri veya mock)."""

    @abstractmethod
    def open(self) -> None:
        """Bağlantıyı aç; başarısızsa anlamlı istisna fırlat."""

    @abstractmethod
    def close(self) -> None:
        """Bağlantıyı kapat (idempotent)."""

    @abstractmethod
    def move_absolute(self, deg: float) -> None:
        """Mutlak açıya (derece) git."""

    @abstractmethod
    def move_relative(self, deg: float) -> None:
        """Bulunduğu açıdan ``deg`` derece göreli dön."""

    @abstractmethod
    def get_angle(self) -> float:
        """Geçerli açıyı (derece, [0, 360)) döndür."""

    @abstractmethod
    def stop(self) -> None:
        """Hareketi derhal durdur."""

    def __enter__(self) -> MotorBase:
        self.open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class SerialMotor(MotorBase):
    """pyserial üzerinden ASCII protokollü motor kontrolcüsü (Arduino/STM32)."""

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 115200,
        timeout_s: float = 1.0,
    ) -> None:
        self._port = str(port)
        self._baud = int(baudrate)
        self._timeout = float(timeout_s)
        self._ser = None
        self._angle = 0.0  # yerel takip (sorgu başarısızsa yedek)

    def open(self) -> None:
        if serial is None:
            raise RuntimeError(
                "pyserial bulunamadı. Motor için 'pyserial' kurun veya --mock "
                f"ile çalıştırın. Orijinal hata: {_SERIAL_IMPORT_ERROR!r}"
            )
        try:
            self._ser = serial.Serial(self._port, self._baud, timeout=self._timeout)
        except Exception as exc:
            raise RuntimeError(
                f"Motor portu açılamadı ({self._port}@{self._baud}). Hata: {exc}"
            ) from exc

    def close(self) -> None:
        try:
            if self._ser is not None:
                self._ser.close()
        finally:
            self._ser = None

    def _send(self, line: str) -> None:
        """Bir ASCII komut satırı gönder."""
        if self._ser is None:
            raise RuntimeError("Motor açık değil; önce open() çağırın.")
        self._ser.write((line + "\n").encode("ascii"))
        self._ser.flush()

    def move_absolute(self, deg: float) -> None:
        target = _wrap_deg(deg)
        self._send(f"M{target:.2f}")
        self._angle = target

    def move_relative(self, deg: float) -> None:
        self._send(f"R{float(deg):.2f}")
        self._angle = _wrap_deg(self._angle + deg)

    def get_angle(self) -> float:
        if self._ser is None:
            raise RuntimeError("Motor açık değil; önce open() çağırın.")
        try:
            self._ser.reset_input_buffer()
            self._send("?")
            resp = self._ser.readline().decode("ascii", errors="ignore").strip()
            if resp.startswith("A"):
                self._angle = _wrap_deg(float(resp[1:]))
        except (ValueError, OSError):
            # Sorgu başarısız → yerel takip değerini kullan (çökme yok).
            pass
        return self._angle

    def stop(self) -> None:
        self._send("S")


class MockMotor(MotorBase):
    """Donanımsız motor taklidi. Komutları yerel açı durumuna uygular.

    Mutlak/göreli hareketlerde gerçekçi bir gecikme (derece başına süre) ile
    kısa bir uyku ekler; böylece stop-and-measure taraması gerçekçi zamanlanır.
    """

    def __init__(self, deg_per_s: float = 180.0, max_settle_s: float = 0.05) -> None:
        self._angle = 0.0
        self._opened = False
        self._deg_per_s = float(deg_per_s)
        self._max_settle_s = float(max_settle_s)

    def open(self) -> None:
        self._opened = True

    def close(self) -> None:
        self._opened = False

    def _simulate_move(self, delta_deg: float) -> None:
        """Hareket süresini taklit et (üst sınırlı kısa uyku)."""
        if self._deg_per_s > 0.0:
            dt = abs(delta_deg) / self._deg_per_s
            time.sleep(min(dt, self._max_settle_s))

    def move_absolute(self, deg: float) -> None:
        if not self._opened:
            raise RuntimeError("MockMotor açık değil; önce open() çağırın.")
        target = _wrap_deg(deg)
        self._simulate_move(target - self._angle)
        self._angle = target

    def move_relative(self, deg: float) -> None:
        if not self._opened:
            raise RuntimeError("MockMotor açık değil; önce open() çağırın.")
        self._simulate_move(deg)
        self._angle = _wrap_deg(self._angle + deg)

    def get_angle(self) -> float:
        return self._angle

    def stop(self) -> None:
        # Mock'ta anlık dur; ek durum yok.
        pass
