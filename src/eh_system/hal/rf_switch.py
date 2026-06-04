"""RF switch (RX/TX yol) kontrolü — EMNİYET KRİTİK sıralama.

CLAUDE.md emniyet kuralı (KATI):
    RF switch sıralaması ZORUNLU: RX durdur → switch → ≥10 µs guard → TX.
    Bu sırayı bozmak LNA'yı yakar. Yanlış sıra ``assert`` ile engellenir.
    RX portuna TX yönlendiren komut dizisi kod seviyesinde bloklanır.

Durum makinesi (geçişler yalnız doğru sırada izinli; aksi ``AssertionError``):

    RX_ACTIVE ──stop_rx──► RX_STOPPED ──switch_to_tx_path──► TX_SWITCHED
        ▲                                                        │
        │                                              (≥10µs guard) enable_tx
        │                                                        ▼
    start_rx ◄─ RX_STOPPED ◄─switch_to_rx_path─ TX_SWITCHED ◄─disable_tx─ TX_ACTIVE

Donanım eylemleri (GPIO, PA, akış) callback'lerle yapılır; verilmezse mock
(no-op) olarak çalışır → donanımsız geliştirme. ``emergency_disable`` herhangi
bir durumdan güvenli RX yoluna anında geçer (acil durdur için).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from enum import Enum, auto

# RX→TX geçişinde anten/switch oturma koruma süresi (s). ≥10 µs ZORUNLU.
GUARD_TIME_S = 10.0e-6


class SwitchState(Enum):
    """RF switch durum makinesi durumları."""

    RX_ACTIVE = auto()    # RX çalışıyor, anten → LNA/RX (güvenli varsayılan)
    RX_STOPPED = auto()   # RX durduruldu, yol hâlâ RX
    TX_SWITCHED = auto()  # yol TX'e alındı (guard sayılıyor / TX henüz kapalı)
    TX_ACTIVE = auto()    # TX yayında


# İsteğe bağlı donanım callback türü (argümansız eylem).
_Action = Callable[[], None]


def _noop() -> None:
    """Donanım yokken kullanılan boş eylem."""


class RFSwitch:
    """RX/TX yol switch'i; zorunlu sıralamayı durum makinesiyle dayatır.

    Args:
        rx_stop: RX akışını durduran donanım eylemi.
        rx_start: RX akışını başlatan donanım eylemi.
        set_path_tx: Switch'i TX yoluna alan donanım eylemi.
        set_path_rx: Switch'i RX yoluna alan donanım eylemi.
        tx_enable: PA/TX'i etkinleştiren donanım eylemi.
        tx_disable: PA/TX'i kapatan donanım eylemi.
        guard_s: RX→TX koruma süresi (s); GUARD_TIME_S'ten küçük OLAMAZ.
    """

    def __init__(
        self,
        rx_stop: _Action | None = None,
        rx_start: _Action | None = None,
        set_path_tx: _Action | None = None,
        set_path_rx: _Action | None = None,
        tx_enable: _Action | None = None,
        tx_disable: _Action | None = None,
        guard_s: float = GUARD_TIME_S,
    ) -> None:
        # Guard süresi asla zorunlu minimumun altına inemez (emniyet).
        assert guard_s >= GUARD_TIME_S, "guard_s ≥ 10 µs olmalı (LNA emniyeti)."
        self._guard_s = float(guard_s)
        self._rx_stop = rx_stop or _noop
        self._rx_start = rx_start or _noop
        self._set_path_tx = set_path_tx or _noop
        self._set_path_rx = set_path_rx or _noop
        self._tx_enable = tx_enable or _noop
        self._tx_disable = tx_disable or _noop
        self._state = SwitchState.RX_ACTIVE
        self._switch_time = 0.0  # TX yoluna geçiş anı (guard ölçümü için)

    @property
    def state(self) -> SwitchState:
        """Geçerli switch durumu."""
        return self._state

    # --- Düşük seviye, sıralı geçişler (her biri ön koşulu assert eder) ---

    def stop_rx(self) -> None:
        """RX akışını durdur (TX'e geçişin ZORUNLU ilk adımı)."""
        assert self._state == SwitchState.RX_ACTIVE, (
            f"stop_rx yalnız RX_ACTIVE'de izinli (şu an {self._state.name})."
        )
        self._rx_stop()
        self._state = SwitchState.RX_STOPPED

    def switch_to_tx_path(self) -> None:
        """Switch'i TX yoluna al. RX DURDURULMADAN çağrılırsa engellenir.

        RX aktifken switch'i TX'e almak anteni doğrudan PA çıkışına/RX girişine
        bağlayabilir → LNA yanar. Bu yüzden ön koşul RX_STOPPED'tır.
        """
        assert self._state == SwitchState.RX_STOPPED, (
            f"switch_to_tx_path için önce RX durdurulmalı (şu an {self._state.name})."
        )
        self._set_path_tx()
        self._switch_time = time.perf_counter()
        self._state = SwitchState.TX_SWITCHED

    def enable_tx(self) -> None:
        """TX/PA'yı etkinleştir. Yol TX değilse VEYA guard dolmadıysa engellenir.

        Bu, RX portuna/yoluna TX yönlendirilmesini kod seviyesinde bloklar:
        yalnız yol TX'teyken (TX_SWITCHED) ve ≥guard geçtikten sonra izinlidir.
        """
        assert self._state == SwitchState.TX_SWITCHED, (
            f"enable_tx için yol TX olmalı (şu an {self._state.name}). "
            "RX yoluna TX engellenir."
        )
        elapsed = time.perf_counter() - self._switch_time
        assert elapsed >= self._guard_s, (
            f"Guard süresi dolmadı ({elapsed * 1e6:.1f} µs < "
            f"{self._guard_s * 1e6:.1f} µs). LNA emniyeti."
        )
        self._tx_enable()
        self._state = SwitchState.TX_ACTIVE

    def disable_tx(self) -> None:
        """TX/PA'yı kapat (yol TX'te kalır)."""
        assert self._state == SwitchState.TX_ACTIVE, (
            f"disable_tx yalnız TX_ACTIVE'de izinli (şu an {self._state.name})."
        )
        self._tx_disable()
        self._state = SwitchState.TX_SWITCHED

    def switch_to_rx_path(self) -> None:
        """Switch'i RX yoluna al (TX kapalı olmalı)."""
        assert self._state == SwitchState.TX_SWITCHED, (
            f"switch_to_rx_path için önce TX kapatılmalı (şu an {self._state.name})."
        )
        self._set_path_rx()
        self._state = SwitchState.RX_STOPPED

    def start_rx(self) -> None:
        """RX akışını yeniden başlat (yol RX olmalı)."""
        assert self._state == SwitchState.RX_STOPPED, (
            f"start_rx yalnız RX_STOPPED'da izinli (şu an {self._state.name})."
        )
        self._rx_start()
        self._state = SwitchState.RX_ACTIVE

    # --- Yüksek seviye, güvenli diziler ---

    def engage_tx(self) -> None:
        """Tam güvenli RX→TX dizisi: durdur → switch → guard → TX."""
        self.stop_rx()
        self.switch_to_tx_path()
        self._wait_guard()
        self.enable_tx()

    def release_tx(self) -> None:
        """Tam güvenli TX→RX dizisi: TX kapat → switch → RX başlat."""
        self.disable_tx()
        self.switch_to_rx_path()
        self.start_rx()

    def emergency_disable(self) -> None:
        """ACİL: herhangi bir durumdan TX'i kes ve güvenli RX yoluna geç.

        Sıralama assert'lerini ATLAR (acil durdur önceliklidir): TX aktifse
        derhal kapatılır, yol RX'e alınır. RX'i otomatik başlatmaz (operatör
        kontrolünde). <500 ms hedefiyle senkron ve hızlıdır.
        """
        if self._state == SwitchState.TX_ACTIVE:
            self._tx_disable()
        # Anteni güvenli (RX) yola al; RX akışı başlatılmaz.
        self._set_path_rx()
        self._state = SwitchState.RX_STOPPED

    def force_rx_active(self) -> None:
        """Herhangi bir durumdan güvenli RX_ACTIVE durumuna kurtar (recovery).

        Acil durdur sonrası (RX_STOPPED) veya tutarsız durumda yeni bir TX
        dizisine başlamadan önce güvenli başlangıç durumunu sağlar: TX açıksa
        kapatır, yolu RX'e alır, RX'i başlatır. Bu bir emniyet kurtarma yoludur,
        sıra assert'lerini atlar.
        """
        if self._state == SwitchState.TX_ACTIVE:
            self._tx_disable()
        self._set_path_rx()
        self._rx_start()
        self._state = SwitchState.RX_ACTIVE

    def _wait_guard(self) -> None:
        """Switch sonrası guard süresini hassas biçimde bekle (busy-wait)."""
        target = self._switch_time + self._guard_s
        while time.perf_counter() < target:
            pass
