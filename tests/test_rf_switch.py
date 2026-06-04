"""hal.rf_switch + tx_worker EMNİYET testleri.

Doğrulananlar (CLAUDE.md KATI emniyet):
- RX→TX zorunlu sıralaması; yanlış sıra AssertionError ile engellenir.
- ≥10 µs guard dolmadan TX etkinleşemez.
- RX portuna TX yönlendirme bloklanır (yol TX değilken enable_tx engellenir).
- emergency_disable herhangi durumdan PA'yı keser, güvenli RX yoluna geçer.
- TX gücü yazılımsal limiti AŞAMAZ (kırpılır).
"""

from __future__ import annotations

import pytest

from eh_system.hal.mock_sdr import MockSDR
from eh_system.hal.rf_switch import GUARD_TIME_S, RFSwitch, SwitchState
from eh_system.workers.tx_worker import TXWorker


def _recording_switch() -> tuple[RFSwitch, list[str]]:
    """Callback çağrı sırasını kaydeden bir RFSwitch üret."""
    calls: list[str] = []
    sw = RFSwitch(
        rx_stop=lambda: calls.append("rx_stop"),
        rx_start=lambda: calls.append("rx_start"),
        set_path_tx=lambda: calls.append("path_tx"),
        set_path_rx=lambda: calls.append("path_rx"),
        tx_enable=lambda: calls.append("tx_enable"),
        tx_disable=lambda: calls.append("tx_disable"),
    )
    return sw, calls


def test_correct_sequence_engages_tx() -> None:
    """Doğru sıralı engage_tx TX_ACTIVE'e ulaşır ve callback sırası doğrudur."""
    sw, calls = _recording_switch()
    assert sw.state == SwitchState.RX_ACTIVE
    sw.engage_tx()
    assert sw.state == SwitchState.TX_ACTIVE
    # ZORUNLU sıra: rx_stop → path_tx → (guard) → tx_enable.
    assert calls == ["rx_stop", "path_tx", "tx_enable"]


def test_release_returns_to_rx() -> None:
    """release_tx güvenli TX→RX dizisini uygular."""
    sw, calls = _recording_switch()
    sw.engage_tx()
    calls.clear()
    sw.release_tx()
    assert sw.state == SwitchState.RX_ACTIVE
    assert calls == ["tx_disable", "path_rx", "rx_start"]


def test_enable_tx_without_stopping_rx_blocked() -> None:
    """RX aktifken doğrudan enable_tx → AssertionError (LNA emniyeti)."""
    sw, _ = _recording_switch()
    with pytest.raises(AssertionError):
        sw.enable_tx()


def test_switch_path_before_rx_stop_blocked() -> None:
    """RX durdurulmadan TX yoluna geçiş → AssertionError."""
    sw, _ = _recording_switch()
    with pytest.raises(AssertionError):
        sw.switch_to_tx_path()


def test_tx_into_rx_path_blocked() -> None:
    """Yol RX iken (RX_STOPPED) enable_tx engellenir (RX portuna TX yok)."""
    sw, _ = _recording_switch()
    sw.stop_rx()  # RX_STOPPED, yol hâlâ RX
    with pytest.raises(AssertionError):
        sw.enable_tx()


def test_guard_time_enforced() -> None:
    """Guard dolmadan enable_tx → AssertionError."""
    sw, _ = _recording_switch()
    sw.stop_rx()
    sw.switch_to_tx_path()  # switch zamanı kaydedilir
    # Guard beklemeden hemen TX denemesi engellenmeli.
    with pytest.raises(AssertionError):
        sw.enable_tx()


def test_guard_below_minimum_rejected() -> None:
    """10 µs altı guard ile RFSwitch kurmak engellenir."""
    with pytest.raises(AssertionError):
        RFSwitch(guard_s=GUARD_TIME_S / 2.0)


def test_emergency_disable_from_tx_active() -> None:
    """TX yayındayken emergency_disable PA'yı keser, güvenli RX yoluna geçer."""
    sw, calls = _recording_switch()
    sw.engage_tx()
    calls.clear()
    sw.emergency_disable()
    assert sw.state == SwitchState.RX_STOPPED
    assert "tx_disable" in calls and "path_rx" in calls


def test_emergency_disable_is_idempotent_when_not_tx() -> None:
    """TX değilken emergency_disable yine güvenli yola geçer, çökmeden."""
    sw, _ = _recording_switch()
    sw.emergency_disable()
    assert sw.state == SwitchState.RX_STOPPED


def test_tx_power_cannot_exceed_limit() -> None:
    """TX güç limitin üstüne ayarlansa bile çıkış genliği tam ölçeği aşamaz."""
    sw, _ = _recording_switch()
    sdr = MockSDR()
    worker = TXWorker(
        sdr, sw, tx_freq=433e6, sample_rate=10e6, tx_gain_db=20.0,
        power_limit_db=0.0, block_size=256,
    )
    # Tam ölçek dalga formu (genliği 1.0).
    import numpy as np
    worker.set_waveform(np.ones(256, dtype=np.complex64))

    # Limitin çok üstünde güç iste → genlik 1.0'da kırpılmalı.
    worker.set_power_db(50.0)
    block = worker._next_block()
    assert float(np.max(np.abs(block))) <= 1.0 + 1e-6

    # Limitin altında güç → genlik belirgin biçimde düşük.
    worker.set_power_db(-20.0)
    block = worker._next_block()
    assert float(np.max(np.abs(block))) < 0.2
