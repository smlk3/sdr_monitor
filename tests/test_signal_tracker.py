"""workers.signal_tracker birim testleri: eşleştirme, ID kalıcılığı, onay."""

from __future__ import annotations

from eh_system.dsp.params import TYPE_ANALOG, SignalParams
from eh_system.workers.signal_tracker import SignalTracker


def _sig(cf: float, stype: str = TYPE_ANALOG) -> SignalParams:
    """Test için minimal SignalParams üret."""
    return SignalParams(
        center_freq=cf,
        bw_10db=10.0e3,
        bw_20db=20.0e3,
        power_dbm=-30.0,
        peak_db=-20.0,
        f_lo=cf - 10.0e3,
        f_hi=cf + 10.0e3,
        signal_type=stype,
    )


def test_confirmation_requires_min_hits() -> None:
    """Sinyal min_hits karede görülene dek onaylanmaz (döndürülmez)."""
    tr = SignalTracker(match_tol_hz=50e3, min_hits=3, max_misses=5)
    assert tr.update([_sig(100e6)], now=0.0) == []   # hit 1
    assert tr.update([_sig(100e6)], now=0.1) == []   # hit 2
    active = tr.update([_sig(100e6)], now=0.2)        # hit 3 → onaylı
    assert len(active) == 1
    assert active[0].params.center_freq == 100e6


def test_persistent_id_across_frames() -> None:
    """Aynı sinyal (yakın frekans) kareler boyunca aynı ID'yi korur."""
    tr = SignalTracker(match_tol_hz=50e3, min_hits=2, max_misses=5)
    tr.update([_sig(200e6)], now=0.0)
    a = tr.update([_sig(200.01e6)], now=0.1)
    b = tr.update([_sig(199.99e6)], now=0.2)
    assert len(a) == 1 and len(b) == 1
    assert a[0].id == b[0].id


def test_single_frame_false_alarm_never_confirmed() -> None:
    """Tek karelik (sonra kaybolan) yanlış alarm hiç onaylanmaz ve düşürülür."""
    tr = SignalTracker(match_tol_hz=50e3, min_hits=3, max_misses=2)
    assert tr.update([_sig(300e6)], now=0.0) == []   # tek hit
    # Sonraki karelerde yok → miss birikir, asla onaylanmaz.
    for i in range(5):
        assert tr.update([], now=0.1 * (i + 1)) == []


def test_dropped_after_max_misses() -> None:
    """Onaylı iz, max_misses ardışık kaçırma sonrası düşürülür."""
    tr = SignalTracker(match_tol_hz=50e3, min_hits=2, max_misses=2)
    tr.update([_sig(400e6)], now=0.0)
    tr.update([_sig(400e6)], now=0.1)  # onaylandı
    assert len(tr.update([], now=0.2)) == 1   # miss 1 → coast
    assert len(tr.update([], now=0.3)) == 1   # miss 2 → coast
    assert tr.update([], now=0.4) == []        # miss 3 → düşür


def test_distinct_signals_distinct_ids() -> None:
    """Tolerans dışındaki iki sinyal ayrı ID alır."""
    tr = SignalTracker(match_tol_hz=50e3, min_hits=1, max_misses=5)
    active = tr.update([_sig(500e6), _sig(600e6)], now=0.0)
    assert len(active) == 2
    assert active[0].id != active[1].id


def test_reset_clears_tracks() -> None:
    """reset() tüm izleri ve ID sayacını temizler."""
    tr = SignalTracker(match_tol_hz=50e3, min_hits=1, max_misses=5)
    tr.update([_sig(700e6)], now=0.0)
    tr.reset()
    assert tr.update([], now=0.1) == []
