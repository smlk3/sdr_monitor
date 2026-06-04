"""dsp.jammer birim testleri: baraj/çoklu-ton/sweep karıştırma üreteçleri."""

from __future__ import annotations

import numpy as np
import pytest

from eh_system.dsp.jammer import barrage_jam, multi_tone_jam, sweep_jam

_SR = 10.0e6


def _psd(iq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """fftshift edilmiş frekans ekseni (Hz) ve lineer güç döndür."""
    spec = np.fft.fftshift(np.fft.fft(iq))
    freqs = np.fft.fftshift(np.fft.fftfreq(iq.shape[0], d=1.0 / _SR))
    return freqs, np.abs(spec) ** 2


def test_barrage_shape_dtype_and_rms() -> None:
    """barrage_jam complex64, doğru uzunluk ve hedef RMS üretir."""
    iq = barrage_jam(8192, _SR, bandwidth_hz=2.0e6, amplitude=0.5, seed=0)
    assert iq.shape == (8192,)
    assert iq.dtype == np.complex64
    rms = float(np.sqrt(np.mean(np.abs(iq) ** 2)))
    assert abs(rms - 0.5) < 0.05


def test_barrage_is_band_limited() -> None:
    """Baraj gürültüsü enerjisinin çoğu istenen bant içinde olmalı."""
    bw = 2.0e6
    iq = barrage_jam(16384, _SR, bandwidth_hz=bw, amplitude=1.0, seed=1)
    freqs, power = _psd(iq)
    in_band = power[np.abs(freqs) <= bw / 2.0].sum()
    total = power.sum()
    # FIR geçiş bandı payı bırakılarak gevşek eşik.
    assert in_band / total > 0.8


def test_multi_tone_peaks_at_offsets() -> None:
    """multi_tone_jam belirtilen ofset frekanslarında tepe üretmeli."""
    offsets = [-1.0e6, 0.5e6]
    iq = multi_tone_jam(16384, _SR, offsets, amplitude=1.0)
    freqs, power = _psd(iq)
    for f_off in offsets:
        bin_idx = int(np.argmin(np.abs(freqs - f_off)))
        # Tepe komşuluğundaki güç, medyanın çok üstünde olmalı.
        local = power[bin_idx - 2 : bin_idx + 3].max()
        assert local > 100.0 * np.median(power)


def test_sweep_constant_envelope_and_coverage() -> None:
    """sweep_jam sabit zarflı olmalı ve süpürme bandını kapsamalı."""
    iq = sweep_jam(
        16384, _SR, f_start_hz=-2.0e6, f_stop_hz=2.0e6, sweep_time_s=1e-3, amplitude=0.8
    )
    assert iq.dtype == np.complex64
    env = np.abs(iq)
    assert abs(float(np.mean(env)) - 0.8) < 1e-3
    assert float(np.std(env)) < 1e-3
    # Enerji geniş banda yayılmalı (tek tona göre çok daha çok dolu bin).
    _freqs, power = _psd(iq)
    occupied = int(np.sum(power > 0.01 * power.max()))
    assert occupied > 50


@pytest.mark.parametrize(
    "call",
    [
        lambda: barrage_jam(0, _SR, 1e6),
        lambda: barrage_jam(1024, _SR, 0.0),
        lambda: barrage_jam(1024, _SR, _SR * 2),
        lambda: multi_tone_jam(1024, _SR, []),
        lambda: multi_tone_jam(1024, _SR, [_SR]),  # |ofset| >= Nyquist
        lambda: sweep_jam(1024, _SR, -1e6, 1e6, 0.0),
    ],
)
def test_jammer_invalid_params_raise(call) -> None:
    """Geçersiz parametreler ValueError fırlatmalı."""
    with pytest.raises(ValueError):
        call()
