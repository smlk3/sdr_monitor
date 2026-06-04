"""dsp.modulator birim testleri: NBFM sabit zarf + doğru sapma + ses zinciri.

Çekirdek doğrulama (Faz 4-5 / CLAUDE.md): NBFM çıktısı SABİT ZARFLI olmalı
(FM genlik bilgisi taşımaz) ve tepe frekans sapması istenen ``deviation``'a
eşit olmalı.
"""

from __future__ import annotations

import numpy as np
import pytest

from eh_system.dsp.modulator import (
    add_ctcss,
    bandpass_filter,
    nbfm_modulate,
    pre_emphasis,
    process_voice,
)

_SR = 48_000.0


def _tone(freq: float, n: int, sr: float = _SR) -> np.ndarray:
    """Tek frekanslı ses tonu üret."""
    t = np.arange(n) / sr
    return np.sin(2.0 * np.pi * freq * t)


def _inst_freq(iq: np.ndarray, sr: float) -> np.ndarray:
    """Anlık frekans (Hz) — sarmasız fazın türevi."""
    phase = np.unwrap(np.angle(iq))
    return np.diff(phase) * sr / (2.0 * np.pi)


def test_nbfm_constant_envelope() -> None:
    """NBFM çıktısı sabit zarflı olmalı (|IQ| ≈ 1, sapma ihmal edilebilir)."""
    audio = _tone(1000.0, 48_000)
    iq = nbfm_modulate(audio, _SR, deviation=2500.0)
    env = np.abs(iq)
    assert iq.dtype == np.complex64
    assert abs(float(np.mean(env)) - 1.0) < 1e-3
    assert float(np.std(env)) < 1e-3


def test_nbfm_deviation_is_correct() -> None:
    """Tepe frekans sapması istenen deviation'a eşit olmalı (tek tonda)."""
    deviation = 2500.0
    audio = _tone(1000.0, 96_000)  # tek tonun tepesi ±1 → tepe sapma = deviation
    iq = nbfm_modulate(audio, _SR, deviation=deviation)
    inst = _inst_freq(iq, _SR)
    peak_dev = float(np.max(np.abs(inst)))
    # Ayrık türev/normalizasyon nedeniyle gevşek tolerans (%5).
    assert abs(peak_dev - deviation) < 0.05 * deviation


def test_nbfm_higher_deviation_wider_swing() -> None:
    """Daha büyük sapma daha geniş anlık frekans salınımı üretmeli."""
    audio = _tone(1000.0, 96_000)
    p_narrow = float(np.max(np.abs(_inst_freq(nbfm_modulate(audio, _SR, 2000.0), _SR))))
    p_wide = float(np.max(np.abs(_inst_freq(nbfm_modulate(audio, _SR, 5000.0), _SR))))
    assert p_wide > p_narrow


def test_bandpass_attenuates_out_of_band() -> None:
    """Bandpass 300-3000 Hz: bant dışı (50 Hz) bant içine (1 kHz) göre zayıflar."""
    n = 48_000
    in_band = bandpass_filter(_tone(1000.0, n), _SR)
    out_band = bandpass_filter(_tone(50.0, n), _SR)
    # Geçici rejimi at, kalıcı bölgenin RMS'ini karşılaştır.
    rms_in = float(np.sqrt(np.mean(in_band[2000:] ** 2)))
    rms_out = float(np.sqrt(np.mean(out_band[2000:] ** 2)))
    assert rms_in > 5.0 * rms_out


def test_pre_emphasis_boosts_high_frequencies() -> None:
    """Pre-emphasis yüksek frekansı düşük frekansa göre yükseltmeli."""
    n = 48_000
    low = pre_emphasis(_tone(300.0, n), _SR)
    high = pre_emphasis(_tone(3000.0, n), _SR)
    gain_low = float(np.sqrt(np.mean(low**2))) / float(np.sqrt(np.mean(_tone(300.0, n) ** 2)))
    gain_high = float(np.sqrt(np.mean(high**2))) / float(np.sqrt(np.mean(_tone(3000.0, n) ** 2)))
    assert gain_high > gain_low


def test_add_ctcss_injects_subaudible_tone() -> None:
    """CTCSS, sessiz girişe alt-ses tonu ekler (enerji artar)."""
    silence = np.zeros(48_000)
    out = add_ctcss(silence, _SR, ctcss_freq_hz=88.5, level=0.2)
    assert float(np.max(np.abs(out))) > 0.1


def test_process_voice_constant_envelope() -> None:
    """Tam ses zinciri de sabit zarflı NBFM IQ üretmeli."""
    audio = _tone(1500.0, 48_000)
    iq = process_voice(audio, _SR, ctcss_freq_hz=88.5, deviation=2500.0)
    assert iq.dtype == np.complex64
    assert float(np.std(np.abs(iq))) < 1e-3


@pytest.mark.parametrize(
    "func,kwargs",
    [
        (nbfm_modulate, {"deviation": 0.0}),
        (pre_emphasis, {"tau": 0.0}),
    ],
)
def test_modulator_invalid_params_raise(func, kwargs: dict) -> None:
    """Geçersiz parametreler ValueError fırlatmalı."""
    audio = _tone(1000.0, 1024)
    with pytest.raises(ValueError):
        func(audio, _SR, **kwargs)
