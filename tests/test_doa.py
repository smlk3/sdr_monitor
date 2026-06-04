"""DF birim testleri: faz interferometri, kalibrasyon, motor, mock çift kanal.

Çekirdek doğrulama (CLAUDE.md / Faz 3): bilinen faz farkından beklenen geliş
açısı geri çıkarılmalı (ground truth). Ayrıca kalibrasyon, bilinen örnek
kayması + faz farkını geri kazanmalı ve λ/2 aşımında aliasing uyarısı verilmeli.
"""

from __future__ import annotations

import numpy as np
import pytest

from eh_system.dsp.calibration import apply_calibration, phase_calibration
from eh_system.dsp.doa import (
    doa_phase_interferometry,
    is_aliasing,
    phase_difference,
)
from eh_system.hal.mock_sdr import MockSDR
from eh_system.hal.motor import MockMotor

_C = 299_792_458.0


def _tone(n: int, f_frac: float = 0.1, seed: int = 0) -> np.ndarray:
    """Düşük gürültülü tek tonlu test sinyali (complex64)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    sig = np.exp(2j * np.pi * f_frac * t)
    sig = sig + 0.01 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    return sig.astype(np.complex64)


def _expected_phase(theta_deg: float, baseline_m: float, freq_hz: float) -> float:
    """Verilen açı/geometri için beklenen kanal faz farkı (rad)."""
    lam = _C / freq_hz
    return 2.0 * np.pi * baseline_m * np.sin(np.radians(theta_deg)) / lam


# --- dsp/doa.py ---

@pytest.mark.parametrize("theta_deg", [-40.0, -15.0, 0.0, 20.0, 35.0])
def test_doa_recovers_known_angle(theta_deg: float) -> None:
    """Bilinen faz farkından beklenen açı geri çıkarılmalı (ground truth)."""
    baseline_m, freq_hz, n = 0.15, 433.0e6, 4096
    rx0 = _tone(n, seed=1)
    phi = _expected_phase(theta_deg, baseline_m, freq_hz)
    rx1 = rx0 * np.exp(1j * phi)
    est = doa_phase_interferometry(rx0, rx1, baseline_m, freq_hz)
    assert abs(est - theta_deg) < 1.0


def test_phase_difference_sign() -> None:
    """phase_difference rx1'in rx0'a göre fazını doğru işaretle döndürür."""
    rx0 = _tone(2048, seed=2)
    rx1 = rx0 * np.exp(1j * 0.5)
    assert abs(phase_difference(rx0, rx1) - 0.5) < 0.05


def test_doa_aliasing_warns_above_half_wavelength() -> None:
    """d > λ/2 olduğunda aliasing uyarısı verilmeli."""
    freq_hz = 433.0e6
    lam = _C / freq_hz
    baseline_m = 0.8 * lam  # λ/2'nin epey üstünde
    rx0 = _tone(1024, seed=3)
    rx1 = rx0 * np.exp(1j * 0.3)
    assert is_aliasing(baseline_m, freq_hz)
    with pytest.warns(UserWarning, match="aliasing"):
        doa_phase_interferometry(rx0, rx1, baseline_m, freq_hz)


def test_doa_no_warning_within_half_wavelength() -> None:
    """d ≤ λ/2 iken uyarı verilmemeli."""
    freq_hz = 433.0e6
    baseline_m = 0.15
    assert not is_aliasing(baseline_m, freq_hz)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # herhangi bir uyarı testi düşürür
        doa_phase_interferometry(_tone(1024), _tone(1024), baseline_m, freq_hz)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"baseline_m": 0.0, "freq_hz": 433e6},   # baseline <= 0
        {"baseline_m": 0.15, "freq_hz": 0.0},    # freq <= 0
    ],
)
def test_doa_invalid_params_raise(kwargs: dict) -> None:
    """Geçersiz geometri ValueError fırlatmalı."""
    rx = _tone(256)
    with pytest.raises(ValueError):
        doa_phase_interferometry(rx, rx, **kwargs)


# --- dsp/calibration.py ---

def test_phase_calibration_recovers_shift_and_phase() -> None:
    """Bilinen örnek kayması + faz farkı geri kazanılmalı."""
    n, true_shift, true_phase = 4096, 5, 0.7
    rx0 = _tone(n, seed=4)
    rx1 = np.roll(rx0, true_shift) * np.exp(1j * true_phase)
    shift, phase = phase_calibration(rx0, rx1)
    assert shift == true_shift
    assert abs(phase - true_phase) < 0.05


def test_apply_calibration_aligns_channels() -> None:
    """apply_calibration kanalları hizalar (kalan faz ≈ 0, kayma giderilir)."""
    n, true_shift, true_phase = 4096, 3, -0.4
    rx0 = _tone(n, seed=5)
    rx1 = np.roll(rx0, true_shift) * np.exp(1j * true_phase)
    shift, phase = phase_calibration(rx0, rx1)
    rx1_corr = apply_calibration(rx1, shift, phase)
    # Hizalama sonrası kalan faz farkı ihmal edilebilir olmalı.
    assert abs(phase_difference(rx0, rx1_corr)) < 0.02


def test_calibration_mismatched_length_raises() -> None:
    """Farklı uzunlukta kanallar ValueError fırlatmalı."""
    with pytest.raises(ValueError):
        phase_calibration(_tone(256), _tone(128))


# --- MockSDR çift kanal + uçtan uca DF zinciri ---

def test_mock_dual_shapes_and_dtype() -> None:
    """read_samples_dual aynı uzunlukta iki complex64 kanal döndürür."""
    sdr = MockSDR(frequency=433e6, sample_rate=10e6)
    sdr.open()
    rx0, rx1 = sdr.read_samples_dual(2048)
    assert rx0.shape == (2048,) and rx1.shape == (2048,)
    assert rx0.dtype == np.complex64 and rx1.dtype == np.complex64
    sdr.close()


def test_mock_pointing_gain_peaks_at_source() -> None:
    """Anten kaynağa yöneldiğinde alınan sinyal gücü, uzağa göre yüksek olmalı."""
    true_doa = 25.0
    sdr = MockSDR(frequency=433e6, sample_rate=10e6, df_true_doa_deg=true_doa)
    sdr.open()

    def mean_power(angle: float) -> float:
        sdr.set_pointing_angle(angle)
        rx0, _ = sdr.read_samples_dual(4096)
        return float(np.mean(np.abs(rx0) ** 2))

    p_on = mean_power(true_doa)          # kaynağa yönelik
    p_off = mean_power(true_doa + 120.0)  # kaynaktan uzak
    assert p_on > p_off
    # None → yönlü kazanç yok (tam güç ≥ uzak yönelim).
    sdr.set_pointing_angle(None)
    rx0, _ = sdr.read_samples_dual(4096)
    assert float(np.mean(np.abs(rx0) ** 2)) > p_off
    sdr.close()


def test_mock_calibration_then_doa_end_to_end() -> None:
    """Referansta kalibre et, ölçümde DoA gerçek açıyı geri çıkarmalı."""
    true_doa, baseline_m, freq_hz = 25.0, 0.15, 433.0e6
    sdr = MockSDR(
        frequency=freq_hz,
        sample_rate=10e6,
        df_sample_shift=3,
        df_phase_offset_rad=0.6,
        df_true_doa_deg=true_doa,
        df_baseline_m=baseline_m,
    )
    sdr.open()

    # 1) Kalibrasyon: referans modu (yalnız donanımsal uyumsuzluk).
    sdr.set_calibration_reference(True)
    c0, c1 = sdr.read_samples_dual(8192)
    shift, phase = phase_calibration(c0, c1)
    assert shift == 3
    assert abs(phase - 0.6) < 0.1

    # 2) Ölçüm: gerçek yön fazı eklenir; kalibrasyon donanımsal kısmı giderir.
    sdr.set_calibration_reference(False)
    rx0, rx1 = sdr.read_samples_dual(8192)
    rx1_corr = apply_calibration(rx1, shift, phase)
    est = doa_phase_interferometry(rx0, rx1_corr, baseline_m, freq_hz)
    assert abs(est - true_doa) < 3.0

    sdr.close()


# --- hal/motor.py (MockMotor) ---

def test_mock_motor_absolute_and_wrap() -> None:
    """Mutlak hareket açıyı [0,360) aralığında ayarlar."""
    m = MockMotor(deg_per_s=1e9)  # uykuyu ihmal et
    m.open()
    m.move_absolute(90.0)
    assert m.get_angle() == 90.0
    m.move_absolute(370.0)  # sarılmalı → 10
    assert abs(m.get_angle() - 10.0) < 1e-9
    m.close()


def test_mock_motor_relative() -> None:
    """Göreli hareket bulunduğu açıya ekler ve sarar."""
    m = MockMotor(deg_per_s=1e9)
    m.open()
    m.move_absolute(350.0)
    m.move_relative(20.0)  # 370 → 10
    assert abs(m.get_angle() - 10.0) < 1e-9
    m.stop()
    m.close()


def test_mock_motor_requires_open() -> None:
    """Açılmadan hareket RuntimeError fırlatmalı."""
    m = MockMotor()
    with pytest.raises(RuntimeError):
        m.move_absolute(10.0)
