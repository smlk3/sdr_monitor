"""dsp.cfar birim testleri: CA-CFAR tespiti ve PFA davranışı.

Sentetik PSD: üstel (Rayleigh genlik) gürültü tabanı üstüne bilinen bir pik
eklenir. Beklentiler:
- CA-CFAR pikleri tespit etmeli, eşik gürültü tabanının üstünde oturmalı.
- PFA küçüldükçe (daha katı) yanlış alarm oranı düşmeli; büyüdükçe artmalı.
"""

from __future__ import annotations

import numpy as np
import pytest

from eh_system.dsp.cfar import ca_cfar, os_cfar


def _noise_psd_db(n: int, seed: int = 0) -> np.ndarray:
    """Üstel güç dağılımlı (gürültü) sentetik PSD'yi dB cinsinden üret."""
    rng = np.random.default_rng(seed)
    # Üstel dağılımlı lineer güç (ortalama 1.0) → gerçekçi CFAR gürültü tabanı.
    p_lin = rng.exponential(scale=1.0, size=n)
    return 10.0 * np.log10(p_lin + 1e-20)


def test_ca_cfar_output_shapes_and_types() -> None:
    """ca_cfar girişle aynı uzunlukta bool maske ve dB eşik döndürür."""
    psd = _noise_psd_db(512)
    det, thr = ca_cfar(psd, num_train=16, num_guard=4, pfa=1e-3)
    assert det.shape == psd.shape
    assert thr.shape == psd.shape
    assert det.dtype == bool
    assert np.all(np.isfinite(thr))


def test_ca_cfar_detects_known_peak() -> None:
    """Gürültü tabanına eklenen güçlü pik tespit edilmeli."""
    n = 1024
    psd = _noise_psd_db(n, seed=1)
    peak_bin = 400
    # Tabanın ~40 dB üstünde bir pik.
    psd[peak_bin] = 40.0
    det, thr = ca_cfar(psd, num_train=16, num_guard=4, pfa=1e-4)
    assert det[peak_bin]
    # Eşik pikin altında, ama gürültü tabanının (medyan) üstünde olmalı.
    assert thr[peak_bin] < psd[peak_bin]
    assert thr[peak_bin] > np.median(psd)


def test_ca_cfar_threshold_above_noise_floor() -> None:
    """Saf gürültüde tespit oranı PFA mertebesinde, eşik taban üstünde kalmalı."""
    n = 8192
    psd = _noise_psd_db(n, seed=2)
    det, _ = ca_cfar(psd, num_train=32, num_guard=4, pfa=1e-3)
    false_alarm_rate = float(np.mean(det))
    # Tasarım PFA'sı 1e-3; gözlenen oran aynı mertebede (gevşek üst sınır).
    assert false_alarm_rate < 0.02


def test_ca_cfar_pfa_monotonic_false_alarms() -> None:
    """Daha büyük PFA, saf gürültüde daha çok (≥) yanlış alarm üretmeli."""
    n = 16384
    psd = _noise_psd_db(n, seed=3)
    rate_strict = float(np.mean(ca_cfar(psd, 32, 4, pfa=1e-5)[0]))
    rate_mid = float(np.mean(ca_cfar(psd, 32, 4, pfa=1e-3)[0]))
    rate_loose = float(np.mean(ca_cfar(psd, 32, 4, pfa=1e-1)[0]))
    # Yanlış alarm oranı PFA ile monoton artmalı.
    assert rate_strict <= rate_mid <= rate_loose
    # Uç değerler beklenen yönde belirgin biçimde ayrışmalı.
    assert rate_loose > rate_strict


def test_ca_cfar_higher_pfa_threshold_lower() -> None:
    """Daha büyük PFA daha düşük eşik (daha hassas) üretmeli."""
    psd = _noise_psd_db(2048, seed=4)
    _, thr_strict = ca_cfar(psd, 32, 4, pfa=1e-6)
    _, thr_loose = ca_cfar(psd, 32, 4, pfa=1e-2)
    # Aynı gürültü kestiriminde gevşek PFA eşiği daha aşağıda olmalı.
    assert np.mean(thr_loose) < np.mean(thr_strict)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_train": 0, "num_guard": 4, "pfa": 1e-3},   # num_train <= 0
        {"num_train": 16, "num_guard": -1, "pfa": 1e-3},  # num_guard < 0
        {"num_train": 16, "num_guard": 4, "pfa": 0.0},    # pfa sınır dışı
        {"num_train": 16, "num_guard": 4, "pfa": 1.0},    # pfa sınır dışı
    ],
)
def test_ca_cfar_invalid_params_raise(kwargs: dict) -> None:
    """Geçersiz parametreler ValueError fırlatmalı."""
    psd = _noise_psd_db(256)
    with pytest.raises(ValueError):
        ca_cfar(psd, **kwargs)


def test_ca_cfar_window_too_large_raises() -> None:
    """CFAR penceresi PSD'den büyükse ValueError."""
    psd = _noise_psd_db(20)
    with pytest.raises(ValueError):
        ca_cfar(psd, num_train=16, num_guard=4, pfa=1e-3)


def test_os_cfar_detects_peak_with_interferer() -> None:
    """OS-CFAR, bantta ikinci güçlü hedef varken bile zayıf piki tespit eder."""
    n = 1024
    psd = _noise_psd_db(n, seed=5)
    target_bin = 300
    interferer_bin = 320  # eğitim penceresine düşecek güçlü girişimci
    psd[target_bin] = 25.0
    psd[interferer_bin] = 45.0
    det, thr = os_cfar(psd, num_train=16, num_guard=4, pfa=1e-3)
    assert det[target_bin]
    assert det[interferer_bin]
    assert thr.shape == psd.shape


def test_os_cfar_rank_frac_validation() -> None:
    """rank_frac (0,1) dışındaysa ValueError."""
    psd = _noise_psd_db(256)
    with pytest.raises(ValueError):
        os_cfar(psd, 16, 4, pfa=1e-3, rank_frac=1.5)
