"""dsp.params birim testleri: ada bulma, merkez frekans, BW, güç, sınıflandırma."""

from __future__ import annotations

import numpy as np

from eh_system.dsp.params import (
    TYPE_ANALOG,
    TYPE_DIGITAL,
    bandwidth_at,
    classify_signal,
    extract_signals,
    find_islands,
    inband_power_dbm,
    weighted_center_freq,
)


def test_find_islands_contiguous_regions() -> None:
    """Bitişik True bölgeleri doğru (start, stop) çiftleri olarak bulunur."""
    det = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1], dtype=bool)
    islands = find_islands(det)
    assert islands == [(1, 3), (5, 6), (7, 10)]


def test_find_islands_empty() -> None:
    """Tespit yoksa boş liste döner."""
    assert find_islands(np.zeros(10, dtype=bool)) == []


def test_weighted_center_freq_symmetric() -> None:
    """Simetrik güç dağılımında centroid orta frekansa düşer."""
    freqs = np.array([100.0, 200.0, 300.0])
    p_lin = np.array([1.0, 4.0, 1.0])
    cf = weighted_center_freq(freqs, p_lin, 0, 3)
    assert abs(cf - 200.0) < 1e-9


def test_weighted_center_freq_skewed() -> None:
    """Ağırlık merkezi yüksek güçlü bölmeye kayar."""
    freqs = np.array([0.0, 10.0, 20.0])
    p_lin = np.array([1.0, 1.0, 8.0])
    cf = weighted_center_freq(freqs, p_lin, 0, 3)
    assert cf > 13.0  # 20'ye doğru kaymalı


def test_bandwidth_at_triangular_peak() -> None:
    """Tepe etrafında simetrik düşüşte -drop_db bant genişliği makul olmalı."""
    n = 101
    freqs = np.linspace(0.0, 100.0, n)  # 1 Hz/bin
    psd_db = np.full(n, -60.0)
    peak = 50
    # Tepe etrafında doğrusal düşüş: her bölmede -2 dB.
    for off in range(-15, 16):
        psd_db[peak + off] = 0.0 - 2.0 * abs(off)
    bw10, f_lo, f_hi = bandwidth_at(freqs, psd_db, peak, 10.0)
    # -10 dB → |off|*2 <= 10 → off in [-5, 5] → 10 bin genişlik.
    assert abs(bw10 - 10.0) < 1.5
    assert f_lo < freqs[peak] < f_hi


def test_inband_power_increases_with_signal() -> None:
    """Bant içi güç, daha güçlü bölmelerde artar."""
    p_lin = np.array([1.0, 1.0, 1.0, 1.0])
    weak = inband_power_dbm(p_lin, 0, 2)
    strong = inband_power_dbm(p_lin * 10.0, 0, 2)
    assert strong > weak
    assert abs((strong - weak) - 10.0) < 1e-6  # 10x güç → +10 dB


def test_inband_power_cal_offset() -> None:
    """Kalibrasyon ofseti doğrudan eklenir."""
    p_lin = np.ones(4)
    base = inband_power_dbm(p_lin, 0, 4, cal_offset_db=0.0)
    shifted = inband_power_dbm(p_lin, 0, 4, cal_offset_db=30.0)
    assert abs((shifted - base) - 30.0) < 1e-9


def test_classify_cw_is_analog() -> None:
    """Sabit zarflı CW tonu 'analog' sınıflanmalı.

    Bant, gerçek işlem hattındaki gibi tonun çevresinde dardır (-20 dB bandı);
    bu, FFT sızıntısı kuyruklarının yapay olarak yakalanmasını önler.
    """
    n = 1024
    fs = 1.0e6
    f0 = 1.5e5
    t = np.arange(n) / fs
    iq = np.exp(2j * np.pi * f0 * t).astype(np.complex64)
    stype = classify_signal(iq, fs, center_freq=0.0, f_lo=1.45e5, f_hi=1.55e5)
    assert stype == TYPE_ANALOG


def test_classify_fm_is_analog() -> None:
    """Düzgün (sinüzoidal mesajlı) dar bant FM 'analog' sınıflanmalı."""
    n = 4096
    fs = 1.0e6
    fc = 2.0e5
    f_msg = 5.0e3
    dev = 4.0e3  # modülasyon indeksi 0.8 → dar bant FM, düzgün anlık frekans
    t = np.arange(n) / fs
    phase = 2 * np.pi * fc * t + (dev / f_msg) * np.sin(2 * np.pi * f_msg * t)
    iq = np.exp(1j * phase).astype(np.complex64)
    stype = classify_signal(iq, fs, center_freq=0.0, f_lo=1.8e5, f_hi=2.2e5)
    assert stype == TYPE_ANALOG


def test_classify_qpsk_is_digital() -> None:
    """Genlik geçişli QPSK öbeği 'dijital' sınıflanmalı."""
    n = 4096
    fs = 1.0e6
    f0 = 2.0e5
    sps = 8
    rng = np.random.default_rng(0)
    sym = rng.integers(0, 4, size=n // sps + 1)
    sym_iq = np.exp(1j * (np.pi / 4 + sym * (np.pi / 2)))
    baseband = np.repeat(sym_iq, sps)[:n]
    t = np.arange(n) / fs
    iq = (baseband * np.exp(2j * np.pi * f0 * t)).astype(np.complex64)
    # Bant, sembol hızına göre genişçe seçilir.
    stype = classify_signal(iq, fs, center_freq=0.0, f_lo=1.0e5, f_hi=3.0e5)
    assert stype == TYPE_DIGITAL


def test_extract_signals_end_to_end() -> None:
    """İki ayrık tespit adasından iki sinyal çıkarılır, merkez f'e göre sıralı."""
    n = 256
    freqs = np.linspace(-5.0e5, 5.0e5, n)
    psd_db = np.full(n, -80.0)
    psd_db[60:64] = -20.0   # birinci ada
    psd_db[180:184] = -15.0  # ikinci ada
    det = np.zeros(n, dtype=bool)
    det[60:64] = True
    det[180:184] = True
    signals = extract_signals(freqs, psd_db, det, iq=None, sample_rate=1.0e6)
    assert len(signals) == 2
    assert signals[0].center_freq < signals[1].center_freq
    for s in signals:
        assert s.bw_20db >= s.bw_10db  # -20 dB bandı en az -10 dB kadar geniş
