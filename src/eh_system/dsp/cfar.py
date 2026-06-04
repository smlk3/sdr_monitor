"""CFAR (Constant False Alarm Rate) tespit fonksiyonları.

Tüm fonksiyonlar saftır (yan etkisiz), yalnızca numpy/scipy kullanır ve pytest
ile test edilebilir. Qt/UI bilmez.

CFAR mantığı lineer güç (mW benzeri) ölçeğinde yürür: girişteki ``psd_db``
``10**(psd_db/10)`` ile lineer güce çevrilir, eğitim (training) hücrelerinin
ortalaması alınır, ölçeklenir ve test altındaki hücre (CUT) bu eşikle
karşılaştırılır. Eşik, raporlamada kolaylık için tekrar dB'ye çevrilerek döner.

CA-CFAR vektörizedir: hücre başına ``for`` döngüsü yoktur; eğitim hücreleri
toplamı ``scipy.ndimage.convolve1d`` ile tek seferde hesaplanır.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import convolve1d

# log10(0) kaçınmak için taban (lineer güç).
_DB_FLOOR = 1e-20
# Kenar (edge) doldurma kipi: convolve1d ve np.pad için.
_EDGE_MODE = "reflect"


def _validate(n: int, num_train: int, num_guard: int, pfa: float) -> None:
    """CFAR parametrelerini doğrula; hatalıysa ValueError fırlat."""
    if num_train <= 0:
        raise ValueError("num_train pozitif olmalı.")
    if num_guard < 0:
        raise ValueError("num_guard negatif olamaz.")
    if not (0.0 < pfa < 1.0):
        raise ValueError("pfa (0, 1) aralığında olmalı.")
    window = 2 * (num_train + num_guard) + 1
    if n < window:
        raise ValueError(
            f"PSD uzunluğu ({n}) CFAR penceresinden ({window}) küçük olamaz."
        )


def _ca_alpha(num_train_cells: int, pfa: float) -> float:
    """CA-CFAR eşik ölçek katsayısı (alpha).

    Üstel (Rayleigh genlik → üstel güç) gürültü varsayımı altında klasik formül:
        alpha = N * (Pfa^(-1/N) - 1)
    Burada N toplam eğitim hücresi sayısıdır.
    """
    return num_train_cells * (pfa ** (-1.0 / num_train_cells) - 1.0)


def _train_kernel(num_train: int, num_guard: int) -> np.ndarray:
    """Eğitim hücrelerini toplayan, guard + CUT bölgesini sıfırlayan çekirdek.

    Uzunluk: ``2*(num_train+num_guard)+1``. Merkezdeki ``2*num_guard+1`` hücre
    (guard'lar + test hücresi) sıfır; iki yandaki ``num_train`` hücre birdir.
    Simetrik olduğu için konvolüsyon/korelasyon ayrımı önemsizdir.
    """
    half = num_train + num_guard
    kernel = np.ones(2 * half + 1, dtype=np.float64)
    kernel[num_train : num_train + 2 * num_guard + 1] = 0.0
    return kernel


def ca_cfar(
    psd_db: np.ndarray,
    num_train: int,
    num_guard: int,
    pfa: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Vektörize Cell-Averaging CFAR (CA-CFAR).

    Args:
        psd_db: dB cinsinden güç spektrumu (1B).
        num_train: Her bir yandaki eğitim hücresi sayısı (toplam 2*num_train).
        num_guard: Her bir yandaki koruma (guard) hücresi sayısı.
        pfa: Hedef yanlış alarm olasılığı (0, 1).

    Returns:
        (detections_bool, threshold_db):
            detections_bool: ``psd_db`` ile aynı uzunlukta bool dizi; tespitler True.
            threshold_db: dB cinsinden hücre başına CFAR eşiği (aynı uzunluk).
    """
    psd_db = np.asarray(psd_db, dtype=np.float64)
    if psd_db.ndim != 1:
        raise ValueError("psd_db 1 boyutlu olmalı.")
    _validate(psd_db.shape[0], num_train, num_guard, pfa)

    p_lin = 10.0 ** (psd_db / 10.0)
    kernel = _train_kernel(num_train, num_guard)
    num_train_cells = 2 * num_train

    # Eğitim hücrelerinin toplamı — tek vektörize çağrı, hücre döngüsü yok.
    train_sum = convolve1d(p_lin, kernel, mode=_EDGE_MODE)
    noise_lin = train_sum / num_train_cells

    alpha = _ca_alpha(num_train_cells, pfa)
    threshold_lin = alpha * noise_lin

    detections = p_lin > threshold_lin
    threshold_db = 10.0 * np.log10(threshold_lin + _DB_FLOOR)

    return detections, threshold_db.astype(np.float64)


def os_cfar(
    psd_db: np.ndarray,
    num_train: int,
    num_guard: int,
    pfa: float,
    rank_frac: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    """Ordered-Statistic CFAR (OS-CFAR) — iskelet/uygulanabilir sürüm.

    CA-CFAR'ın aksine gürültü kestirimi olarak eğitim hücrelerinin ORTALAMASI
    değil, sıralı k. değeri (order statistic) kullanılır. Bu, bant içinde başka
    güçlü hedefler (interferer) varken eşik kaymasını engeller.

    Kayan pencere ``numpy.lib.stride_tricks.sliding_window_view`` ile kurulur,
    k. sıralı değer ``np.partition`` ile O(N) bulunur — tam sıralama yapılmaz.

    NOT: OS-CFAR'ın tam eşik katsayısı, Pfa'yı sağlayan kapalı-form çözümü
    olmadığından sayısal kök bulma gerektirir. Burada iskelet amacıyla CA-CFAR
    alpha formülü pragmatik bir yaklaşıklama olarak kullanılır; üretimde
    OS-CFAR'a özgü katsayı sayısal olarak çözülmelidir.

    Args:
        psd_db: dB cinsinden güç spektrumu (1B).
        num_train: Her yandaki eğitim hücresi sayısı.
        num_guard: Her yandaki koruma hücresi sayısı.
        pfa: Hedef yanlış alarm olasılığı.
        rank_frac: Kullanılacak sıralı değerin konumu (0..1). 0.75 → üst çeyrek.

    Returns:
        (detections_bool, threshold_db): ``ca_cfar`` ile aynı sözleşme.
    """
    psd_db = np.asarray(psd_db, dtype=np.float64)
    if psd_db.ndim != 1:
        raise ValueError("psd_db 1 boyutlu olmalı.")
    _validate(psd_db.shape[0], num_train, num_guard, pfa)
    if not (0.0 < rank_frac < 1.0):
        raise ValueError("rank_frac (0, 1) aralığında olmalı.")

    p_lin = 10.0 ** (psd_db / 10.0)
    half = num_train + num_guard
    window = 2 * half + 1

    # Kenarları yansıtarak pad'le ki her hücre tam pencereye sahip olsun.
    padded = np.pad(p_lin, half, mode=_EDGE_MODE)
    # Şekil: (n, window) — her satır bir CUT'ın penceresi.
    win_view = sliding_window_view(padded, window)

    # Guard + CUT hücrelerini eğitimden çıkar.
    train_mask = np.ones(window, dtype=bool)
    train_mask[num_train : num_train + 2 * num_guard + 1] = False
    train = win_view[:, train_mask]  # (n, 2*num_train)

    num_train_cells = train.shape[1]
    k = int(round(rank_frac * (num_train_cells - 1)))
    k = min(max(k, 0), num_train_cells - 1)

    # k. sıralı değer — tam sıralama yerine partition.
    kth = np.partition(train, k, axis=1)[:, k]

    alpha = _ca_alpha(num_train_cells, pfa)
    threshold_lin = alpha * kth

    detections = p_lin > threshold_lin
    threshold_db = 10.0 * np.log10(threshold_lin + _DB_FLOOR)

    return detections, threshold_db.astype(np.float64)
