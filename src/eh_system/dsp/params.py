"""Tespit edilen sinyallerin parametre çıkarımı (saf fonksiyonlar).

CFAR tespit maskesi üzerinden ``scipy.ndimage.label`` ile bitişik "adalar"
bulunur; her ada için:
- Merkez frekans: PSD (lineer güç) ağırlıklı centroid.
- -10 dB ve -20 dB bant genişliği: tepe noktasından dışa doğru ölçülür.
- Bant içi güç: lineer PSD'nin ada üzerinde integrali, dB(m) olarak.
- Analog/dijital sınıflandırma: bant-sınırlı IQ'nun anlık genlik/frekans
  varyansına dayalı BASİT bir sezgisel.

Tüm fonksiyonlar saftır (yan etkisiz), yalnızca numpy/scipy kullanır. Qt/UI bilmez.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_closing, find_objects, label

# log10(0) kaçınmak için taban (lineer güç).
_DB_FLOOR = 1e-20

# --- Sınıflandırma sezgisel eşikleri (deneysel, ayarlanabilir) ---
# Bant-sınırlı IQ'nun ANLIK FREKANSI (sarmalı açılmış fazın farkı, rad/örnek)
# üzerinden iki özellik:
#  1) std (etkinlik): CW'de ~0 (taşıyıcı sabit), FM/dijitalde anlamlı. Eşik
#     altındaysa sinyal CW gibi sabit frekanslı → "analog".
#  2) basıklık/kurtosis (darbesellik): dijital faz SIÇRAMALARI seyrek ve ani →
#     yüksek kurtosis; analog FM düzgün/sürekli → düşük kurtosis (~1.5-3).
# Dijital koşulu: yeterli etkinlik VE darbesel anlık frekans. Bu, CW'yi
# (etkinlik yok) ve FM'i (darbesel değil) doğru biçimde "analog" bırakır.
_IFREQ_ACTIVITY_FLOOR = 0.02
_IFREQ_KURTOSIS_DIGITAL = 6.0
# Sınıflandırma için gereken minimum örnek sayısı.
_CLASS_MIN_SAMPLES = 16

# Sınıf etiketleri.
TYPE_ANALOG = "analog"
TYPE_DIGITAL = "dijital"
TYPE_UNKNOWN = "belirsiz"


@dataclass(frozen=True)
class SignalParams:
    """Tek bir tespit edilen sinyalin (anlık, kare bazlı) parametreleri.

    Yan etkisizdir; izleme (ID, ilk/son görülme) üst katmanın sorumluluğudur.
    """

    center_freq: float   # PSD ağırlıklı merkez frekans (Hz)
    bw_10db: float       # -10 dB bant genişliği (Hz)
    bw_20db: float       # -20 dB bant genişliği (Hz)
    power_dbm: float     # bant içi integre güç (dBm; cal_offset_db'ye bağlı)
    peak_db: float       # ada içindeki tepe güç (dB)
    f_lo: float          # -20 dB alt kenar frekansı (Hz)
    f_hi: float          # -20 dB üst kenar frekansı (Hz)
    signal_type: str     # "analog" | "dijital" | "belirsiz"


def find_islands(detections: np.ndarray, close_gap: int = 0) -> list[tuple[int, int]]:
    """Bool tespit maskesindeki bitişik True bölgelerini (start, stop) döndür.

    ``scipy.ndimage.label`` ile etiketler; ``stop`` Python dilimi gibi dışlayıcıdır.

    Args:
        detections: Bool tespit maskesi.
        close_gap: >0 ise, ``2*close_gap+1`` uzunlukta yapı elemanıyla morfolojik
            kapama (closing) uygulanır; böylece tek bir sinyalin küçük güç
            düşüşleri yüzünden parçalanan adaları birleştirilir. 0 → kapama yok.
    """
    detections = np.asarray(detections, dtype=bool)
    if close_gap > 0 and detections.any():
        structure = np.ones(2 * close_gap + 1, dtype=bool)
        detections = binary_closing(detections, structure=structure)
    labeled, count = label(detections)
    if count == 0:
        return []
    slices = find_objects(labeled)
    return [(sl[0].start, sl[0].stop) for sl in slices]


def weighted_center_freq(
    freqs: np.ndarray, p_lin: np.ndarray, start: int, stop: int
) -> float:
    """Bir adanın lineer güç ağırlıklı merkez frekansını (centroid) hesapla."""
    seg_f = freqs[start:stop]
    seg_p = p_lin[start:stop]
    total = float(np.sum(seg_p))
    if total <= 0.0:
        # Dejenere durum: aritmetik orta.
        return float(np.mean(seg_f))
    return float(np.sum(seg_f * seg_p) / total)


def bandwidth_at(
    freqs: np.ndarray, psd_db: np.ndarray, peak_idx: int, drop_db: float
) -> tuple[float, float, float]:
    """Tepe noktasından dışa doğru -``drop_db`` seviyesindeki bant genişliği.

    Tepe etrafında, gücün ``peak - drop_db`` eşiğinin üstünde kaldığı bitişik
    bölgenin frekans genişliğini döndürür.

    Returns:
        (bw, f_lo, f_hi): bant genişliği (Hz) ve alt/üst kenar frekansları.
    """
    n = psd_db.shape[0]
    threshold = psd_db[peak_idx] - drop_db

    lo = peak_idx
    while lo > 0 and psd_db[lo - 1] >= threshold:
        lo -= 1
    hi = peak_idx
    while hi < n - 1 and psd_db[hi + 1] >= threshold:
        hi += 1

    f_lo = float(freqs[lo])
    f_hi = float(freqs[hi])
    return abs(f_hi - f_lo), f_lo, f_hi


def inband_power_dbm(
    p_lin: np.ndarray, start: int, stop: int, cal_offset_db: float = 0.0
) -> float:
    """Adadaki lineer güçleri integre edip dB(m) cinsine çevir.

    ``cal_offset_db`` mutlak kalibrasyon ofsetidir (varsayılan 0 → dBFS benzeri,
    göreli ölçek). Gerçek dBm için anten/LNA/ADC zinciri kalibre edilmelidir.
    """
    total = float(np.sum(p_lin[start:stop]))
    return 10.0 * np.log10(total + _DB_FLOOR) + cal_offset_db


def _extract_band_iq(
    iq: np.ndarray,
    sample_rate: float,
    center_freq: float,
    f_lo: float,
    f_hi: float,
) -> np.ndarray:
    """IQ'yu [f_lo, f_hi] bandına FFT maskesiyle sınırla; karmaşık dizi döndür.

    Frekans alanında bant dışı bölmeler sıfırlanır, IFFT ile zaman alanına
    dönülür. Taşıyıcı ofseti sınıflandırmayı bozmasın diye anlık frekansın
    ortalaması (taşıyıcı) çıkarıldığından ek bir baseband kaydırma gerekmez.
    """
    n = iq.shape[0]
    spec = np.fft.fftshift(np.fft.fft(iq))
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / sample_rate)) + center_freq
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    spec_band = np.where(mask, spec, 0.0)
    return np.fft.ifft(np.fft.ifftshift(spec_band))


def classify_signal(
    iq: np.ndarray,
    sample_rate: float,
    center_freq: float,
    f_lo: float,
    f_hi: float,
) -> str:
    """Bant-sınırlı IQ'dan basit analog/dijital sınıflandırma.

    Sezgisel: dijital modülasyonların (PSK/QAM/FSK) anlık frekansı sembol
    geçişlerinde ANİ sıçramalar yapar → yüksek kurtosis ve anlamlı etkinlik.
    CW'nin anlık frekansı neredeyse sabittir (etkinlik ~0); analog FM ise
    DÜZGÜN değişir (düşük kurtosis). Bu yüzden dijital koşulu hem yeterli
    etkinlik hem de darbesellik ister; aksi halde "analog".

    Yetersiz örnek/güç durumunda "belirsiz" döner. Saf fonksiyon; yalnızca
    numpy kullanır.
    """
    if iq.shape[0] < _CLASS_MIN_SAMPLES:
        return TYPE_UNKNOWN

    z = _extract_band_iq(iq, sample_rate, center_freq, f_lo, f_hi)
    mean_amp = float(np.mean(np.abs(z)))
    if mean_amp <= 0.0 or not np.isfinite(mean_amp):
        return TYPE_UNKNOWN

    # Anlık frekans = sarmalı açılmış fazın farkı (rad/örnek); taşıyıcı ofseti
    # ortalama çıkarılarak giderilir.
    phase = np.unwrap(np.angle(z))
    inst_freq = np.diff(phase)
    inst_freq = inst_freq - np.mean(inst_freq)
    ifreq_std = float(np.std(inst_freq))
    if ifreq_std <= 0.0:
        return TYPE_ANALOG
    # Normalize edilmiş dördüncü moment (basıklık).
    kurtosis = float(np.mean((inst_freq / ifreq_std) ** 4))

    if ifreq_std > _IFREQ_ACTIVITY_FLOOR and kurtosis > _IFREQ_KURTOSIS_DIGITAL:
        return TYPE_DIGITAL
    return TYPE_ANALOG


def extract_signals(
    freqs: np.ndarray,
    psd_db: np.ndarray,
    detections: np.ndarray,
    iq: np.ndarray | None = None,
    sample_rate: float = 1.0,
    center_freq: float = 0.0,
    cal_offset_db: float = 0.0,
    min_bins: int = 1,
    close_gap: int = 0,
) -> list[SignalParams]:
    """Tespit maskesinden sinyal listesi çıkar (saf, kare bazlı).

    Args:
        freqs: Hz cinsinden frekans ekseni (artan).
        psd_db: dB cinsinden güç spektrumu.
        detections: CFAR bool tespit maskesi.
        iq: Sınıflandırma için (isteğe bağlı) zaman alanı IQ; None ise tür
            "belirsiz" döner. PSD ile tutarlı olması için ``freqs`` ekseniyle
            aynı FFT boyutunu kapsamalıdır.
        sample_rate: Örnekleme hızı (Hz) — sınıflandırma için.
        center_freq: Merkez frekansı (Hz) — sınıflandırma için.
        cal_offset_db: Güç kalibrasyon ofseti (dBm için).
        min_bins: Bir adanın geçerli sayılması için minimum bölme sayısı.
        close_gap: Parçalanan adaları birleştirmek için morfolojik kapama
            yarıçapı (bkz. ``find_islands``).

    Returns:
        Merkez frekansa göre artan sıralı ``SignalParams`` listesi.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    psd_db = np.asarray(psd_db, dtype=np.float64)
    p_lin = 10.0 ** (psd_db / 10.0)

    signals: list[SignalParams] = []
    for start, stop in find_islands(detections, close_gap=close_gap):
        if stop - start < min_bins:
            continue

        peak_idx = start + int(np.argmax(psd_db[start:stop]))
        cf = weighted_center_freq(freqs, p_lin, start, stop)
        bw10, _, _ = bandwidth_at(freqs, psd_db, peak_idx, 10.0)
        bw20, f_lo, f_hi = bandwidth_at(freqs, psd_db, peak_idx, 20.0)
        power = inband_power_dbm(p_lin, start, stop, cal_offset_db)
        peak_db = float(psd_db[peak_idx])

        if iq is not None:
            stype = classify_signal(iq, sample_rate, center_freq, f_lo, f_hi)
        else:
            stype = TYPE_UNKNOWN

        signals.append(
            SignalParams(
                center_freq=cf,
                bw_10db=bw10,
                bw_20db=bw20,
                power_dbm=power,
                peak_db=peak_db,
                f_lo=f_lo,
                f_hi=f_hi,
                signal_type=stype,
            )
        )

    signals.sort(key=lambda s: s.center_freq)
    return signals
