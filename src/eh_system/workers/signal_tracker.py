"""Kare-arası sinyal izleyici (stateful) — hit/miss onaylı.

``dsp.params.extract_signals`` her karede yan etkisiz, anlık bir sinyal listesi
üretir. Bu izleyici ardışık kareleri merkez frekans yakınlığına göre eşleştirip
kalıcı bir ID ve ilk/son görülme zaman damgası tutar.

Tek karelik CFAR yanlış alarmlarının "sinyal" gibi görünmesini önlemek için
M-of-N onay mantığı kullanılır (radar pratiği):
- Bir iz ancak ``min_hits`` kez eşleştikten sonra "onaylı" (gösterilebilir) olur.
- ``max_misses`` ardışık kare boyunca eşleşmeyen iz düşürülür.
Böylece nadir, rastgele frekanslı yanlış alarmlar asla onaylanmadan elenir;
gerçek sinyaller birkaç karede onaylanır ve kaybolunca kısa sürede coast edilip
düşürülür.

Saf DSP değildir (durum taşır); bu yüzden ``dsp/`` yerine ``workers/`` katmanında.
Qt bilmez.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..dsp.params import SignalParams


@dataclass(frozen=True)
class TrackedSignal:
    """Kalıcı ID ve görülme zamanlarıyla izlenen (onaylı) sinyal."""

    id: int
    params: SignalParams
    first_seen: float  # epoch saniye
    last_seen: float   # epoch saniye


# İç tutulan, değiştirilebilir iz kaydı.
@dataclass
class _Track:
    id: int
    params: SignalParams
    first_seen: float
    last_seen: float
    hits: int      # toplam eşleşme sayısı
    misses: int    # ardışık eşleşmeme sayısı


class SignalTracker:
    """Merkez frekans yakınlığına göre kareler arası sinyal eşleştirme + onay.

    Args:
        match_tol_hz: İki tespitin aynı sinyal sayılması için izin verilen
            merkez frekans farkı (Hz).
        min_hits: Bir izin "onaylı" (döndürülen listeye dahil) olması için
            gereken minimum toplam eşleşme sayısı.
        max_misses: Bir izin düşürülmeden önce dayanabileceği ardışık
            eşleşmeme (kare) sayısı.
    """

    def __init__(
        self,
        match_tol_hz: float = 90.0e3,
        min_hits: int = 3,
        max_misses: int = 5,
    ) -> None:
        self._match_tol_hz = float(match_tol_hz)
        self._min_hits = int(min_hits)
        self._max_misses = int(max_misses)
        self._tracks: list[_Track] = []
        self._next_id = 1

    def update(self, signals: list[SignalParams], now: float) -> list[TrackedSignal]:
        """Yeni kare sinyalleriyle izleri güncelle; ONAYLI izleri döndür.

        Eşleştirme açgözlüdür: her yeni sinyal, henüz eşleşmemiş en yakın
        (tolerans içindeki) ize bağlanır; eşleşme yoksa yeni (henüz onaysız) iz
        açar. Eşleşmeyen mevcut izlerin ``misses`` sayacı artar; ``max_misses``
        aşılınca düşürülür. Yalnızca ``hits >= min_hits`` olan izler döndürülür.
        """
        unmatched = list(self._tracks)
        survivors: list[_Track] = []

        for sig in signals:
            best: _Track | None = None
            best_dist = self._match_tol_hz
            for tr in unmatched:
                dist = abs(tr.params.center_freq - sig.center_freq)
                if dist <= best_dist:
                    best_dist = dist
                    best = tr
            if best is not None:
                unmatched.remove(best)
                best.params = sig
                best.last_seen = now
                best.hits += 1
                best.misses = 0
                survivors.append(best)
            else:
                survivors.append(
                    _Track(
                        id=self._next_id,
                        params=sig,
                        first_seen=now,
                        last_seen=now,
                        hits=1,
                        misses=0,
                    )
                )
                self._next_id += 1

        # Eşleşmeyen izler: kaçırma sayacını artır, sınırı aşmadıysa coast et.
        for tr in unmatched:
            tr.misses += 1
            if tr.misses <= self._max_misses:
                survivors.append(tr)

        self._tracks = survivors

        # Yalnızca onaylı izleri döndür (tek karelik yanlış alarmlar elenir).
        active = [
            TrackedSignal(
                id=tr.id,
                params=replace(tr.params),
                first_seen=tr.first_seen,
                last_seen=tr.last_seen,
            )
            for tr in self._tracks
            if tr.hits >= self._min_hits
        ]
        active.sort(key=lambda t: t.params.center_freq)
        return active

    def reset(self) -> None:
        """Tüm izleri temizle (ör. frekans/SR değişince çağrılır)."""
        self._tracks = []
        self._next_id = 1
