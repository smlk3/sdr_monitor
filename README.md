# EH Sistemi — LimeSDR Tabanlı Elektronik Harp Arayüzü

TEKNOFEST 2026 Elektronik Harp Yarışması için LimeSDR USB tabanlı bütünleşik EH
sisteminin kontrol ve analiz yazılımı.

Bu sürüm **Faz 1** iskeletini içerir: çalışan bir gerçek zamanlı spektrum
analizör + kayan waterfall. CFAR, yön bulma (DF), karıştırma, aldatma ve motor
kontrolü sonraki fazlarda gelecektir.

## Mimari

Üç katman birbirinden ayrıktır (bkz. `CLAUDE.md`):

- `hal/` — Donanım soyutlama (SDR, ileride motor/switch). UI bilmez.
- `dsp/` — Saf numpy/scipy sinyal işleme (pencereleme, PSD). Qt bilmez.
- `ui/` — PySide6 arayüz; ağır işi `workers/` (QThread) üzerinden yürütür.

Ağır iş (SDR I/O, FFT) asla GUI thread'inde çalışmaz. Worker→UI iletişimi Qt
signal/slot, UI→Worker iletişimi thread-safe kuyruk (`queue.Queue`) iledir.

## Gereksinimler

- Ubuntu 24.04 LTS (hedef), Python 3.13
- [uv](https://docs.astral.sh/uv/) paket yöneticisi
- Gerçek donanım için: LimeSuite + SoapySDR (`sudo apt install soapysdr-tools
  python3-soapysdr limesuite`). Donanımsız geliştirme için gerekmez.

## Kurulum

```bash
uv sync
```

## Çalıştırma

Donanımsız (sentetik IQ — birkaç CW tonu + dar bant FM gürültüsü + AWGN):

```bash
uv run eh-system --mock
```

Gerçek LimeSDR ile:

```bash
uv run eh-system
```

Başsız/sunucu ortamında smoke test (Qt offscreen):

```bash
QT_QPA_PLATFORM=offscreen uv run eh-system --mock
```

## Kullanım

- **Frekans (MHz):** merkez frekansı; Enter'a basınca uygulanır.
- **SR:** örnekleme hızı (2–30 MHz). Sürdürülebilir IBW ~30 MHz hedeflenir.
- **FFT:** FFT boyutu (512–4096); waterfall tamponu otomatik yeniden boyutlanır.
- **Kazanç:** RX kazancı (dB).
- **Başlat / Durdur:** RX worker thread'ini yönetir.
- **Durum çubuğu:** bağlantı durumu, FPS ve overflow sayacı.

## Test ve Lint

```bash
uv run pytest        # birim testler (dsp.spectrum)
uv run ruff check .  # lint
```

## Yapılandırma

`config/default.toml` — varsayılan frekans, örnekleme hızı, FFT boyutu, kazanç,
waterfall satır sayısı vb. `--config <yol>` ile farklı bir dosya verilebilir.

## Dizin yapısı

```
src/eh_system/
├── main.py              # giriş noktası (argparse --mock)
├── hal/                 # SDRBase, MockSDR, LimeSDR
├── dsp/                 # spectrum (pencereleme, PSD)
├── workers/             # SDRWorker (QThread RX+FFT)
└── ui/                  # MainWindow (spektrum + waterfall)
config/default.toml
tests/test_spectrum.py
```
