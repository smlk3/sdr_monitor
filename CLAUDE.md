# Proje: LimeSDR Tabanlı Elektronik Harp Sistemi Arayüzü

## Ne yapıyoruz
TEKNOFEST 2026 Elektronik Harp Yarışması için LimeSDR USB tabanlı bütünleşik bir EH sisteminin kontrol ve analiz yazılımı. Yazılım iki görev ailesini tek arayüzden yönetir:
- Elektronik Destek (ED): sinyal tespiti, parametre çıkarımı, yön bulma (DF)
- Elektronik Taarruz (ET): sürekli karıştırma, analog telsiz aldatma

Donanım: LimeSDR USB (LMS7002M), 2 RX + 2 TX kanal, motorlu tripod (360° tarama), Log-Periyodik anten (400-700 MHz) + Vivaldi anten (1000-3800 MHz), harici LNA/PA, RF switch, motor için Arduino/STM32 kontrolcü (USB-seri).

## Teknoloji yığını (kesin, değiştirme)
- OS hedefi: Ubuntu 24.04 LTS
- Python: 3.13 (standart build)
- Paket yönetimi: uv (pyproject.toml)
- SDR sürücü: SoapySDR (driver='lime') + LimeSuite
- UI: PySide6 (Qt for Python, LGPL) + Qt Designer (.ui dosyaları)
- Grafik: pyqtgraph (real-time spektrum + ImageItem waterfall)
- Sayısal: numpy, scipy
- Veri kaydı: sigmf
- Motor seri haberleşme: pyserial
- Yapılandırma: tomllib (okuma) / tomli-w (yazma)
- Test: pytest
- Lint/format: ruff

## Mimari kuralları (KATI)
1. Üç katman ayrık olmalı, birbirine sızmamalı:
   - `hal/` (Hardware Abstraction Layer): SDR, motor, switch kontrolü. UI bilmez.
   - `dsp/`: FFT, CFAR, parametre çıkarımı, DF, modülasyon. Qt/UI bilmez, saf numpy/scipy.
   - `ui/`: PySide6 pencereleri, widget'lar. DSP'yi doğrudan çağırmaz, worker üzerinden.
2. Tüm ağır iş (SDR I/O, FFT, CFAR) ana GUI thread'inde ASLA çalışmaz. QThread worker'lar kullan.
3. Worker → UI iletişimi yalnızca Qt signal/slot ile. UI → Worker iletişimi thread-safe (queue.Queue veya QMetaObject.invokeMethod). Paylaşılan numpy array'ler kopyalanarak geçer.
4. Donanım erişimi yokken geliştirme yapılabilmeli: HAL bir `MockSDR` sağlamalı (sentetik IQ üretir). `--mock` bayrağıyla gerçek donanım olmadan tüm UI çalışmalı.
5. Her donanım komutu try/except ile sarılmalı; hata UI'ya signal ile bildirilmeli, çökme olmamalı.

## Dizin yapısı
```
eh_system/
├── pyproject.toml
├── CLAUDE.md
├── README.md
├── config/
│   └── default.toml
├── src/eh_system/
│   ├── __init__.py
│   ├── main.py              # giriş noktası, argparse (--mock)
│   ├── hal/
│   │   ├── sdr_base.py      # soyut arayüz
│   │   ├── lime_sdr.py      # gerçek LimeSDR
│   │   ├── mock_sdr.py      # sentetik IQ
│   │   ├── motor.py         # seri port motor kontrolü
│   │   └── rf_switch.py     # switch + guard time mantığı
│   ├── dsp/
│   │   ├── spectrum.py      # pencereleme, FFT, PSD
│   │   ├── cfar.py          # CA-CFAR (vektörize), OS-CFAR
│   │   ├── params.py        # merkez f, BW, güç, tip
│   │   ├── doa.py           # faz interferometri
│   │   ├── calibration.py   # faz/zaman kalibrasyonu
│   │   ├── jammer.py        # baraj/çoklu/sweep üretimi
│   │   └── modulator.py     # NBFM, CTCSS, pre-emphasis
│   ├── workers/
│   │   ├── sdr_worker.py    # RX akışı + FFT thread
│   │   ├── tx_worker.py     # TX akışı thread
│   │   └── scan_worker.py   # motor tarama + DF thread
│   ├── ui/
│   │   ├── main_window.py
│   │   ├── widgets/
│   │   └── designer/        # .ui dosyaları
│   └── recording/
│       └── sigmf_writer.py
└── tests/
    ├── test_cfar.py
    ├── test_modulator.py
    └── test_doa.py
```

## Donanım tuzakları (kodda mutlaka dikkat et)
- LimeSDR stream formatı CS16 iste, içeride float32'ye çevir (USB throughput 2x).
- FFT öncesi mutlaka pencereleme (Hann varsayılan).
- DC offset: TX hedefini tam merkeze değil ~200 kHz kaydırarak ayarla.
- Faz kalibrasyonu HER LO retune sonrası geçersiz olur → calibration_required bayrağı.
- DF taraması stop-and-measure: dön → dur → 50 ms bekle → ölç. Hareket halinde ölçme.
- RF switch sıralaması ZORUNLU: RX durdur → switch → ≥10 µs guard → TX. Bu sırayı bozmak LNA yakar.
- Sürdürülebilir IBW 30 MHz hedefle (60 MHz reklamdır, USB throughput tutmaz).

## Emniyet kuralları (KATI)
- TX gücü yazılımsal üst limitle sınırlı; limit aşılamaz.
- Acil durdur (emergency stop): tek çağrıda TX kes + motor durdur, <500 ms.
- RX portuna TX yönlendiren komut dizisi kod seviyesinde engellenmeli (assert/guard).
- TX başlatmadan önce operatör onayı + yasal uyarı.

## Kodlama standardı
- PEP 8, ruff ile format. Tip ipuçları (type hints) zorunlu.
- Fonksiyon ve sınıflara kısa Türkçe docstring.
- Sihirli sayı yok; sabitler config veya modül başında named constant.
- Her dsp/ fonksiyonu saf (yan etkisiz) ve pytest ile test edilebilir olmalı.
- Commit'ler küçük ve anlamlı; her faz kendi branch'inde.

## Çalıştırma
- Kurulum: `uv sync`
- Çalıştırma (donanımsız): `uv run eh-system --mock`
- Çalıştırma (donanımlı): `uv run eh-system`
- Test: `uv run pytest`
- Lint: `uv run ruff check .`
