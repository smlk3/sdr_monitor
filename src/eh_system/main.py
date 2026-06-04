"""Giriş noktası: argparse ile SDR kaynağını seç ve MainWindow'u başlat.

Kullanım:
    uv run eh-system --mock     # donanımsız, sentetik IQ
    uv run eh-system            # gerçek LimeSDR
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

from .hal.mock_sdr import MockSDR
from .hal.motor import MockMotor, MotorBase
from .hal.sdr_base import SDRBase

# Varsayılan config yolu: paket kökünden iki üst (repo) / config / default.toml
_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "default.toml"


def _load_config(path: Path) -> dict:
    """TOML yapılandırmasını oku; yoksa boş sözlük döndür."""
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)


def _build_sdr(
    use_mock: bool, sdr_cfg: dict, doa_cfg: dict, mock_cfg: dict
) -> SDRBase:
    """Bayrağa göre MockSDR veya LimeSDR örneği üret."""
    freq = float(sdr_cfg.get("frequency", 433.0e6))
    rate = float(sdr_cfg.get("sample_rate", 10.0e6))
    gain = float(sdr_cfg.get("gain", 30.0))
    if use_mock:
        # DF için çift kanal ground-truth'u config'ten geçir.
        return MockSDR(
            frequency=freq,
            sample_rate=rate,
            gain_db=gain,
            df_sample_shift=int(mock_cfg.get("df_sample_shift", 3)),
            df_phase_offset_rad=float(mock_cfg.get("df_phase_offset_rad", 0.6)),
            df_true_doa_deg=float(mock_cfg.get("df_true_doa_deg", 25.0)),
            df_baseline_m=float(doa_cfg.get("baseline_m", 0.15)),
        )
    # Gerçek donanım: import burada yapılır ki --mock yolunda SoapySDR şart olmasın.
    from .hal.lime_sdr import LimeSDR

    channel = int(sdr_cfg.get("channel", 0))
    return LimeSDR(frequency=freq, sample_rate=rate, gain_db=gain, channel=channel)


def _build_motor(use_mock: bool, motor_cfg: dict) -> MotorBase:
    """Bayrağa göre MockMotor veya SerialMotor örneği üret."""
    if use_mock:
        return MockMotor()
    from .hal.motor import SerialMotor

    return SerialMotor(
        port=str(motor_cfg.get("port", "/dev/ttyUSB0")),
        baudrate=int(motor_cfg.get("baudrate", 115200)),
    )


def main(argv: list[str] | None = None) -> int:
    """Uygulamayı başlat. Dönüş: süreç çıkış kodu."""
    parser = argparse.ArgumentParser(
        prog="eh-system", description="LimeSDR EH Sistemi spektrum analizör"
    )
    parser.add_argument(
        "--mock", action="store_true", help="Donanımsız sentetik IQ üreteci kullan."
    )
    parser.add_argument(
        "--config", type=Path, default=_DEFAULT_CONFIG, help="config TOML yolu."
    )
    args = parser.parse_args(argv)

    cfg = _load_config(args.config)
    sdr_cfg = cfg.get("sdr", {})
    dsp_cfg = cfg.get("dsp", {})
    ui_cfg = cfg.get("ui", {})
    doa_cfg = cfg.get("doa", {})
    motor_cfg = cfg.get("motor", {})
    scan_cfg = cfg.get("scan", {})
    mock_cfg = cfg.get("mock", {})
    tx_cfg = cfg.get("tx", {})
    limits_cfg = cfg.get("limits", {})

    sdr = _build_sdr(args.mock, sdr_cfg, doa_cfg, mock_cfg)
    motor = _build_motor(args.mock, motor_cfg)

    # Qt importları burada (testlerin başsız import edebilmesi için tembel).
    from PySide6.QtWidgets import QApplication

    from .ui.main_window import MainWindow

    app = QApplication(sys.argv[:1])
    win = MainWindow(
        sdr,
        fft_size=int(dsp_cfg.get("fft_size", 1024)),
        read_size=int(sdr_cfg.get("read_size", 4096)),
        sample_rate=float(sdr_cfg.get("sample_rate", 10.0e6)),
        frequency=float(sdr_cfg.get("frequency", 433.0e6)),
        gain_db=float(sdr_cfg.get("gain", 30.0)),
        waterfall_rows=int(ui_cfg.get("waterfall_rows", 200)),
        window=str(dsp_cfg.get("window", "hann")),
        num_train=int(dsp_cfg.get("cfar_num_train", 16)),
        num_guard=int(dsp_cfg.get("cfar_num_guard", 4)),
        pfa_exp=int(dsp_cfg.get("cfar_pfa_exp", 4)),
        motor=motor,
        baseline_m=float(doa_cfg.get("baseline_m", 0.15)),
        scan_start_deg=float(scan_cfg.get("start_deg", 0.0)),
        scan_stop_deg=float(scan_cfg.get("stop_deg", 180.0)),
        scan_step_deg=float(scan_cfg.get("step_deg", 10.0)),
        power_limit_db=float(limits_cfg.get("tx_power_max_db", 0.0)),
        tx_gain_db=float(tx_cfg.get("gain_db", 40.0)),
    )
    win.show()
    win.start()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
