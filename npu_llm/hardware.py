"""Intel NPU detection and property reporting (OpenVINO)."""

from __future__ import annotations

from dataclasses import dataclass

from rich.table import Table

from .console import console, error


def _core():
    import openvino

    return openvino.Core()


def has_npu() -> bool:
    try:
        return "NPU" in _core().available_devices
    except Exception:
        return False


def available_devices() -> list[str]:
    try:
        return list(_core().available_devices)
    except Exception:
        return []


def require_npu() -> None:
    """Exit with a helpful message if no NPU is available."""
    if not has_npu():
        error(
            "No NPU detected. Install the latest Intel NPU driver:\n"
            "  Windows: https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html\n"
            "  Linux:   https://github.com/intel/linux-npu-driver/releases/"
        )
        raise SystemExit(1)


@dataclass
class NpuInfo:
    name: str = "Unknown NPU"
    tops: float | None = None
    total_mem_bytes: int | None = None
    driver: str | None = None


_PROPS = [
    ("Name", "FULL_DEVICE_NAME"),
    ("TOPS", "DEVICE_GOPS"),
    ("NPU memory", "NPU_DEVICE_TOTAL_MEM_SIZE"),
    ("Driver", "NPU_DRIVER_VERSION"),
    ("Capabilities", "OPTIMIZATION_CAPABILITIES"),
    ("Precision", "INFERENCE_PRECISION_HINT"),
]


def _get(core, prop):
    try:
        return core.get_property("NPU", prop)
    except Exception:
        return None


def get_info() -> NpuInfo:
    """Lightweight, machine-readable NPU summary for the dashboard."""
    if not has_npu():
        return NpuInfo()
    core = _core()
    gops = _get(core, "DEVICE_GOPS")
    tops = None
    if isinstance(gops, dict) and gops:
        # DEVICE_GOPS maps precision -> GOPS; report the max as TOPS.
        try:
            tops = max(float(v) for v in gops.values()) / 1000.0
        except Exception:
            tops = None
    elif gops:
        try:
            tops = float(gops) / 1000.0
        except Exception:
            tops = None
    mem = _get(core, "NPU_DEVICE_TOTAL_MEM_SIZE")
    return NpuInfo(
        name=str(_get(core, "FULL_DEVICE_NAME") or "Intel NPU"),
        tops=tops,
        total_mem_bytes=int(mem) if isinstance(mem, (int, float)) else None,
        driver=str(_get(core, "NPU_DRIVER_VERSION")) if _get(core, "NPU_DRIVER_VERSION") else None,
    )


def print_info() -> None:
    """Pretty NPU summary table for ``--info`` and the ``npuinfo`` chat command."""
    if not has_npu():
        error("No NPU detected.")
        devs = available_devices()
        if devs:
            console.print(f"[muted]Available OpenVINO devices: {', '.join(devs)}[/muted]")
        return

    core = _core()
    table = Table(title="Intel NPU", title_style="info", show_header=False, box=None)
    table.add_column("prop", style="muted")
    table.add_column("value")
    for label, prop in _PROPS:
        val = _get(core, prop)
        if val is None:
            continue
        if prop == "DEVICE_GOPS" and isinstance(val, dict):
            try:
                val = f"{max(float(v) for v in val.values()) / 1000.0:.1f} TOPS"
            except Exception:
                val = str(val)
        elif prop == "NPU_DEVICE_TOTAL_MEM_SIZE" and isinstance(val, (int, float)):
            val = f"{val / (1024 ** 3):.1f} GiB"
        table.add_row(label, str(val))
    console.print(table)
