"""Cross-platform Intel NPU hardware-utilization sampling.

* **Windows** — reads the PDH performance counter ``\\GPU Engine(*)\\Utilization
  Percentage`` (the NPU is a compute-accelerator adapter that surfaces there, the
  same source Task Manager uses). The NPU's adapter LUID is auto-calibrated: the
  compute-engine LUID that is busiest while *we* are generating is the NPU.
* **Linux** — reads ``npu_busy_time_us`` from the ``intel_vpu`` accel sysfs node
  and differentiates it into a busy percentage.

Both expose ``sample() -> float | None`` returning 0..100 (or ``None`` if the
platform can't be read). A :func:`get_monitor` factory picks the right one.
"""

from __future__ import annotations

import glob
import platform
import re
import time


class _NullMonitor:
    available = False

    def sample(self, generating: bool = False) -> float | None:
        return None


# --------------------------------------------------------------------------- #
# Windows (PDH via ctypes)
# --------------------------------------------------------------------------- #
class _WindowsMonitor:
    available = True

    _PDH_FMT_DOUBLE = 0x00000200

    def __init__(self):
        import ctypes
        from ctypes import wintypes

        self._ok = False
        self.npu_luid: str | None = None
        try:
            self._pdh = ctypes.WinDLL("pdh.dll")
            self._ct = ctypes
            self._wt = wintypes

            class PDH_FMT_COUNTERVALUE(ctypes.Structure):
                _fields_ = [("CStatus", wintypes.DWORD), ("doubleValue", ctypes.c_double)]

            class PDH_FMT_COUNTERVALUE_ITEM_W(ctypes.Structure):
                _fields_ = [("szName", wintypes.LPWSTR), ("FmtValue", PDH_FMT_COUNTERVALUE)]

            self._ITEM = PDH_FMT_COUNTERVALUE_ITEM_W

            self._query = wintypes.HANDLE()
            if self._pdh.PdhOpenQueryW(None, 0, ctypes.byref(self._query)) != 0:
                return
            self._counter = wintypes.HANDLE()
            path = "\\GPU Engine(*)\\Utilization Percentage"
            if self._pdh.PdhAddEnglishCounterW(
                self._query, path, 0, ctypes.byref(self._counter)
            ) != 0:
                return
            # Prime the query (rate counters need a baseline collection).
            self._pdh.PdhCollectQueryData(self._query)
            self._ok = True
        except Exception:
            self._ok = False

    def _collect(self) -> dict[str, float]:
        """Return {luid: summed compute-engine utilization} for the latest sample."""
        ct, wt = self._ct, self._wt
        if self._pdh.PdhCollectQueryData(self._query) != 0:
            return {}
        size = wt.DWORD(0)
        count = wt.DWORD(0)
        PDH_MORE_DATA = 0x800007D2
        rc = self._pdh.PdhGetFormattedCounterArrayW(
            self._counter, self._PDH_FMT_DOUBLE, ct.byref(size), ct.byref(count), None
        )
        if rc != PDH_MORE_DATA or count.value == 0:
            return {}
        buf = ct.create_string_buffer(size.value)
        arr = ct.cast(buf, ct.POINTER(self._ITEM))
        rc = self._pdh.PdhGetFormattedCounterArrayW(
            self._counter, self._PDH_FMT_DOUBLE, ct.byref(size), ct.byref(count), arr
        )
        if rc != 0:
            return {}

        per_luid: dict[str, float] = {}
        for i in range(count.value):
            item = arr[i]
            name = item.szName or ""
            if "engtype_compute" not in name.lower():
                continue
            m = re.search(r"luid_[0-9a-fx]+_[0-9a-fx]+", name.lower())
            if not m:
                continue
            luid = m.group(0)
            val = float(item.FmtValue.doubleValue)
            per_luid[luid] = per_luid.get(luid, 0.0) + val
        return per_luid

    def sample(self, generating: bool = False) -> float | None:
        if not self._ok:
            return None
        per_luid = self._collect()
        if not per_luid:
            return 0.0
        # Calibrate: while we're generating, the busiest compute LUID is the NPU.
        busiest = max(per_luid, key=per_luid.get)
        if generating and per_luid[busiest] > 5.0:
            self.npu_luid = busiest
        target = self.npu_luid if self.npu_luid in per_luid else busiest
        return min(100.0, per_luid.get(target, 0.0))


# --------------------------------------------------------------------------- #
# Linux (intel_vpu accel sysfs)
# --------------------------------------------------------------------------- #
class _LinuxMonitor:
    def __init__(self):
        self.available = False
        self._node: str | None = None
        for cand in glob.glob("/sys/class/accel/accel*/device/npu_busy_time_us"):
            self._node = cand
            self.available = True
            break
        self._last_us: int | None = None
        self._last_t: float | None = None

    def sample(self, generating: bool = False) -> float | None:
        if not self.available or not self._node:
            return None
        try:
            with open(self._node) as f:
                busy_us = int(f.read().strip())
        except Exception:
            return None
        now = time.time()
        if self._last_us is None:
            self._last_us, self._last_t = busy_us, now
            return 0.0
        dt = now - self._last_t
        d_busy = (busy_us - self._last_us) / 1e6  # seconds busy
        self._last_us, self._last_t = busy_us, now
        if dt <= 0:
            return 0.0
        return max(0.0, min(100.0, (d_busy / dt) * 100.0))


def get_monitor():
    system = platform.system()
    try:
        if system == "Windows":
            m = _WindowsMonitor()
            return m if m.available else _NullMonitor()
        if system == "Linux":
            m = _LinuxMonitor()
            return m if m.available else _NullMonitor()
    except Exception:
        pass
    return _NullMonitor()
