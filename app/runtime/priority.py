"""CPU priority between the ASR worker and the vision worker.

In CPU fallback mode the ASR worker must not be starved by the vision models:
- AsrPriority.busy is set while the ASR worker transcribes a segment. The vision
  pipeline skips pose frames while it is set.
- The ASR worker thread runs at above normal and the vision worker thread at below normal
  OS thread priority (Windows SetThreadPriority, no effect elsewhere). ONNX Runtime uses
  the calling thread as one of its intra-op threads, so this reaches part of the model
  work.
"""

from __future__ import annotations

import os
import threading

THREAD_PRIORITY = {"below_normal": -1, "normal": 0, "above_normal": 1}


class AsrPriority:
    def __init__(self):
        self.busy = threading.Event()


def disable_power_throttling() -> bool:
    """Opt this process out of Windows power throttling (EcoQoS). True when it was applied.

    SetProcessInformation with ProcessPowerThrottling (4) and a
    PROCESS_POWER_THROTTLING_STATE of Version 1, ControlMask EXECUTION_SPEED (0x1) and
    IGNORE_TIMER_RESOLUTION (0x4), StateMask 0: Windows then does not run the process's
    threads in efficiency mode and honours its timer resolution. On 2026-09-25 the
    server's Whisper encoder ran about 6 times slower in some background runs than in
    others with the same load, which this setting is meant to prevent.
    """
    if os.name != "nt":
        return False
    import ctypes
    from ctypes import wintypes

    class State(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    state = State(1, 0x1 | 0x4, 0)
    return bool(kernel32.SetProcessInformation(kernel32.GetCurrentProcess(), 4, ctypes.byref(state),
                                               ctypes.sizeof(state)))


def set_current_thread_priority(level: str) -> bool:
    """Set the calling thread's OS priority. True when it was applied."""
    if os.name != "nt":
        return False
    import ctypes

    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentThread.restype = ctypes.c_void_p
    kernel32.SetThreadPriority.argtypes = [ctypes.c_void_p, ctypes.c_int]
    return bool(kernel32.SetThreadPriority(kernel32.GetCurrentThread(), THREAD_PRIORITY[level]))
