"""compute device detection and selection utilities

provides functions to detect available compute devices (cpu cuda mps)
present an interactive selection prompt at startup and manage device
fallback logic
"""

import subprocess
import sys
from typing import List, Dict, Any

try:
    import torch
except ImportError:
    torch = None

def list_available_devices() -> List[Dict[str, Any]]:
    """detect and list all available compute devices on the system

    returns
        a list of dicts with keys index (int) type (str) name (str)
    """

    devices: List[Dict[str, Any]] = [
        {"index": 0, "type": "cpu", "name": "CPU (no GPU acceleration)"},
    ]
    devices = _add_cuda_devices(devices)
    devices = _add_mps_device(devices)
    return devices

def select_device(config_preference: str) -> Any:
    """select a compute device based on the config preference"""
    if torch is None:
        print("[WARNING] PyTorch not installed - defaulting to CPU.")
        print("          Run install.bat to enable GPU selection.")
        return _create_cpu_device()

    preference = config_preference.strip().lower()
    if preference == "auto":
        return _interactive_device_selection()
    return _direct_device_selection(preference)

def get_device_label(device: Any) -> str:
    """generate a human-readable on-screen label for the active device"""
    if torch is None or not hasattr(device, "type"):
        return "Device: CPU"

    dtype = str(device.type).lower()
    if dtype.startswith("cuda"):
        idx = device.index if device.index is not None else 0
        try:
            name = torch.cuda.get_device_name(idx)
            return f"GPU: {name} (CUDA:{idx})"
        except Exception:
            return f"GPU: CUDA:{idx}"
    if dtype == "mps":
        return "GPU: Apple Silicon (MPS)"
    return "Device: CPU"

def _query_nvidia_gpus() -> List[str]:
    """return nvidia gpu names via nvidia-smi when available"""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode != 0:
            return []
        names: List[str] = []
        for line in result.stdout.strip().splitlines():
            parts = line.split(",", 1)
            if len(parts) == 2:
                names.append(parts[1].strip())
        return names
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []

def _add_cuda_devices(devices: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """append all available nvidia cuda devices to the list"""
    if torch is not None and torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            try:
                name = torch.cuda.get_device_name(i)
            except Exception:
                name = f"CUDA device {i}"
            devices.append({
                "index": len(devices),
                "type": f"cuda:{i}",
                "name": f"NVIDIA {name} (CUDA:{i})",
            })
        return devices

    nvidia_names = _query_nvidia_gpus()
    if nvidia_names:
        for i, name in enumerate(nvidia_names):
            devices.append({
                "index": len(devices),
                "type": f"cuda:{i}",
                "name": f"NVIDIA {name} (CUDA:{i}) - reinstall PyTorch CUDA",
            })
    return devices

def _add_mps_device(devices: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """append apple silicon mps device if available"""
    if torch is None:
        return devices
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append({
            "index": len(devices),
            "type": "mps",
            "name": "Apple Silicon GPU (MPS)",
        })
    return devices

def _interactive_device_selection() -> Any:
    """print available devices and prompt the user to pick one"""
    devices = list_available_devices()

    print("\n+----------------------------------------------+")
    print("|    Hand AR Tracker -- GPU Selection          |")
    print("+----------------------------------------------+")
    for dev in devices:
        marker = "[*]" if dev["type"].startswith("cuda") else "   "
        line = f"| {marker} [{dev['index']}] {dev['name']}"
        print(f"{line:<48}|")
    print("+----------------------------------------------+")

    default_idx = 0
    for dev in devices:
        if dev["type"].startswith("cuda"):
            default_idx = dev["index"]
            break

    if not sys.stdin.isatty():
        selected = devices[default_idx]
        print(f"[INFO] Non-interactive mode - using: {selected['name']}")
        return _resolve_device(selected["type"])

    try:
        raw = input(f"Select device number [{default_idx}]: ").strip()
        chosen = int(raw) if raw else default_idx
        if 0 <= chosen < len(devices):
            selected = devices[chosen]
            print(f"[INFO] Using: {selected['name']}\n")
            return _resolve_device(selected["type"])
        print(f"[WARNING] Invalid choice, using default [{default_idx}].\n")
        return _resolve_device(devices[default_idx]["type"])
    except (ValueError, KeyboardInterrupt, EOFError):
        print(f"\n[WARNING] No valid input - using default [{default_idx}].\n")
        return _resolve_device(devices[default_idx]["type"])

def _resolve_device(device_type: str) -> Any:
    """return a torch device falling back to cpu when cuda is unavailable"""
    if device_type == "cpu":
        return _create_cpu_device()
    if device_type.startswith("cuda"):
        if torch is not None and _is_cuda_available(device_type):
            return torch.device(device_type)
        print(
            f"[WARNING] {device_type.upper()} selected but CUDA PyTorch is not "
            "available. Run install.bat to install CUDA PyTorch."
        )
        return _create_cpu_device()
    if device_type == "mps" and _is_cuda_available("mps"):
        return torch.device("mps")
    return _create_cpu_device()

def _direct_device_selection(preference: str) -> Any:
    """use a device specified directly in config"""
    if preference == "cpu":
        print("[INFO] Config set to CPU.")
        return _create_cpu_device()
    if _is_cuda_available(preference):
        print(f"[INFO] Config device: {preference.upper()}")
        return torch.device(preference)
    print(f"[WARNING] Device '{preference}' not available - falling back to CPU.")
    return _create_cpu_device()

def _is_cuda_available(device_type: str) -> bool:
    """check whether a given device string is usable"""
    if torch is None:
        return False
    if device_type.startswith("cuda"):
        if not torch.cuda.is_available():
            return False
        parts = device_type.split(":")
        if len(parts) == 2:
            try:
                return int(parts[1]) < torch.cuda.device_count()
            except ValueError:
                return False
        return True
    if device_type == "mps":
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    return False

def _create_cpu_device() -> Any:
    """return a cpu device (uses simplenamespace when torch is absent)"""
    if torch is not None:
        return torch.device("cpu")
    from types import SimpleNamespace
    return SimpleNamespace(type="cpu", index=None)
